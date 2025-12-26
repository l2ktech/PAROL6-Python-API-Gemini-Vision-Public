#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM + MuJoCo 物理抓取仿真
========================

使用 VLM（视觉语言模型）通过自然语言控制 MuJoCo 仿真中的 PAROL6 机械臂
完成真实物理抓取动作（夹爪闭合、物体提升、放置等）

功能:
1. 自然语言理解（VLM）
2. 物理仿真抓取（MuJoCo）
3. 夹爪控制、物体抓取、放置动作
4. 场景包含桌子、红色杯子、蓝色方块

使用:
    交互模式: python3 26_vlm_mujoco_grasp.py
    测试模式: python3 26_vlm_mujoco_grasp.py --test
    无头测试: python3 26_vlm_mujoco_grasp.py --test --headless

作者: wzy
日期: 2025-12-23
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

# ==================== 路径配置 ====================

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "12-unified-control/examples"))

# 导入依赖
try:
    from openai import OpenAI
except ImportError:
    print("请安装 openai 库: pip install openai")
    sys.exit(1)

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("请安装 mujoco 库: pip install mujoco")
    sys.exit(1)


# ==================== VLM 配置 ====================

def load_config() -> Dict[str, str]:
    """加载配置：环境变量 > 配置文件 > 默认值"""
    config = {
        "api_base": "http://localhost:8317/v1",
        "api_key": "",
        "model": "gpt-5.1"
    }
    
    # 1. 尝试读取配置文件
    config_paths = [
        Path("vlm_config.json"),
        Path.home() / ".config" / "vlm" / "config.json",
        Path(__file__).parent / "vlm_config.json"
    ]
    
    for p in config_paths:
        if p.exists():
            try:
                import json
                file_config = json.loads(p.read_text())
                config.update(file_config)
                print(f"[配置] 已加载配置文件: {p}")
                break
            except Exception as e:
                print(f"[警告] 配置文件加载失败 {p}: {e}")

    # 2. 环境变量覆盖
    if os.environ.get("VLM_API_BASE"):
        config["api_base"] = os.environ["VLM_API_BASE"]
    if os.environ.get("VLM_API_KEY"):
        config["api_key"] = os.environ["VLM_API_KEY"]
    if os.environ.get("VLM_MODEL"):
        config["model"] = os.environ["VLM_MODEL"]
        
    return config

# 加载配置
CONFIG = load_config()
API_BASE = CONFIG["api_base"]
API_KEY = CONFIG["api_key"]
MODEL = CONFIG["model"]

# 检查 API 密钥
if not API_KEY:
    # 兼容旧的密钥文件检查
    old_key_file = Path.home() / ".config" / "vlm" / "api_key"
    if old_key_file.exists():
        API_KEY = old_key_file.read_text().strip()
    else:
        print("[警告] 未设置 API Key (环境变量或配置文件)")
        API_KEY = "test-key-placeholder"


# ==================== 数据类 ====================

@dataclass
class GraspResult:
    """抓取结果"""
    success: bool
    object_lifted: bool
    grasp_force: float
    final_object_pos: np.ndarray
    message: str


# ==================== IK 求解器 ====================

class SimpleIKSolver:
    """
    简化的逆运动学求解器
    
    使用雅可比矩阵 + 阻尼最小二乘法将笛卡尔位置转换为关节角度
    """
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        # 获取末端执行器 site ID
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
        if self.ee_site_id < 0:
            print("[警告] 未找到 end_effector site，使用默认位置")
            self.ee_site_id = 0
    
    def solve(self, target_pos: np.ndarray, current_q: np.ndarray = None, 
              max_iter: int = 100, tol: float = 0.02) -> Tuple[np.ndarray, float]:
        """
        求解逆运动学
        
        参数:
            target_pos: 目标位置 [x, y, z]
            current_q: 当前关节角度，None 则使用零位
            max_iter: 最大迭代次数
            tol: 收敛阈值
        
        返回:
            (关节角度, 最终距离)
        """
        if current_q is None:
            current_q = np.zeros(6)
        
        q = current_q.copy()
        dist = float('inf')
        
        for iteration in range(max_iter):
            # 设置关节并更新运动学
            self.data.qpos[:6] = q
            mujoco.mj_forward(self.model, self.data)
            
            # 计算误差
            ee_pos = self.data.site_xpos[self.ee_site_id]
            error = target_pos - ee_pos
            dist = np.linalg.norm(error)
            
            if dist < tol:
                return q, dist
            
            # 获取雅可比矩阵
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site_id)
            
            # 阻尼最小二乘法
            J = jacp[:, :6]
            damping = 0.1
            JJT = J @ J.T + damping**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)
            
            # 更新关节角度
            q = q + 0.5 * dq
            
            # 限制关节范围（根据 PAROL6 实际范围）
            q = np.clip(q, 
                       [-1.7, -0.98, -2.0, -2.0, -2.1, -6.28],
                       [1.7, 1.0, 1.3, 2.0, 2.1, 6.28])
        
        return q, dist


# ==================== VLM 客户端 ====================

class VLMClient:
    """VLM API 客户端"""
    
    def __init__(self):
        self.client = OpenAI(base_url=API_BASE, api_key=API_KEY)
        self.model = MODEL
        print(f"[VLM] 初始化: {API_BASE}, 模型: {self.model}")
    
    def chat(self, message: str) -> str:
        """发送请求"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": message}],
                max_tokens=512,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[错误] {e}"
    
    def parse_command(self, user_input: str) -> Dict[str, Any]:
        """
        解析用户指令为机器人命令
        
        支持的命令:
        - home: 回零位置
        - pick: 抓取物体 (target: 物体名称或颜色)
        - place: 放置物体 (location: 位置)
        - wave: 挥手动作
        - status: 查看状态
        """
        prompt = f"""你是 PAROL6 机械臂控制助手。场景中有一个红色杯子(target_cup)和一个蓝色方块(target_block)。

用户说: "{user_input}"

可用命令:
- home: 回零位置
- pick: 抓取物体 (参数 target: "target_cup" 或 "target_block")
- place: 放置物体 (参数 location: "left", "right", "center", "front")
- wave: 挥手动作
- status: 查看当前状态

颜色映射:
- 红色杯子 → target_cup
- 蓝色方块 → target_block

请分析用户意图，返回 JSON 格式:
{{"command": "命令名", "params": {{"参数名": "值"}}}}

只返回 JSON，不要其他内容。"""

        response = self.chat(prompt)
        
        try:
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            return json.loads(response)
        except:
            return {"command": "chat", "params": {"response": response}}


# ==================== MuJoCo 物理抓取仿真器 ====================

class GraspSimulator:
    """PAROL6 物理抓取仿真器"""
    
    def __init__(self):
        self.model = None
        self.data = None
        self.viewer = None
        
        # 关节索引
        self.arm_joint_ids = []
        self.gripper_joint_ids = []
        
        # 夹爪状态
        self.gripper_open = True
        self.holding_object = None
        
        # 创建场景
        self._create_scene()
        self._init_joint_indices()
        
        # 初始化 IK 求解器
        self.ik_solver = SimpleIKSolver(self.model, self.data)
        
        # 初始化仿真状态，确保物体位置正确
        mujoco.mj_forward(self.model, self.data)
        self.step(100)  # 让物体稳定
        
        print("[MuJoCo] 物理抓取仿真器初始化完成（包含 IK 求解器）")
    
    def _create_scene(self):
        """创建带物体的抓取场景 - 使用验证过的 parol6_fixed.xml 模型"""
        import xml.etree.ElementTree as ET
        
        # 基础模型路径：优先使用环境变量，否则使用相对路径
        model_path_env = os.environ.get("PAROL6_MODEL_PATH")
        if model_path_env:
            base_model_path = Path(model_path_env)
        else:
            # 相对于项目根目录
            base_model_path = PROJECT_ROOT / "11.1-lerobot-mujoco-smolvla" / "asset" / "parol6" / "parol6_fixed.xml"
        
        if not base_model_path.exists():
            raise FileNotFoundError(f"PAROL6 模型文件不存在: {base_model_path}\n"
                                    f"请设置 PAROL6_MODEL_PATH 环境变量或确保文件存在")
        
        # 读取原始 XML
        tree = ET.parse(base_model_path)
        root = tree.getroot()
        
        # 找到 worldbody
        worldbody = root.find('worldbody')
        
        # 添加红色杯子（在机械臂可达范围内）
        # parol6_fixed.xml 中桌面约在 Z=0.87m，末端可达 Z 约 0.9-1.2m
        cup_body = ET.SubElement(worldbody, 'body')
        cup_body.set('name', 'target_cup')
        cup_body.set('pos', '0.12 0.05 0.92')  # 桌面上方
        
        cup_joint = ET.SubElement(cup_body, 'freejoint')
        cup_joint.set('name', 'cup_joint')
        
        cup_geom = ET.SubElement(cup_body, 'geom')
        cup_geom.set('name', 'cup_body')
        cup_geom.set('type', 'cylinder')
        cup_geom.set('size', '0.025 0.04')  # 稍大一点，更容易抓取
        cup_geom.set('rgba', '1.0 0.3 0.3 1')
        cup_geom.set('density', '500')  # 增加密度，更稳定
        cup_geom.set('friction', '1.5 0.1 0.1')
        
        # 添加蓝色方块
        block_body = ET.SubElement(worldbody, 'body')
        block_body.set('name', 'target_block')
        block_body.set('pos', '0.08 -0.08 0.91')  # 桌面上方
        
        block_joint = ET.SubElement(block_body, 'freejoint')
        block_joint.set('name', 'block_joint')
        
        block_geom = ET.SubElement(block_body, 'geom')
        block_geom.set('name', 'block_body')
        block_geom.set('type', 'box')
        block_geom.set('size', '0.02 0.02 0.02')  # 稍大一点
        block_geom.set('rgba', '0.3 0.3 1.0 1')
        block_geom.set('density', '800')  # 增加密度，更稳定
        block_geom.set('friction', '1.5 0.1 0.1')
        
        # 保存到与原模型相同的目录（mesh 使用相对路径）
        temp_path = base_model_path.parent / "parol6_grasp_scene.xml"
        tree.write(str(temp_path), encoding='unicode')
        
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(str(temp_path))
        self.data = mujoco.MjData(self.model)
        print("[MuJoCo] 抓取场景创建完成（使用 parol6_fixed.xml 模型）")
    
    def _init_joint_indices(self):
        """初始化关节索引"""
        # 使用执行器索引，与 18_mujoco_grasp_simulation.py 一致
        # 机械臂关节 0-5，夹爪关节 6-7
        self.arm_joint_ids = list(range(6))
        self.gripper_joint_ids = [6, 7]
        
        print(f"[MuJoCo] 机械臂关节: {len(self.arm_joint_ids)}, 夹爪关节: {len(self.gripper_joint_ids)}")
    
    def reset(self):
        """重置仿真"""
        mujoco.mj_resetData(self.model, self.data)
        self.gripper_open = True
        self.holding_object = None
    
    def set_arm_joints(self, angles_rad: List[float]):
        """设置机械臂关节角度"""
        for i, angle in enumerate(angles_rad[:6]):
            self.data.ctrl[i] = angle
    
    def set_gripper(self, open_width: float):
        """
        设置夹爪开度
        
        参数:
            open_width: 0=闭合, 0.04=完全打开
        """
        self.data.ctrl[6] = open_width  # 左手指
        self.data.ctrl[7] = open_width  # 右手指
        self.gripper_open = open_width > 0.01
    
    def get_object_position(self, body_name: str) -> np.ndarray:
        """获取物体位置"""
        # 先更新运动学以确保位置是最新的
        mujoco.mj_forward(self.model, self.data)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id].copy()
    
    def step(self, n_steps: int = 1):
        """仿真步进"""
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
    
    def home(self) -> bool:
        """回零位置"""
        print("[MuJoCo] 执行: 回零")
        self.set_arm_joints([0, 0, 0, 0, 0, 0])
        self.set_gripper(0.02)  # 张开夹爪
        self.step(500)
        print("[MuJoCo] 回零完成")
        return True
    
    def pick(self, target: str = "target_cup") -> GraspResult:
        """
        执行物理抓取
        
        参数:
            target: "target_cup" 或 "target_block"
        """
        print(f"\n[MuJoCo] 执行: 抓取 {target}")
        
        # 获取目标初始位置
        target_pos = self.get_object_position(target)
        initial_z = target_pos[2]
        print(f"  目标位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        # ===== 使用 IK 求解器动态计算轨迹 =====
        
        # 1. 待机位置
        print("  阶段1: 待机位置")
        self.set_arm_joints([0, 0, 0, 0, 0, 0])
        self.set_gripper(0.03)  # 张开夹爪
        self.step(400)
        self._update_viewer()
        
        # 2. 计算预抓取位置（目标上方 10cm）
        pre_grasp_pos = target_pos.copy()
        pre_grasp_pos[2] += 0.10  # 上方 10cm
        print(f"  阶段2: IK 求解预抓取位置 ({pre_grasp_pos[0]:.3f}, {pre_grasp_pos[1]:.3f}, {pre_grasp_pos[2]:.3f})")
        
        pre_q, pre_dist = self.ik_solver.solve(pre_grasp_pos)
        if pre_dist > 0.05:
            print(f"    [错误] IK 求解失败，误差: {pre_dist:.3f}m > 0.05m")
            print(f"    目标位置可能超出可达范围，安全回零")
            self.home()  # 安全回零而不是使用固定轨迹
            return GraspResult(
                success=False,
                object_lifted=False,
                grasp_force=0.0,
                final_object_pos=target_pos,
                message=f"✗ IK 求解失败，目标位置不可达 (误差 {pre_dist:.3f}m)"
            )
        else:
            print(f"    IK 成功，误差: {pre_dist:.3f}m")
        
        self.set_arm_joints(pre_q.tolist())
        self.step(500)
        self._update_viewer()
        
        # 3. 计算抓取位置（物体中心附近）
        grasp_pos = target_pos.copy()
        grasp_pos[2] -= 0.02  # 略低于物体中心，确保夹爪接触
        print(f"  阶段3: IK 求解抓取位置 ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
        
        grasp_q, grasp_dist = self.ik_solver.solve(grasp_pos, pre_q)
        if grasp_dist > 0.05:
            print(f"    [错误] IK 求解失败，误差: {grasp_dist:.3f}m > 0.05m")
            self.home()
            return GraspResult(
                success=False,
                object_lifted=False,
                grasp_force=0.0,
                final_object_pos=target_pos,
                message=f"✗ 抓取位置 IK 失败 (误差 {grasp_dist:.3f}m)"
            )
        else:
            print(f"    IK 成功，误差: {grasp_dist:.3f}m")
        
        self.set_arm_joints(grasp_q.tolist())
        self.step(500)
        self._update_viewer()
        
        # 4. 闭合夹爪
        print("  阶段4: 闭合夹爪")
        self.set_gripper(0.0)
        self.step(400)
        self._update_viewer()
        
        # 5. 提升物体（IK 求解提升位置）
        lift_pos = grasp_pos.copy()
        lift_pos[2] += 0.12  # 提升 12cm
        print(f"  阶段5: IK 求解提升位置 ({lift_pos[0]:.3f}, {lift_pos[1]:.3f}, {lift_pos[2]:.3f})")
        
        lift_q, lift_dist = self.ik_solver.solve(lift_pos, grasp_q)
        if lift_dist > 0.05:
            print(f"    [警告] 提升位置 IK 误差较大: {lift_dist:.3f}m，尝试使用当前关节继续")
            # 提升阶段夹爪已闭合，即使 IK 不精确也尝试提升
            # 使用一个保守的提升动作：在当前位置基础上调整
            lift_q = grasp_q.copy()
            lift_q[1] -= 0.3  # 肩关节抬起一些
        else:
            print(f"    IK 成功，误差: {lift_dist:.3f}m")
        
        self.set_arm_joints(lift_q.tolist())
        self.step(600)
        self._update_viewer()
        
        # 6. 检查结果
        final_pos = self.get_object_position(target)
        lift_height = final_pos[2] - initial_z
        lifted = lift_height > 0.03
        
        if lifted:
            self.holding_object = target
            msg = f"✓ 抓取成功！提升 {lift_height:.3f}m"
        else:
            msg = f"✗ 抓取失败，提升 {lift_height:.3f}m"
        
        print(f"  结果: {msg}")
        
        return GraspResult(
            success=lifted,
            object_lifted=lifted,
            grasp_force=0.0,
            final_object_pos=final_pos,
            message=msg
        )
    
    def place(self, location: str = "left") -> bool:
        """
        放置物体
        
        参数:
            location: "left", "right", "center", "front"
        """
        print(f"\n[MuJoCo] 执行: 放置到 {location}")
        
        if self.holding_object is None:
            print("  没有抓取的物体！")
            return False
        
        # 根据位置设置目标关节角度
        location_joints = {
            "left": [-0.8, -0.4, 0.4, 0, 0, 0],    # 左侧
            "right": [0.8, -0.4, 0.4, 0, 0, 0],    # 右侧
            "center": [0, -0.4, 0.4, 0, 0, 0],     # 中间
            "front": [0, -0.6, 0.6, 0, 0, 0],      # 前方
        }
        
        joints = location_joints.get(location, location_joints["center"])
        
        # 1. 移动到放置位置
        print(f"  阶段1: 移动到 {location}")
        self.set_arm_joints(joints)
        self.step(500)
        self._update_viewer()
        
        # 2. 下降
        print("  阶段2: 下降")
        joints[1] -= 0.2  # 降低 L2
        joints[2] += 0.2  # 调整 L3
        self.set_arm_joints(joints)
        self.step(400)
        self._update_viewer()
        
        # 3. 打开夹爪释放物体
        print("  阶段3: 释放物体")
        self.set_gripper(0.025)
        self.step(300)
        self._update_viewer()
        
        # 4. 缩回
        print("  阶段4: 缩回")
        self.set_arm_joints([0, -0.3, 0.3, 0, 0, 0])
        self.step(400)
        self._update_viewer()
        
        self.holding_object = None
        print("  放置完成")
        return True
    
    def wave(self) -> bool:
        """挥手动作"""
        print("[MuJoCo] 执行: 挥手")
        
        # 举起手臂
        self.set_arm_joints([0, -0.5, 0.5, 0, 0.5, 0])
        self.step(300)
        self._update_viewer()
        
        # 挥手动作
        for i in range(3):
            self.set_arm_joints([0, -0.5, 0.5, 0, 0.5, 0.5])
            self.step(150)
            self._update_viewer()
            
            self.set_arm_joints([0, -0.5, 0.5, 0, 0.5, -0.5])
            self.step(150)
            self._update_viewer()
        
        # 恢复
        self.set_arm_joints([0, 0, 0, 0, 0, 0])
        self.step(300)
        self._update_viewer()
        
        print("[MuJoCo] 挥手完成")
        return True
    
    def status(self) -> str:
        """获取当前状态"""
        cup_pos = self.get_object_position("target_cup")
        block_pos = self.get_object_position("target_block")
        
        status = f"""
当前状态:
  夹爪: {'打开' if self.gripper_open else '闭合'}
  抓取物体: {self.holding_object or '无'}
  红色杯子位置: [{cup_pos[0]:.3f}, {cup_pos[1]:.3f}, {cup_pos[2]:.3f}]
  蓝色方块位置: [{block_pos[0]:.3f}, {block_pos[1]:.3f}, {block_pos[2]:.3f}]
"""
        return status
    
    def start_viewer(self):
        """启动可视化窗口"""
        print("[MuJoCo] 启动可视化...")
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self.viewer
    
    def _update_viewer(self):
        """更新可视化"""
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()


# ==================== VLM 抓取控制器 ====================

class VLMGraspController:
    """VLM + 物理抓取控制器"""
    
    def __init__(self):
        self.vlm = VLMClient()
        self.sim = GraspSimulator()
        
        # 命令映射
        self.commands = {
            "home": self._cmd_home,
            "pick": self._cmd_pick,
            "place": self._cmd_place,
            "wave": self._cmd_wave,
            "status": self._cmd_status,
        }
        
        print("[Controller] VLM + 物理抓取控制器初始化完成")
    
    def _cmd_home(self, params: dict) -> str:
        """回零命令"""
        self.sim.home()
        return "✓ 已回到初始位置"
    
    def _cmd_pick(self, params: dict) -> str:
        """抓取命令"""
        target = params.get("target", "target_cup")
        
        # 支持中文目标名称
        target_map = {
            "红色杯子": "target_cup",
            "杯子": "target_cup",
            "cup": "target_cup",
            "蓝色方块": "target_block",
            "方块": "target_block",
            "block": "target_block",
        }
        target = target_map.get(target, target)
        
        result = self.sim.pick(target)
        return result.message
    
    def _cmd_place(self, params: dict) -> str:
        """放置命令"""
        location = params.get("location", "left")
        
        # 支持中文位置名称
        location_map = {
            "左边": "left",
            "左": "left",
            "右边": "right",
            "右": "right",
            "中间": "center",
            "前面": "front",
            "前": "front",
        }
        location = location_map.get(location, location)
        
        success = self.sim.place(location)
        return "✓ 放置完成" if success else "✗ 放置失败（没有抓取的物体）"
    
    def _cmd_wave(self, params: dict) -> str:
        """挥手命令"""
        self.sim.wave()
        return "✓ 挥手完成"
    
    def _cmd_status(self, params: dict) -> str:
        """状态命令"""
        return self.sim.status()
    
    def process(self, user_input: str) -> str:
        """处理用户自然语言输入"""
        print(f"\n[用户] {user_input}")
        
        # VLM 解析
        result = self.vlm.parse_command(user_input)
        cmd = result.get("command", "unknown")
        params = result.get("params", {})
        
        print(f"[VLM] 解析: command={cmd}, params={params}")
        
        # 执行命令
        if cmd in self.commands:
            try:
                return self.commands[cmd](params)
            except Exception as e:
                return f"✗ 执行失败: {e}"
        elif cmd == "chat":
            return params.get("response", "我理解了你的问题")
        else:
            return f"? 未知命令: {cmd}"


# ==================== 主程序 ====================

def interactive_mode():
    """交互模式"""
    print("=" * 60)
    print("VLM + MuJoCo 物理抓取仿真")
    print("=" * 60)
    print("场景: 桌上有红色杯子和蓝色方块")
    print("示例指令:")
    print("  '抓取红色杯子'")
    print("  '把它放到左边'")
    print("  '抓取蓝色方块'")
    print("  '回到初始位置'")
    print("  '查看状态'")
    print("\n输入 'view' 启动可视化, 'quit' 退出")
    print()
    
    controller = VLMGraspController()
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("再见！")
                break
            
            if user_input.lower() == "view":
                controller.sim.start_viewer()
                print("可视化窗口已启动")
                continue
            
            result = controller.process(user_input)
            print(f"\n助手: {result}")
            
        except KeyboardInterrupt:
            print("\n\n已退出")
            break


def test_mode(headless: bool = False):
    """测试模式"""
    print("=" * 60)
    print("VLM + MuJoCo 物理抓取测试")
    print("=" * 60)
    
    controller = VLMGraspController()
    
    # 启动可视化
    if not headless:
        controller.sim.start_viewer()
        time.sleep(1)
    
    # 测试指令序列
    test_commands = [
        "查看当前状态",
        "抓取红色杯子",
        "把杯子放到左边",
        "抓取蓝色方块",
        "把它放到右边",
        "挥挥手打个招呼",
        "回到初始位置",
    ]
    
    print("\n开始自动测试...\n")
    results = []
    
    for cmd in test_commands:
        print("-" * 40)
        result = controller.process(cmd)
        print(f"结果: {result}")
        results.append((cmd, result))
        
        if not headless:
            time.sleep(2)
        else:
            time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for cmd, result in results:
        status = "✓" if "✓" in result or "位置" in result or "状态" in result else "?"
        print(f"  {status} {cmd[:20]:<20}")
    
    # 保持窗口打开
    if not headless and controller.sim.viewer and controller.sim.viewer.is_running():
        print("\n可视化窗口保持打开，按 ESC 关闭")
        while controller.sim.viewer.is_running():
            controller.sim._update_viewer()
            time.sleep(0.1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VLM + MuJoCo 物理抓取仿真")
    parser.add_argument("--test", action="store_true", help="自动测试模式")
    parser.add_argument("--headless", action="store_true", help="无头模式")
    args = parser.parse_args()
    
    if args.test:
        test_mode(headless=args.headless)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
