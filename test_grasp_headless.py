#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuJoCo 物理抓取测试（纯无头模式）
=================================

完全无头模式测试，不依赖图形界面
"""

import atexit
import os
import sys
import tempfile
import time
import numpy as np
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "12-unified-control/examples"))

try:
    import mujoco
except ImportError:
    print("请安装 mujoco 库: pip install mujoco")
    sys.exit(1)


def _cleanup_temp_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


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
    """简化的逆运动学求解器"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'end_effector')
        if self.ee_site_id < 0:
            print("[警告] 未找到 end_effector site")
            self.ee_site_id = 0
    
    def solve(self, target_pos: np.ndarray, current_q: np.ndarray = None, 
              max_iter: int = 100, tol: float = 0.02) -> Tuple[np.ndarray, float]:
        """求解逆运动学"""
        if current_q is None:
            current_q = np.zeros(6)
        
        q = current_q.copy()
        dist = float('inf')
        
        for iteration in range(max_iter):
            self.data.qpos[:6] = q
            mujoco.mj_forward(self.model, self.data)
            
            ee_pos = self.data.site_xpos[self.ee_site_id]
            error = target_pos - ee_pos
            dist = np.linalg.norm(error)
            
            if dist < tol:
                return q, dist
            
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.ee_site_id)
            
            J = jacp[:, :6]
            damping = 0.1
            JJT = J @ J.T + damping**2 * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error)
            
            q = q + 0.5 * dq
            q = np.clip(q, [-1.7, -0.98, -2.0, -2.0, -2.1, -6.28],
                       [1.7, 1.0, 1.3, 2.0, 2.1, 6.28])
        
        return q, dist


# ==================== MuJoCo 物理抓取仿真器 ====================

class GraspSimulator:
    """PAROL6 物理抓取仿真器（无头模式）"""
    
    def __init__(self):
        self.model = None
        self.data = None
        self._temp_scene_path = None
        self.arm_joint_ids = []
        self.gripper_joint_ids = []
        self.gripper_open = True
        self.holding_object = None
        
        self._create_scene()
        self._init_joint_indices()
        self.ik_solver = SimpleIKSolver(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.step(100)
        
        print("[MuJoCo] 物理抓取仿真器初始化完成（无头模式）")
    
    def _create_scene(self):
        """创建带物体的抓取场景"""
        import xml.etree.ElementTree as ET
        
        model_path_env = os.environ.get("PAROL6_MODEL_PATH")
        if model_path_env:
            base_model_path = Path(model_path_env)
        else:
            base_model_path = PROJECT_ROOT / "11.1-lerobot-mujoco-smolvla" / "asset" / "parol6" / "parol6_fixed.xml"
        
        if not base_model_path.exists():
            raise FileNotFoundError(f"PAROL6 模型文件不存在: {base_model_path}")
        
        tree = ET.parse(base_model_path)
        root = tree.getroot()
        worldbody = root.find('worldbody')
        
        # 添加红色杯子
        cup_body = ET.SubElement(worldbody, 'body')
        cup_body.set('name', 'target_cup')
        cup_body.set('pos', '0.12 0.05 0.92')
        
        cup_joint = ET.SubElement(cup_body, 'freejoint')
        cup_joint.set('name', 'cup_joint')
        
        cup_geom = ET.SubElement(cup_body, 'geom')
        cup_geom.set('name', 'cup_body')
        cup_geom.set('type', 'cylinder')
        cup_geom.set('size', '0.025 0.04')
        cup_geom.set('rgba', '1.0 0.3 0.3 1')
        cup_geom.set('density', '500')
        cup_geom.set('friction', '1.5 0.1 0.1')
        
        # 添加蓝色方块
        block_body = ET.SubElement(worldbody, 'body')
        block_body.set('name', 'target_block')
        block_body.set('pos', '0.08 -0.08 0.91')
        
        block_joint = ET.SubElement(block_body, 'freejoint')
        block_joint.set('name', 'block_joint')
        
        block_geom = ET.SubElement(block_body, 'geom')
        block_geom.set('name', 'block_body')
        block_geom.set('type', 'box')
        block_geom.set('size', '0.02 0.02 0.02')
        block_geom.set('rgba', '0.3 0.3 1.0 1')
        block_geom.set('density', '800')
        block_geom.set('friction', '1.5 0.1 0.1')
        
        with tempfile.NamedTemporaryFile(
            dir=base_model_path.parent,
            prefix="parol6_grasp_scene_",
            suffix=".xml",
            delete=False,
        ) as temp_file:
            self._temp_scene_path = Path(temp_file.name)
        atexit.register(_cleanup_temp_file, self._temp_scene_path)
        tree.write(str(self._temp_scene_path), encoding='unicode')
        
        self.model = mujoco.MjModel.from_xml_path(str(self._temp_scene_path))
        self.data = mujoco.MjData(self.model)
        print("[MuJoCo] 抓取场景创建完成")
    
    def _init_joint_indices(self):
        """初始化关节索引"""
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
        """设置夹爪开度"""
        self.data.ctrl[6] = open_width
        self.data.ctrl[7] = open_width
        self.gripper_open = open_width > 0.01
    
    def get_object_position(self, body_name: str) -> np.ndarray:
        """获取物体位置"""
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
        self.set_gripper(0.02)
        self.step(500)
        print("[MuJoCo] 回零完成")
        return True
    
    def pick(self, target: str = "target_cup") -> GraspResult:
        """执行物理抓取"""
        print(f"\n[MuJoCo] 执行: 抓取 {target}")
        
        target_pos = self.get_object_position(target)
        initial_z = target_pos[2]
        print(f"  目标位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        # 1. 待机位置
        print("  阶段1: 待机位置")
        self.set_arm_joints([0, 0, 0, 0, 0, 0])
        self.set_gripper(0.03)
        self.step(400)
        
        # 2. 预抓取位置
        pre_grasp_pos = target_pos.copy()
        pre_grasp_pos[2] += 0.10
        print(f"  阶段2: IK 求解预抓取位置 ({pre_grasp_pos[0]:.3f}, {pre_grasp_pos[1]:.3f}, {pre_grasp_pos[2]:.3f})")
        
        pre_q, pre_dist = self.ik_solver.solve(pre_grasp_pos)
        if pre_dist > 0.05:
            print(f"    [错误] IK 求解失败，误差: {pre_dist:.3f}m")
            self.home()
            return GraspResult(
                success=False, object_lifted=False, grasp_force=0.0,
                final_object_pos=target_pos,
                message=f"✗ IK 求解失败 (误差 {pre_dist:.3f}m)"
            )
        
        print(f"    IK 成功，误差: {pre_dist:.3f}m")
        self.set_arm_joints(pre_q.tolist())
        self.step(500)
        
        # 3. 抓取位置
        grasp_pos = target_pos.copy()
        grasp_pos[2] -= 0.02
        print(f"  阶段3: IK 求解抓取位置 ({grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f})")
        
        grasp_q, grasp_dist = self.ik_solver.solve(grasp_pos, pre_q)
        if grasp_dist > 0.05:
            print(f"    [错误] IK 求解失败，误差: {grasp_dist:.3f}m")
            self.home()
            return GraspResult(
                success=False, object_lifted=False, grasp_force=0.0,
                final_object_pos=target_pos,
                message=f"✗ 抓取位置 IK 失败 (误差 {grasp_dist:.3f}m)"
            )
        
        print(f"    IK 成功，误差: {grasp_dist:.3f}m")
        self.set_arm_joints(grasp_q.tolist())
        self.step(500)
        
        # 4. 闭合夹爪
        print("  阶段4: 闭合夹爪")
        self.set_gripper(0.0)
        self.step(400)
        
        # 5. 提升物体
        lift_pos = grasp_pos.copy()
        lift_pos[2] += 0.12
        print(f"  阶段5: IK 求解提升位置 ({lift_pos[0]:.3f}, {lift_pos[1]:.3f}, {lift_pos[2]:.3f})")
        
        lift_q, lift_dist = self.ik_solver.solve(lift_pos, grasp_q)
        if lift_dist > 0.05:
            print(f"    [警告] 提升位置 IK 误差较大: {lift_dist:.3f}m")
            lift_q = grasp_q.copy()
            lift_q[1] -= 0.3
        else:
            print(f"    IK 成功，误差: {lift_dist:.3f}m")
        
        self.set_arm_joints(lift_q.tolist())
        self.step(600)
        
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
        """放置物体"""
        print(f"\n[MuJoCo] 执行: 放置到 {location}")
        
        if self.holding_object is None:
            print("  没有抓取的物体！")
            return False
        
        location_joints = {
            "left": [-0.8, -0.4, 0.4, 0, 0, 0],
            "right": [0.8, -0.4, 0.4, 0, 0, 0],
            "center": [0, -0.4, 0.4, 0, 0, 0],
            "front": [0, -0.6, 0.6, 0, 0, 0],
        }
        
        joints = location_joints.get(location, location_joints["center"])
        
        print(f"  阶段1: 移动到 {location}")
        self.set_arm_joints(joints)
        self.step(500)
        
        print("  阶段2: 下降")
        joints[1] -= 0.2
        joints[2] += 0.2
        self.set_arm_joints(joints)
        self.step(400)
        
        print("  阶段3: 释放物体")
        self.set_gripper(0.025)
        self.step(300)
        
        print("  阶段4: 缩回")
        self.set_arm_joints([0, -0.3, 0.3, 0, 0, 0])
        self.step(400)
        
        self.holding_object = None
        print("  放置完成")
        return True
    
    def wave(self) -> bool:
        """挥手动作"""
        print("[MuJoCo] 执行: 挥手")
        
        self.set_arm_joints([0, -0.5, 0.5, 0, 0.5, 0])
        self.step(300)
        
        for i in range(3):
            self.set_arm_joints([0, -0.5, 0.5, 0, 0.5, 0.5])
            self.step(150)
            
            self.set_arm_joints([0, -0.5, 0.5, 0, 0.5, -0.5])
            self.step(150)
        
        self.set_arm_joints([0, 0, 0, 0, 0, 0])
        self.step(300)
        
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


# ==================== 主程序 ====================

def test_grasp_sequence():
    """测试完整抓取序列"""
    print("=" * 60)
    print("MuJoCo 完整抓取序列测试（无头模式）")
    print("=" * 60)
    
    sim = GraspSimulator()
    
    tasks = [
        ("回零", lambda: sim.home()),
        ("抓取红色杯子", lambda: sim.pick("target_cup")),
        ("放置到左边", lambda: sim.place("left")),
        ("抓取蓝色方块", lambda: sim.pick("target_block")),
        ("放置到右边", lambda: sim.place("right")),
        ("挥手", lambda: sim.wave()),
        ("回零", lambda: sim.home()),
    ]
    
    print("\n开始执行任务序列...\n")
    
    results = []
    for i, (task_name, task_func) in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] 执行: {task_name}")
        try:
            result = task_func()
            if isinstance(result, GraspResult):
                success = result.success
                msg = result.message
            else:
                success = result
                msg = "完成" if result else "失败"
            
            status = "✓" if success else "✗"
            print(f"  结果: {status} {msg}")
            results.append((task_name, success))
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            results.append((task_name, False))
        
        time.sleep(0.5)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for task_name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {task_name}")
    
    success_count = sum(1 for _, s in results if s)
    print(f"\n成功率: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")


if __name__ == "__main__":
    test_grasp_sequence()
