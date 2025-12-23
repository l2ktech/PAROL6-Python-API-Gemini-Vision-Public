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
from typing import Optional, Dict, Any, List
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

API_BASE = "http://localhost:8317/v1"
API_KEY = "cliproxy-ag-b9cd9ab23f51968c1afdf8fd2b7a6e26"
MODEL = "gpt-5.1"


# ==================== 数据类 ====================

@dataclass
class GraspResult:
    """抓取结果"""
    success: bool
    object_lifted: bool
    grasp_force: float
    final_object_pos: np.ndarray
    message: str


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
        
        # 初始化仿真状态，确保物体位置正确
        mujoco.mj_forward(self.model, self.data)
        self.step(100)  # 让物体稳定
        
        print("[MuJoCo] 物理抓取仿真器初始化完成")
    
    def _create_scene(self):
        """创建带物体的抓取场景"""
        scene_xml = '''<?xml version="1.0"?>
<mujoco model="parol6_grasp_scene">
  <compiler angle="radian" autolimits="true"/>
  <option integrator="implicitfast" timestep="0.001"/>
  
  <default>
    <default class="PAROL6">
      <joint armature="0.1" damping="50.0"/>
      <position kp="500" forcerange="-500 500"/>
    </default>
    <default class="gripper">
      <joint armature="0.1" damping="10.0"/>
      <position kp="1000" forcerange="-100 100"/>
    </default>
  </default>
  
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.8 0.8 0.8" width="512" height="512"/>
    <texture name="plane_tex" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7" width="512" height="512"/>
    <material name="plane_mat" texture="plane_tex" texrepeat="5 5"/>
  </asset>
  
  <worldbody>
    <!-- 地面 -->
    <geom name="ground" type="plane" size="2 2 0.1" material="plane_mat"/>
    <light pos="0 0 2" dir="0 0 -1"/>
    <light pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.5 0.5 0.5"/>

    <!-- 桌子 -->
    <body name="table" pos="0.3 0 0.4">
      <geom name="table_top" type="box" size="0.4 0.4 0.02" rgba="0.6 0.4 0.3 1"/>
      <geom name="table_leg1" type="cylinder" size="0.03 0.2" pos="0.35 0.35 -0.2" rgba="0.5 0.3 0.2 1"/>
      <geom name="table_leg2" type="cylinder" size="0.03 0.2" pos="-0.35 0.35 -0.2" rgba="0.5 0.3 0.2 1"/>
      <geom name="table_leg3" type="cylinder" size="0.03 0.2" pos="0.35 -0.35 -0.2" rgba="0.5 0.3 0.2 1"/>
      <geom name="table_leg4" type="cylinder" size="0.03 0.2" pos="-0.35 -0.35 -0.2" rgba="0.5 0.3 0.2 1"/>
    </body>

    <!-- 红色杯子 -->
    <body name="target_cup" pos="0.17 -0.21 0.35">
      <freejoint name="cup_joint"/>
      <geom name="cup_body" type="cylinder" size="0.02 0.03" rgba="1.0 0.3 0.3 1" 
            density="200" friction="1.5 0.1 0.1"/>
    </body>

    <!-- 蓝色方块 -->
    <body name="target_block" pos="0.15 -0.22 0.34">
      <freejoint name="block_joint"/>
      <geom name="block_body" type="box" size="0.015 0.015 0.015" rgba="0.3 0.3 1.0 1" 
            density="500" friction="1.5 0.1 0.1"/>
    </body>

    <!-- PAROL6机械臂 -->
    <body name="base_link" pos="0 0 0.42">
      <inertial pos="0 0 0.03" mass="0.8" diaginertia="0.001 0.001 0.001"/>
      <geom name="base" type="cylinder" size="0.05 0.03" rgba="0.2 0.2 0.2 1"/>
      
      <!-- L1 base rotation -->
      <body name="L1" pos="0 0 0.05">
        <inertial pos="0 0 0.05" mass="0.6" diaginertia="0.001 0.001 0.001"/>
        <joint name="L1" type="hinge" axis="0 0 1" range="-2.97 2.97" class="PAROL6"/>
        <geom type="cylinder" size="0.04 0.05" rgba="0.3 0.3 0.8 1"/>
        
        <!-- L2 shoulder -->
        <body name="L2" pos="0.023 0 0.11">
          <inertial pos="0 -0.09 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
          <joint name="L2" type="hinge" axis="0 1 0" range="-1.74 0.78" class="PAROL6"/>
          <geom type="capsule" fromto="0 0 0 0 -0.18 0" size="0.03" rgba="0.8 0.3 0.3 1"/>
          
          <!-- L3 elbow -->
          <body name="L3" pos="0 -0.18 0">
            <inertial pos="0.07 0 0" mass="0.5" diaginertia="0.0005 0.0005 0.0005"/>
            <joint name="L3" type="hinge" axis="0 1 0" range="-1.74 1.40" class="PAROL6"/>
            <geom type="capsule" fromto="0 0 0 0.149 0 0" size="0.025" rgba="0.3 0.8 0.3 1"/>
            
            <!-- L4 wrist1 -->
            <body name="L4" pos="0.149 0 0">
              <inertial pos="0 0 -0.05" mass="0.3" diaginertia="0.0003 0.0003 0.0003"/>
              <joint name="L4" type="hinge" axis="1 0 0" range="-3.14 3.14" class="PAROL6"/>
              <geom type="cylinder" size="0.02 0.02" rgba="0.8 0.8 0.3 1"/>
              
              <!-- L5 wrist2 -->
              <body name="L5" pos="0 0 -0.155">
                <inertial pos="0 0 0" mass="0.2" diaginertia="0.0001 0.0001 0.0001"/>
                <joint name="L5" type="hinge" axis="0 1 0" range="-2.10 2.10" class="PAROL6"/>
                <geom type="cylinder" size="0.02 0.015" rgba="0.8 0.3 0.8 1"/>
                
                <!-- L6 wrist3 -->
                <body name="L6" pos="0 -0.06 0">
                  <inertial pos="0 0 0" mass="0.1" diaginertia="0.00005 0.00005 0.00005"/>
                  <joint name="L6" type="hinge" axis="0 0 1" range="-6.28 6.28" class="PAROL6"/>
                  <geom type="cylinder" size="0.018 0.01" rgba="0.3 0.8 0.8 1"/>
                  
                  <!-- 末端site -->
                  <site name="end_effector" pos="0 0 -0.1" size="0.01"/>
                  
                    <!-- 夹爪基座 -->
                    <body name="gripper_base" pos="0 0 -0.03">
                      <inertial pos="0 0 0" mass="0.1" diaginertia="0.0001 0.0001 0.0001"/>
                      <geom type="box" size="0.02 0.03 0.015" rgba="0.5 0.5 0.5 1"/>
                      
                      <!-- 左手指 -->
                      <body name="gripper_left" pos="0 0.02 -0.04">
                        <inertial pos="0 0.01 0" mass="0.05" diaginertia="0.00003 0.00003 0.00003"/>
                        <joint name="rh_l1" type="slide" axis="0 1 0" range="0 0.04" class="gripper"/>
                        <geom type="box" size="0.015 0.005 0.03" rgba="0.4 0.4 0.4 1" friction="1.5 0.1 0.1"/>
                      </body>
                      
                      <!-- 右手指 -->
                      <body name="gripper_right" pos="0 -0.02 -0.04">
                        <inertial pos="0 -0.01 0" mass="0.05" diaginertia="0.00003 0.00003 0.00003"/>
                        <joint name="rh_r1" type="slide" axis="0 -1 0" range="0 0.04" class="gripper"/>
                        <geom type="box" size="0.015 0.005 0.03" rgba="0.4 0.4 0.4 1" friction="1.5 0.1 0.1"/>
                      </body>
                    </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <!-- 机械臂关节 -->
    <position name="L1_act" joint="L1" class="PAROL6"/>
    <position name="L2_act" joint="L2" class="PAROL6"/>
    <position name="L3_act" joint="L3" class="PAROL6"/>
    <position name="L4_act" joint="L4" class="PAROL6"/>
    <position name="L5_act" joint="L5" class="PAROL6"/>
    <position name="L6_act" joint="L6" class="PAROL6"/>
    
    <!-- 夹爪 -->
    <position name="rh_l1_motor" joint="rh_l1" ctrlrange="0 0.04" class="gripper"/>
    <position name="rh_r1_motor" joint="rh_r1" ctrlrange="0 0.04" class="gripper"/>
  </actuator>
</mujoco>'''
        
        self.model = mujoco.MjModel.from_xml_string(scene_xml)
        self.data = mujoco.MjData(self.model)
        print("[MuJoCo] 抓取场景创建完成")
    
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
        
        # 抓取轨迹 - 使用验证过的轨迹（来自 18_mujoco_grasp_simulation.py）
        # 杰子在 [0.17, -0.21, 0.35]，可达范围: X[0.13-0.19], Y[-0.23,-0.21], Z[0.30-0.32]
        
        # 1. 待机位置
        print("  阶段1: 待机位置")
        self.set_arm_joints([0, 0, 0, 0, 0, 0])
        self.set_gripper(0.03)  # 张开夹爪
        self.step(500)
        self._update_viewer()
        
        # 2. 移动到目标上方
        print("  阶段2: 移动到目标上方")
        self.set_arm_joints([0.5, -0.5, 0.5, 0, 0, 0])
        self.step(500)
        self._update_viewer()
        
        # 3. 接近目标
        print("  阶段3: 下降接近")
        self.set_arm_joints([0.8, -0.8, 0.6, 0, 0, 0])
        self.step(500)
        self._update_viewer()
        
        # 4. 精确定位
        print("  阶段4: 精确定位")
        self.set_arm_joints([1.0, -1.0, 0.8, 0, 0, 0])
        self.step(500)
        self._update_viewer()
        
        # 5. 闭合夹爪
        print("  阶段5: 闭合夹爪")
        self.set_gripper(0.0)
        self.step(400)
        self._update_viewer()
        
        # 6. 提升物体
        print("  阶段6: 提升物体")
        self.set_arm_joints([0.5, -0.3, 0.3, 0, 0, 0])
        self.step(500)
        self._update_viewer()
        
        # 7. 检查结果
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
