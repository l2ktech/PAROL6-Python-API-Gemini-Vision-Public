#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM + MuJoCo 仿真控制
=====================

使用 VLM（视觉语言模型）通过自然语言控制 MuJoCo 仿真中的 PAROL6 机械臂

功能:
1. 自然语言理解（VLM）
2. MuJoCo 仿真控制
3. 抓取、放置、回零等动作

使用:
    python3 25_vlm_mujoco_control.py

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

# ==================== 路径配置 ====================

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "12-unified-control/examples"))
sys.path.insert(0, str(PROJECT_ROOT / "06-VLM-Gemini-Vision"))

# 尝试导入依赖
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

# 使用 cli-proxy-api
API_BASE = "http://localhost:8317/v1"
API_KEY = "cliproxy-ag-b9cd9ab23f51968c1afdf8fd2b7a6e26"
MODEL = "gpt-5.1"

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
                temperature=0.3  # 低温度获得更确定性的响应
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[错误] {e}"
    
    def parse_command(self, user_input: str) -> Dict[str, Any]:
        """
        用 VLM 解析用户指令为机器人命令
        
        返回: {"command": "命令名", "params": {...}}
        """
        prompt = f"""你是 PAROL6 机械臂控制助手。

用户说: "{user_input}"

可用命令:
- home: 回零位置
- pick: 抓取物体
- place: 放置物体
- wave: 挥手动作
- move: 移动到指定位置

请分析用户意图，返回 JSON 格式:
{{"command": "命令名", "params": {{"参数名": "值"}}}}

只返回 JSON，不要其他内容。如果不理解指令，返回:
{{"command": "unknown", "params": {{"message": "原因"}}}}"""

        response = self.chat(prompt)
        
        try:
            # 尝试提取 JSON
            response = response.strip()
            if response.startswith("```"):
                # 移除代码块标记
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            return json.loads(response)
        except:
            return {"command": "chat", "params": {"response": response}}


# ==================== MuJoCo 仿真控制器 ====================

class MuJoCoSimulator:
    """MuJoCo PAROL6 仿真器（简化版）"""
    
    def __init__(self):
        self.model = None
        self.data = None
        self.viewer = None
        self._create_scene()
        print("[MuJoCo] 仿真器初始化完成")
    
    def _create_scene(self):
        """创建仿真场景"""
        # 查找 PAROL6 模型
        model_paths = [
            PROJECT_ROOT / "11.1-lerobot-mujoco-smolvla/asset/parol6/parol6_fixed.xml",
            PROJECT_ROOT / "12-unified-control/models/parol6.xml",
        ]
        
        xml_path = None
        for path in model_paths:
            if path.exists():
                xml_path = path
                break
        
        if xml_path is None:
            # 使用简单的内置场景
            print("[MuJoCo] 未找到 PAROL6 模型，使用简单场景")
            xml = """
            <mujoco>
                <worldbody>
                    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                    <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
                    <body name="arm" pos="0 0 0.5">
                        <geom type="box" size="0.1 0.1 0.5" rgba="0.8 0.2 0.2 1"/>
                    </body>
                </mujoco>
            """
            self.model = mujoco.MjModel.from_xml_string(xml)
        else:
            print(f"[MuJoCo] 加载模型: {xml_path}")
            self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        
        self.data = mujoco.MjData(self.model)
        
        # 获取关节信息
        self.joint_names = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_names.append(name)
        print(f"[MuJoCo] 关节: {self.joint_names[:6]}")
    
    def home(self):
        """回零位置"""
        print("[MuJoCo] 执行: 回零")
        # 设置所有关节为 0
        self.data.qpos[:min(6, len(self.data.qpos))] = 0
        self._step(200)
        return True
    
    def pick(self, target: str = "object"):
        """抓取动作"""
        print(f"[MuJoCo] 执行: 抓取 {target}")
        # 模拟抓取动作序列
        # 1. 移动到抓取位置
        if len(self.data.qpos) >= 6:
            self.data.qpos[0] = 0  # J1
            self.data.qpos[1] = np.radians(-45)  # J2
            self.data.qpos[2] = np.radians(90)  # J3
            self.data.qpos[3] = 0  # J4
            self.data.qpos[4] = np.radians(45)  # J5
            self.data.qpos[5] = 0  # J6
        self._step(300)
        print("[MuJoCo] 抓取完成")
        return True
    
    def place(self, location: str = "table"):
        """放置动作"""
        print(f"[MuJoCo] 执行: 放置到 {location}")
        # 模拟放置动作
        if len(self.data.qpos) >= 6:
            self.data.qpos[0] = np.radians(45)  # 旋转
            self.data.qpos[1] = np.radians(-30)
            self.data.qpos[2] = np.radians(60)
        self._step(300)
        print("[MuJoCo] 放置完成")
        return True
    
    def wave(self):
        """挥手动作"""
        print("[MuJoCo] 执行: 挥手")
        # 挥手动作序列
        for i in range(3):
            if len(self.data.qpos) >= 6:
                self.data.qpos[5] = np.radians(30)
            self._step(100)
            if len(self.data.qpos) >= 6:
                self.data.qpos[5] = np.radians(-30)
            self._step(100)
        self.data.qpos[5] = 0
        self._step(100)
        print("[MuJoCo] 挥手完成")
        return True
    
    def move(self, joints: List[float] = None):
        """移动到指定关节位置"""
        if joints is None:
            joints = [0, 0, 0, 0, 0, 0]
        print(f"[MuJoCo] 执行: 移动到 {joints}")
        for i, angle in enumerate(joints[:min(6, len(self.data.qpos))]):
            self.data.qpos[i] = np.radians(angle)
        self._step(300)
        return True
    
    def _step(self, n_steps: int = 100):
        """仿真步进"""
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
    
    def start_viewer(self):
        """启动可视化窗口"""
        print("[MuJoCo] 启动可视化...")
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self.viewer
    
    def update_viewer(self):
        """更新可视化"""
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()


# ==================== VLM 仿真控制器 ====================

class VLMSimController:
    """VLM + MuJoCo 仿真控制器"""
    
    def __init__(self):
        self.vlm = VLMClient()
        self.sim = MuJoCoSimulator()
        
        # 命令映射
        self.commands = {
            "home": self.sim.home,
            "pick": self.sim.pick,
            "place": self.sim.place,
            "wave": self.sim.wave,
            "move": self.sim.move,
        }
        
        print("[Controller] VLM + MuJoCo 控制器初始化完成")
    
    def process(self, user_input: str) -> str:
        """
        处理用户自然语言输入
        
        流程: 用户输入 → VLM 解析 → 执行动作 → 返回结果
        """
        print(f"\n[用户] {user_input}")
        
        # 1. VLM 解析
        result = self.vlm.parse_command(user_input)
        cmd = result.get("command", "unknown")
        params = result.get("params", {})
        
        print(f"[VLM] 解析: command={cmd}, params={params}")
        
        # 2. 执行命令
        if cmd in self.commands:
            try:
                # 提取参数执行
                if cmd == "pick" and "target" in params:
                    success = self.commands[cmd](params["target"])
                elif cmd == "place" and "location" in params:
                    success = self.commands[cmd](params["location"])
                elif cmd == "move" and "joints" in params:
                    success = self.commands[cmd](params["joints"])
                else:
                    success = self.commands[cmd]()
                
                # 更新可视化
                self.sim.update_viewer()
                
                return f"✓ 执行成功: {cmd}"
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
    print("VLM + MuJoCo 仿真控制")
    print("=" * 60)
    print("输入自然语言指令控制机械臂")
    print("  示例: '回零', '抓取红色方块', '放到左边', '挥手'")
    print("  输入 'quit' 退出")
    print("  输入 'view' 启动可视化窗口")
    print()
    
    controller = VLMSimController()
    
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
            
            # 处理指令
            result = controller.process(user_input)
            print(f"\n助手: {result}")
            
        except KeyboardInterrupt:
            print("\n\n已退出")
            break


def test_mode(headless: bool = False):
    """测试模式 - 无交互自动测试"""
    print("=" * 60)
    print("VLM + MuJoCo 自动测试")
    print("=" * 60)
    
    controller = VLMSimController()
    
    # 启动可视化（可选）
    if not headless:
        controller.sim.start_viewer()
        time.sleep(1)
    
    # 测试指令列表
    test_commands = [
        "回到初始位置",
        "抓取红色方块",
        "把它放到左边",
        "打个招呼挥挥手",
    ]
    
    print("\n开始自动测试...\n")
    
    for cmd in test_commands:
        print("-" * 40)
        result = controller.process(cmd)
        print(f"结果: {result}")
        if not headless:
            time.sleep(2)
            controller.sim.update_viewer()
        else:
            time.sleep(0.5)
    
    print("\n测试完成！")
    
    # 保持窗口打开（非 headless 模式）
    if not headless and controller.sim.viewer and controller.sim.viewer.is_running():
        print("可视化窗口保持打开，按 ESC 关闭")
        while controller.sim.viewer.is_running():
            controller.sim.update_viewer()
            time.sleep(0.1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VLM + MuJoCo 仿真控制")
    parser.add_argument("--test", action="store_true", help="自动测试模式")
    parser.add_argument("--headless", action="store_true", help="无头模式（无可视化）")
    args = parser.parse_args()
    
    if args.test:
        test_mode(headless=args.headless)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
