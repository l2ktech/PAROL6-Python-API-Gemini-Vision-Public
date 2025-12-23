#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM + PAROL6 综合交互测试
===========================

测试内容:
1. VLM API连接
2. 机器人基础控制
3. VLM理解+机器人联动
4. 视觉和夹爪(可选)

使用:
    python3 23_vlm_full_test.py

作者: wzy
日期: 2025-12-15
"""

import os
import sys
import time
import json
from pathlib import Path

# 添加路径
sys.path.insert(0, '/l2k/home/wzy/21-L2Karm/12-unified-control/src')

# 尝试导入
try:
    from openai import OpenAI
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False
    print("⚠️ openai未安装")

try:
    from parol6_usb import PAROL6USB
    ROBOT_OK = True
except ImportError:
    ROBOT_OK = False
    print("⚠️ parol6_usb未找到")


# ==================== 配置 ====================

API_BASE = "http://localhost:8317/v1"
API_KEY = "cliproxy-ag-b9cd9ab23f51968c1afdf8fd2b7a6e26"
MODEL = "gpt-5.1"

# 测试位置
STANDBY_POS = [0, -90, 180, 0, 0, 90]
TEST_POS_1 = [10, -80, 170, 0, 0, 90]
TEST_POS_2 = [-10, -100, 190, 0, 0, 90]


# ==================== 串口自动检测 ====================

def detect_parol6_port():
    """
    自动检测PAROL6机械臂串口
    
    通过/dev/serial/by-id查找STMicroelectronics设备
    
    返回:
        str: 串口路径，如 /dev/ttyACM2
        None: 未找到
    """
    import subprocess
    import os
    
    # 方法1: 通过by-id查找STM32
    by_id_path = '/dev/serial/by-id'
    if os.path.exists(by_id_path):
        try:
            for name in os.listdir(by_id_path):
                if 'STMicroelectronics' in name or 'F446' in name:
                    link_path = os.path.join(by_id_path, name)
                    real_path = os.path.realpath(link_path)
                    print(f"[串口检测] 找到: {name}")
                    print(f"  -> {real_path}")
                    return real_path
        except Exception as e:
            print(f"[串口检测] by-id扫描失败: {e}")
    
    # 方法2: 使用lsusb + dmesg
    try:
        result = subprocess.run(
            ['lsusb'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if 'STMicroelectronics' in result.stdout:
            print("[串口检测] lsusb检测到STM32设备")
            
            # 尝试从dmesg获取最近的ttyACM设备
            dmesg_result = subprocess.run(
                ['dmesg'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            lines = dmesg_result.stdout.split('\n')
            for line in reversed(lines):
                if 'ttyACM' in line:
                    import re
                    match = re.search(r'(ttyACM\d+)', line)
                    if match:
                        port = f'/dev/{match.group(1)}'
                        print(f"[串口检测] dmesg找到: {port}")
                        return port
    except Exception as e:
        print(f"[串口检测] lsusb方法失败: {e}")
    
    # 方法3: 扫描ttyACM设备
    for i in range(10):
        port = f'/dev/ttyACM{i}'
        if os.path.exists(port):
            print(f"[串口检测] 尝试: {port}")
            return port
    
    print("[串口检测] 未找到PAROL6设备")
    return None


# ==================== 测试类 ====================

class VLMRobotTest:
    """VLM机器人综合测试"""
    
    def __init__(self):
        self.vlm_client = None
        self.robot = None
        self.results = []
    
    def setup(self):
        """初始化"""
        print("="*60)
        print("VLM + PAROL6 综合交互测试")
        print("="*60)
        
        # VLM客户端
        if OPENAI_OK:
            try:
                self.vlm_client = OpenAI(base_url=API_BASE, api_key=API_KEY)
                print("✓ VLM客户端初始化")
            except Exception as e:
                print(f"✗ VLM初始化失败: {e}")
        
        # 机器人
        if ROBOT_OK:
            try:
                # 自动检测串口
                detected_port = detect_parol6_port()
                if detected_port:
                    self.robot = PAROL6USB(port=detected_port)
                    if self.robot.connect():
                        print(f"✓ 机器人连接成功 ({detected_port})")
                        self.robot.enable()
                        print("✓ 机器人使能")
                        
                        # 自动回零
                        print("⏳ 执行回零（约30秒）...")
                        if self.robot.home(wait=True, timeout=60.0):
                            print("✓ 回零完成")
                        else:
                            print("⚠️ 回零可能未完成")
                    else:
                        print("✗ 机器人连接失败")
                        self.robot = None
                else:
                    print("✗ 未检测到串口")
                    self.robot = None
            except Exception as e:
                print(f"✗ 机器人初始化失败: {e}")
                self.robot = None
        
        return self.vlm_client is not None
    
    def teardown(self):
        """清理"""
        if self.robot:
            try:
                self.robot.disconnect()
                print("✓ 机器人断开")
            except:
                pass
    
    def record(self, name, success, detail=""):
        """记录结果"""
        self.results.append({
            "name": name,
            "success": success,
            "detail": detail
        })
        status = "✓" if success else "✗"
        print(f"  {status} {name}: {detail}")
    
    # ==================== 测试1: VLM API ====================
    
    def test_vlm_connection(self):
        """测试VLM API连接"""
        print("\n[测试1] VLM API连接")
        print("-"*40)
        
        if not self.vlm_client:
            self.record("VLM连接", False, "客户端未初始化")
            return False
        
        try:
            start = time.time()
            resp = self.vlm_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "你好，用中文说一句话"}],
                max_tokens=50
            )
            elapsed = time.time() - start
            content = resp.choices[0].message.content[:50]
            self.record("VLM连接", True, f"{elapsed:.2f}s - {content}...")
            return True
        except Exception as e:
            self.record("VLM连接", False, str(e)[:50])
            return False
    
    def test_vlm_robot_understanding(self):
        """测试VLM理解机器人指令"""
        print("\n[测试2] VLM机器人指令理解")
        print("-"*40)
        
        if not self.vlm_client:
            self.record("指令理解", False, "VLM不可用")
            return False
        
        # 测试不同指令
        test_cases = [
            ("向前移动50毫米", "move"),
            ("打开夹爪", "gripper"),
            ("回到零点", "home"),
            ("画一个圆", "circle"),
        ]
        
        success_count = 0
        for cmd, expected in test_cases:
            try:
                prompt = f"""分析机器人指令: "{cmd}"
返回JSON: {{"action": "类型", "params": {{}}}}
只返回JSON。"""
                
                resp = self.vlm_client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100
                )
                content = resp.choices[0].message.content
                
                # 简单检查
                if expected.lower() in content.lower():
                    success_count += 1
                    self.record(f"理解'{cmd}'", True, "正确识别")
                else:
                    self.record(f"理解'{cmd}'", False, content[:30])
            except Exception as e:
                self.record(f"理解'{cmd}'", False, str(e)[:30])
        
        return success_count >= 2
    
    # ==================== 测试3: 机器人控制 ====================
    
    def test_robot_basic(self):
        """测试机器人基础控制"""
        print("\n[测试3] 机器人基础控制")
        print("-"*40)
        
        if not self.robot:
            self.record("机器人控制", False, "未连接")
            return False
        
        try:
            # 移动到待机位置
            print("  移动到待机位置...")
            result = self.robot.move_joints(STANDBY_POS, speed=40, wait=True)
            time.sleep(2)
            self.record("移动到待机", result, str(STANDBY_POS))
            
            # 小幅移动测试
            print("  J1小幅移动...")
            result = self.robot.move_joints(TEST_POS_1, speed=30, wait=True)
            time.sleep(2)
            self.record("J1移动", result, str(TEST_POS_1))
            
            # 回到待机
            print("  返回待机...")
            result = self.robot.move_joints(STANDBY_POS, speed=40, wait=True)
            time.sleep(1)
            self.record("返回待机", result, "完成")
            
            return True
        except Exception as e:
            self.record("机器人控制", False, str(e)[:50])
            return False
    
    # ==================== 测试4: VLM+机器人联动 ====================
    
    def test_vlm_robot_integration(self):
        """测试VLM+机器人联动"""
        print("\n[测试4] VLM+机器人联动")
        print("-"*40)
        
        if not self.vlm_client or not self.robot:
            self.record("VLM+Robot联动", False, "组件不完整")
            return False
        
        try:
            # VLM生成运动指令
            prompt = """你是PAROL6机械臂控制器。
当前位置: [0, -90, 180, 0, 0, 90]度
用户说: "把第一个关节向右转5度"
返回新的目标位置，JSON格式:
{"joints": [j1, j2, j3, j4, j5, j6]}
只返回JSON。"""
            
            resp = self.vlm_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            content = resp.choices[0].message.content
            
            # 解析目标
            import re
            match = re.search(r'\[([^\]]+)\]', content)
            if match:
                joints_str = match.group(1)
                target = [float(x.strip()) for x in joints_str.split(',')]
                
                if len(target) == 6:
                    print(f"  VLM生成目标: {target}")
                    
                    # 执行移动
                    result = self.robot.move_joints(target, speed=25, wait=True)
                    time.sleep(2)
                    self.record("VLM生成+执行", result, str(target))
                    
                    # 返回
                    self.robot.move_joints(STANDBY_POS, speed=30, wait=True)
                    return result
            
            self.record("VLM生成+执行", False, "解析失败")
            return False
            
        except Exception as e:
            self.record("VLM+Robot联动", False, str(e)[:50])
            return False
    
    # ==================== 运行所有测试 ====================
    
    def run_all(self):
        """运行所有测试"""
        if not self.setup():
            print("\n初始化失败，部分测试将跳过")
        
        print("\n" + "="*60)
        print("开始测试")
        print("="*60)
        
        # 运行测试
        self.test_vlm_connection()
        self.test_vlm_robot_understanding()
        self.test_robot_basic()
        self.test_vlm_robot_integration()
        
        # 清理
        self.teardown()
        
        # 结果汇总
        print("\n" + "="*60)
        print("测试结果汇总")
        print("="*60)
        
        passed = sum(1 for r in self.results if r['success'])
        total = len(self.results)
        
        print(f"\n通过: {passed}/{total}")
        
        if passed < total:
            print("\n失败项目:")
            for r in self.results:
                if not r['success']:
                    print(f"  ✗ {r['name']}: {r['detail']}")
        
        print("\n" + "="*60)
        return passed == total


# ==================== 主程序 ====================

if __name__ == "__main__":
    tester = VLMRobotTest()
    success = tester.run_all()
    sys.exit(0 if success else 1)
