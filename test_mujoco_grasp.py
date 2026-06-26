#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MuJoCo 物理抓取测试（无VLM）
============================

直接测试 MuJoCo 中的物理抓取功能，不依赖 VLM API
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "12-unified-control/examples"))

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("请安装 mujoco 库: pip install mujoco")
    sys.exit(1)

# 复用无 VLM 的独立抓取仿真实现，避免依赖不存在的模块名
from test_grasp_standalone import GraspSimulator, GraspResult


def test_basic_grasp():
    """测试基本抓取功能"""
    print("=" * 60)
    print("MuJoCo 物理抓取测试（无VLM）")
    print("=" * 60)
    
    # 初始化仿真器
    print("\n[1/6] 初始化仿真器...")
    sim = GraspSimulator()
    
    # 启动可视化
    print("[2/6] 启动可视化窗口...")
    viewer = sim.start_viewer()
    time.sleep(1)
    
    # 测试1: 回零
    print("\n[3/6] 测试: 回零位置")
    sim.home()
    time.sleep(1)
    
    # 测试2: 查看状态
    print("\n[4/6] 测试: 查看状态")
    status = sim.status()
    print(status)
    time.sleep(1)
    
    # 测试3: 抓取红色杯子
    print("\n[5/6] 测试: 抓取红色杯子")
    result = sim.pick("target_cup")
    print(f"结果: {result.message}")
    print(f"  - 成功: {result.success}")
    print(f"  - 提升高度: {result.final_object_pos[2]:.3f}m")
    time.sleep(2)
    
    # 测试4: 放置
    if result.success:
        print("\n[6/6] 测试: 放置到左边")
        place_result = sim.place("left")
        print(f"放置结果: {'成功' if place_result else '失败'}")
        time.sleep(1)
    else:
        print("\n[6/6] 抓取失败，跳过放置测试")
    
    # 保持窗口打开
    print("\n" + "=" * 60)
    print("测试完成！可视化窗口保持打开")
    print("按 Ctrl+C 退出")
    print("=" * 60)
    
    try:
        while viewer.is_running():
            sim._update_viewer()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n已退出")


def test_grasp_sequence():
    """测试完整抓取序列"""
    print("=" * 60)
    print("MuJoCo 完整抓取序列测试")
    print("=" * 60)
    
    sim = GraspSimulator()
    viewer = sim.start_viewer()
    time.sleep(1)
    
    # 完整任务序列
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
            results.append((task_name, False))
        
        time.sleep(1)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for task_name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {task_name}")
    
    success_count = sum(1 for _, s in results if s)
    print(f"\n成功率: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    
    # 保持窗口打开
    print("\n可视化窗口保持打开，按 Ctrl+C 退出")
    try:
        while viewer.is_running():
            sim._update_viewer()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n已退出")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MuJoCo 物理抓取测试")
    parser.add_argument("--sequence", action="store_true", help="运行完整抓取序列")
    args = parser.parse_args()
    
    if args.sequence:
        test_grasp_sequence()
    else:
        test_basic_grasp()


if __name__ == "__main__":
    main()
