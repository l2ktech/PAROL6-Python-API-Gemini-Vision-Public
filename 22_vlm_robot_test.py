#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM机械臂控制测试 - OpenAI兼容API
===================================

使用自定义API测试VLM与PAROL6机械臂的交互

使用:
    python3 22_vlm_robot_test.py
    python3 22_vlm_robot_test.py --model gpt5.1

配置 (优先级: 环境变量 > vlm_config.json > 默认值):
    export VLM_API_KEY=你的密钥
    export VLM_API_BASE=http://localhost:8317/v1   # 可选
    export VLM_MODEL=gpt-5.1                        # 可选
    或复制 vlm_config.example.json 为 vlm_config.json 并填写

作者: wzy
日期: 2025-12-15
"""

import os
import sys
import json
import time
import base64
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

# 尝试导入openai库
try:
    from openai import OpenAI
except ImportError:
    print("请安装openai库: pip install openai")
    sys.exit(1)

# ==================== 配置 ====================
#
# 配置优先级: 环境变量 > vlm_config.json > 默认值
# 切勿在源码中硬编码 API Key（本仓库为公开仓库）。
#   - 环境变量: VLM_API_BASE / VLM_API_KEY / VLM_MODEL
#   - 配置文件: vlm_config.json (参见 vlm_config.example.json)


def load_config() -> Dict[str, str]:
    """加载配置：环境变量 > 配置文件 > 默认值"""
    config = {
        "api_base": "http://localhost:8317/v1",
        "api_key": "",
        "model": "gpt-5.1",
    }

    # 1. 尝试读取配置文件
    config_paths = [
        Path("vlm_config.json"),
        Path.home() / ".config" / "vlm" / "config.json",
        Path(__file__).parent / "vlm_config.json",
    ]
    for p in config_paths:
        if p.exists():
            try:
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


CONFIG = load_config()
API_BASE = CONFIG["api_base"]
API_KEY = CONFIG["api_key"]

if not API_KEY:
    print("[警告] 未设置 API Key（环境变量 VLM_API_KEY 或 vlm_config.json）")
    API_KEY = "test-key-placeholder"

# 可用模型别名 -> 实际模型 ID (用户自定义API)
MODELS = {
    "gpt5": "gpt-5",
    "gpt5.1": "gpt-5.1",
    "gpt5.2": "gpt-5.2",
    "gpt5-codex": "gpt-5-codex",
    "gpt4o-search": "gpt-4o-search-preview",
}

# 默认模型别名：优先使用配置中的 model（按其 ID 反查别名），否则回退到 gpt5.1
DEFAULT_MODEL = "gpt5.1"
for _alias, _model_id in MODELS.items():
    if _model_id == CONFIG.get("model"):
        DEFAULT_MODEL = _alias
        break


# ==================== VLM客户端 ====================

class VLMClient:
    """VLM API客户端"""
    
    def __init__(self, api_base: str = API_BASE, api_key: str = API_KEY):
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key
        )
        self.model = MODELS[DEFAULT_MODEL]
        print(f"[VLM] 初始化客户端")
        print(f"  API: {api_base}")
        print(f"  模型: {self.model}")
    
    def set_model(self, model_name: str):
        """设置模型"""
        if model_name in MODELS:
            self.model = MODELS[model_name]
        else:
            self.model = model_name
        print(f"[VLM] 切换模型: {self.model}")
    
    def chat(self, message: str, image_path: str = None) -> str:
        """
        发送聊天请求
        
        参数:
            message: 用户消息
            image_path: 可选图像路径
        
        返回:
            AI响应文本
        """
        messages = []
        
        # 构建用户消息
        if image_path and Path(image_path).exists():
            # 带图像的消息
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            
            # 判断图像类型
            ext = Path(image_path).suffix.lower()
            media_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }.get(ext, "image/jpeg")
            
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        }
                    }
                ]
            })
        else:
            # 纯文本消息
            messages.append({
                "role": "user",
                "content": message
            })
        
        try:
            print(f"[VLM] 发送请求...")
            start = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.7
            )
            
            elapsed = time.time() - start
            result = response.choices[0].message.content
            print(f"[VLM] 响应耗时: {elapsed:.2f}s")
            
            return result
            
        except Exception as e:
            return f"[错误] API请求失败: {e}"
    
    def test_connection(self) -> bool:
        """测试API连接"""
        try:
            response = self.chat("Say 'Hello' in Chinese")
            print(f"[VLM] 测试响应: {response[:100]}...")
            return True
        except Exception as e:
            print(f"[VLM] 连接测试失败: {e}")
            return False


# ==================== 机器人集成 ====================

class VLMRobotController:
    """VLM控制机器人"""
    
    def __init__(self, vlm: VLMClient):
        self.vlm = vlm
        self.robot_connected = False
        
        # 定义机器人控制指令映射
        self.commands = {
            "home": self._home,
            "pick": self._pick,
            "place": self._place,
            "wave": self._wave,
            "status": self._status,
        }
    
    def _check_robot(self) -> bool:
        """检查机器人连接"""
        try:
            # 尝试导入robot_api
            sys.path.insert(0, str(Path(__file__).parent.parent / "Headless"))
            from robot_api import get_robot_status
            status = get_robot_status()
            self.robot_connected = True
            return True
        except Exception as e:
            print(f"[Robot] 连接检查失败: {e}")
            self.robot_connected = False
            return False
    
    def _home(self):
        """回零"""
        try:
            from robot_api import home_robot
            return home_robot()
        except Exception as e:
            return f"回零失败: {e}"
    
    def _pick(self, target: str = "object"):
        """抓取"""
        return f"[模拟] 抓取 {target}"
    
    def _place(self, location: str = "table"):
        """放置"""
        return f"[模拟] 放置到 {location}"
    
    def _wave(self):
        """挥手"""
        return "[模拟] 挥手"
    
    def _status(self):
        """状态"""
        return f"机器人连接: {self.robot_connected}"
    
    def process_command(self, user_input: str) -> str:
        """
        用VLM理解用户指令并执行
        
        参数:
            user_input: 自然语言指令
        
        返回:
            执行结果
        """
        # 使用VLM理解用户意图
        prompt = f"""你是PAROL6机械臂控制助手。

用户说: "{user_input}"

可用命令:
- home: 回零
- pick [目标]: 抓取
- place [位置]: 放置
- wave: 挥手
- status: 状态

请分析用户意图，返回JSON格式:
{{"command": "命令名", "args": {{"参数": "值"}}}}

只返回JSON，不要其他内容。"""

        response = self.vlm.chat(prompt)
        
        try:
            # 解析VLM响应
            result = json.loads(response.strip())
            cmd = result.get("command", "")
            args = result.get("args", {})
            
            if cmd in self.commands:
                if args:
                    return self.commands[cmd](**args)
                else:
                    return self.commands[cmd]()
            else:
                return f"未知命令: {cmd}"
                
        except json.JSONDecodeError:
            return f"VLM响应: {response}"


# ==================== 主程序 ====================

def test_vlm_api(model: str = None):
    """测试VLM API连接"""
    print("="*60)
    print("VLM API连接测试")
    print("="*60)

    vlm = VLMClient()
    if model:
        vlm.set_model(model)

    print("\n1. 测试基本连接...")
    if vlm.test_connection():
        print("   ✓ API连接成功")
    else:
        print("   ✗ API连接失败")
        print("\n请确保API服务已启动:")
        print(f"  端口: 35621")
        return False
    
    print("\n2. 测试文本对话...")
    response = vlm.chat("用中文介绍一下PAROL6机械臂")
    print(f"   响应: {response[:200]}...")
    
    print("\n测试完成!")
    return True


def interactive_mode(model: str = None):
    """交互模式"""
    print("="*60)
    print("VLM机械臂控制 - 交互模式")
    print("="*60)
    print("输入 'quit' 退出")
    print("输入 'model <名称>' 切换模型")
    print("  可用: " + ", ".join(MODELS.keys()))
    print()

    vlm = VLMClient()
    if model:
        vlm.set_model(model)
    controller = VLMRobotController(vlm)
    
    while True:
        try:
            user_input = input("\n你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == "quit":
                print("再见!")
                break
            
            if user_input.lower().startswith("model "):
                model_name = user_input[6:].strip()
                vlm.set_model(model_name)
                continue
            
            # 处理用户指令
            result = controller.process_command(user_input)
            print(f"\n助手: {result}")
            
        except KeyboardInterrupt:
            print("\n\n已退出")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="VLM机械臂控制测试")
    parser.add_argument("--test", action="store_true", help="仅测试API连接")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="选择模型 (可用: " + ", ".join(MODELS.keys()) + ")",
    )
    args = parser.parse_args()

    if args.test:
        test_vlm_api(args.model)
    else:
        interactive_mode(args.model)


if __name__ == "__main__":
    main()
