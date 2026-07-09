# PAROL6 智能机械臂控制系统 - 中文使用说明

## Public Portfolio Summary

This repository is the public evidence entry for the PAROL6 + Gemini Vision control demo. It shows how a natural-language and vision-guided robot workflow can be assembled around the Headless PAROL6 Python API, Gemini Vision, RealSense RGB-D input, MuJoCo-side grasp tests, and hardware-facing validation notes.

For interviews, read it as an integration and adaptation project: upstream PAROL6 robot assets and API concepts are used as the base, while this repository focuses on the public VLM control workflow, test scripts, documentation, safety notes, and reproducible evidence around vision-guided operation.

## Portfolio / Interview Context

- Portfolio overview: https://notion.l2k.tech:28443/article/interview-portfolio
- P02 project page: https://notion.l2k.tech:28443/api/report-media/server-upload/notionnext-videos/interview-portfolio/20260708/project-homepages/p02-parol6/index.html

This repository is one public support entry for the P02 PAROL6 control-stack project. It is suitable for discussing the public VLM/API workflow, while private robot connection details, calibration logs, safety gates, credentials, and unredacted real-world media remain outside this public repository.

## Demo / Evidence Entry Points

- [`01_PAROL6完整功能测试教程.ipynb`](01_PAROL6完整功能测试教程.ipynb): notebook-style end-to-end test walkthrough.
- [`22_vlm_robot_test.py`](22_vlm_robot_test.py) and [`23_vlm_full_test.py`](23_vlm_full_test.py): VLM-to-robot control test entry points.
- [`25_vlm_mujoco_control.py`](25_vlm_mujoco_control.py) and [`26_vlm_mujoco_grasp.py`](26_vlm_mujoco_grasp.py): MuJoCo-side control and grasp validation examples.
- [`运行核验报告.md`](运行核验报告.md): current public verification report.
- [`项目架构图.mmd`](项目架构图.mmd): Mermaid architecture source for the control pipeline.
- [`README_en.md`](README_en.md) and [`API-Specific-README.md`](API-Specific-README.md): English overview and API-focused reference.

## Public Boundary

This public repository intentionally excludes API keys, `.env` values, private robot network addresses, serial/CAN identifiers, private calibration logs, hardware safety gate settings, and unredacted real-world image/video data. Use the checked-in examples and reports as public-facing evidence, not as a complete private lab deployment snapshot.

## 📋 项目概述

这是一个基于 Gemini Vision AI 的 PAROL6 机械臂智能控制系统，能够通过自然语言和视觉识别实现机器人的自动化操作。该系统将传统的 GUI 控制方式转变为可编程的研究级平台，支持视觉引导操作和自然语言控制。

### 核心特性
- ✅ **自然语言控制**：支持文本和语音输入
- ✅ **视觉识别**：集成 Intel RealSense D435I 深度相机
- ✅ **AI 驱动**：使用 Google Gemini 2.5 Flash 视觉模型
- ✅ **精确控制**：毫米级定位精度
- ✅ **远程操作**：UDP 客户端-服务器架构
- ✅ **安全保护**：多层安全检查和急停监控

---

## 🔧 系统架构

### 三层架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    用户交互层 (顶层)                          │
│  - 自然语言输入（文本/语音）                                    │
│  - Gemini AI 理解和规划                                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   视觉处理层 (中层)                           │
│  - RealSense 深度相机                                        │
│  - 物体检测和3D定位                                           │
│  - 手眼标定                                                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  机器人控制层 (底层)                          │
│  - UDP 服务器（headless_commander.py）                       │
│  - 运动学计算和轨迹规划                                       │
│  - 串口通信和实时控制 (100Hz)                                 │
└─────────────────────────────────────────────────────────────┘
```

### 文件结构
```
PAROL6-Python-API-Gemini-Vision-Public/
├── Headless/
│   ├── headless_commander.py      # 机器人控制服务器
│   ├── robot_api.py                # 客户端 API（30+ 控制函数）
│   ├── smooth_motion.py            # 平滑轨迹生成
│   ├── PAROL6_ROBOT.py             # 机器人参数配置
│   ├── Gemini/
│   │   ├── gemini.py               # Gemini AI 控制器
│   │   ├── robot_vision_tools.py   # 视觉工具函数
│   │   ├── bounding_box_model.py   # 物体检测模型
│   │   └── supplementary_functions.py  # 辅助功能
│   └── Vision/
│       ├── vision_controller.py    # 视觉处理控制器
│       ├── calibrate_hand_eye.py   # 手眼标定
│       └── validate_calibration.py # 标定验证
├── README.md                       # 英文项目说明
├── API-Specific-README.md          # API 详细文档
└── 中文使用说明.md                  # 本文档
```

---

## 📥 输入与输出

### 输入
| 输入类型 | 说明 | 示例 |
|---------|------|------|
| **自然语言命令** | 文本或语音描述任务 | "请抓取红色的方块并放到蓝色盒子里" |
| **视觉数据** | RealSense D435I 深度相机 | RGB-D 图像流（640×480 @ 30fps）|
| **API 调用** | Python 函数直接控制 | `move_robot_joints([90, -45, 90, 0, 45, 180])` |
| **关节角度** | 6个关节的角度值（度） | `[0, -90, 180, 0, 0, 180]` |
| **笛卡尔坐标** | 位置和姿态 (mm 和度) | `[250, 0, 200, 180, 0, 90]` |

### 输出
| 输出类型 | 说明 | 精度 |
|---------|------|------|
| **关节运动** | 6轴协调运动 | ±0.1° |
| **笛卡尔运动** | 直线/曲线路径 | ±1mm |
| **夹爪控制** | 开合和力控制 | 0-255 级位置 |
| **状态反馈** | 位置、速度、IO状态 | 实时更新 |
| **视觉信息** | 物体3D位置和方向 | 毫米级精度 |
| **语音/文本反馈** | AI 响应和状态报告 | 自然语言 |

---

## 🎯 主要功能

### 1. 基础运动控制
- **关节运动** (`move_robot_joints`): 控制6个关节到指定角度
- **位置运动** (`move_robot_pose`): 移动到笛卡尔坐标位置
- **笛卡尔直线** (`move_robot_cartesian`): 保证直线路径运动
- **点动控制** (`jog_robot_joint`, `jog_cartesian`): 手动微调位置

### 2. 高级平滑运动
- **圆形轨迹** (`smooth_circle`): 在指定平面画圆
- **圆弧运动** (`smooth_arc_center`, `smooth_arc_parametric`): 精确圆弧路径
- **样条曲线** (`smooth_spline`): 通过多个路点的平滑曲线
- **螺旋运动** (`smooth_helix`): 螺旋式运动路径
- **混合运动** (`smooth_blend`): 多段运动平滑衔接

### 3. 视觉引导操作
- **物体检测**: 使用 Gemini Vision AI 识别物体
- **3D 定位**: 深度相机获取物体空间坐标
- **智能抓取**: 自动规划抓取姿态和路径
- **精确放置**: 视觉引导的放置操作
- **手眼标定**: ChArUco 标定板标定系统

### 4. 夹爪控制
- **电动夹爪**: 位置、速度、力控制 (0-255)
- **气动夹爪**: 开/合控制（数字输出端口）
- **物体检测**: 夹爪状态反馈

### 5. 自然语言控制
- **文本模式**: 逐步文本命令控制
- **语音模式**: 按住空格键语音输入
- **自主模式**: AI 自动分解复杂任务
- **组合功能**: 多步骤操作自动执行

---

## 🚀 快速开始

### 安装依赖

#### 服务器端（连接机器人的电脑）
```bash
# 核心依赖
pip3 install roboticstoolbox-python==1.0.3
pip3 install numpy==1.23.4
pip3 install scipy==1.11.4
pip3 install spatialmath

# 串口和计时
pip3 install pyserial
pip3 install oclock
pip3 install keyboard

# 视觉处理
pip3 install pyrealsense2
pip3 install opencv-python

# Gemini AI
pip3 install google-genai
pip3 install python-dotenv

# 音频处理（可选，用于语音输入）
pip3 install pyaudio
```

#### 客户端（任意电脑）
```bash
# 仅需要标准库，如果需要姿态矩阵功能：
pip3 install numpy==1.23.4
pip3 install spatialmath
```

### 配置 API 密钥

在项目根目录创建 `.env` 文件：
```bash
GEMINI_API_KEY=你的_Gemini_API_密钥
```

获取 API 密钥: https://makersuite.google.com/app/apikey

---

## 📖 使用方法

### 方式一：自然语言控制（推荐）

#### 1. 启动机器人控制服务器
```bash
cd Headless
python headless_commander.py
```
等待提示：`✓ Robot ready` 和 `Listening for commands on port 5001`

#### 2. 启动 Gemini AI 控制器

**文本模式**（推荐新手）:
```bash
cd Headless/Gemini
python gemini.py --mode text
```

**自主模式**（复杂任务）:
```bash
python gemini.py --mode autonomous
```

**语音输入模式**:
```bash
python gemini.py --mode text --input audio
```

#### 3. 发送命令示例

**文本模式下的命令示例：**
```
你: 请向前移动 50mm
AI: [执行移动并确认]

你: 抓取红色的方块
AI: [检测物体 → 规划路径 → 抓取]

你: 把它放到蓝色盒子里
AI: [移动到目标 → 放置 → 松开]
```

**自主模式下的复杂任务：**
```
你: 清理桌面上所有的红色物体，把它们放到绿色容器里
AI: [自动执行]
    1. 扫描检测所有红色物体
    2. 逐个抓取
    3. 放置到容器
    4. 返回检测更多物体
    5. 完成并报告
```

---

### 方式二：Python API 编程

创建控制脚本 `my_control.py`:

```python
from Headless.robot_api import *

# 1. 机器人回零
home_robot()

# 2. 移动到指定关节角度（5.5秒完成）
move_robot_joints([90, -45, 90, 0, 45, 180], duration=5.5)

# 3. 移动到笛卡尔坐标（位置：x,y,z 姿态：Rx,Ry,Rz）
move_robot_pose([250, 0, 200, 180, 0, 90], speed_percentage=50)

# 4. 直线运动到目标点
move_robot_cartesian([200, -50, 180, 180, 0, 135], duration=4.0)

# 5. 画圆（中心点，半径，平面，时间）
smooth_circle(center=[200, 0, 200], radius=50, plane='XY', duration=5.0)

# 6. 控制夹爪
control_electric_gripper(action='open')
control_electric_gripper(action='move', position=200, speed=150)

# 7. 查询状态
pose = get_robot_pose()
print(f"当前位置: {pose}")

angles = get_robot_joint_angles()
print(f"当前关节角度: {angles}")

# 8. 执行轨迹
trajectory = [
    [200, 0, 200, 0, 0, 0],
    [250, 50, 200, 0, 0, 45],
    [200, 100, 200, 0, 0, 90]
]
execute_trajectory(trajectory, timing_mode='duration',
                   timing_value=10.0, motion_type='spline')
```

运行：
```bash
python my_control.py
```

---

### 方式三：视觉引导抓取

使用视觉工具进行智能抓取：

```python
from Headless.Gemini.robot_vision_tools import RobotVisionTools
from Headless.Vision.vision_controller import VisionController

# 初始化视觉系统
vision = VisionController()
vision.start()

robot_vision = RobotVisionTools(vision)

# 智能抓取流程
result = robot_vision.pick_object(
    object_description="红色的立方体",
    approach_style="vertical"  # 垂直抓取
)

if result['success']:
    # 放置到指定位置
    robot_vision.place_at_position(
        target_position=[300, 100, 50],  # 目标位置
        approach_height=50  # 接近高度
    )
```

---

## 🔍 详细 API 说明

### 基础运动命令

#### 1. 关节运动
```python
move_robot_joints(
    joint_angles=[90, -45, 90, 0, 45, 180],  # 6个关节角度（度）
    duration=5.0,              # 运动时间（秒）
    # 或者使用速度控制：
    # speed_percentage=75,     # 速度百分比 (0-100)
    wait_for_ack=True,         # 等待完成确认
    timeout=10.0               # 超时时间
)
```

#### 2. 笛卡尔位置运动
```python
move_robot_pose(
    pose=[250, 0, 200, 180, 0, 90],  # [x, y, z, Rx, Ry, Rz]
    speed_percentage=50,              # 速度百分比
    wait_for_ack=True
)
```

#### 3. 直线运动（保证路径）
```python
move_robot_cartesian(
    pose=[200, -50, 180, 180, 0, 135],
    duration=4.0
)
```

#### 4. 点动控制
```python
# 单关节点动
jog_robot_joint(
    joint_index=0,        # 关节索引 (0-5正向, 6-11负向)
    speed_percentage=40,
    duration=2.0          # 时间
    # 或者：distance_deg=15  # 角度
)

# 多关节同时点动
jog_multiple_joints(
    joints=[0, 3, 5],          # 关节列表
    speeds=[70, 40, 60],       # 对应速度
    duration=1.2
)

# 笛卡尔点动
jog_cartesian(
    frame='TRF',              # 'TRF'工具坐标系 或 'WRF'世界坐标系
    axis='Z+',                # 'X+', 'X-', 'Y+', 'Y-', 'Z+', 'Z-'
    speed_percentage=50,
    duration=1.5
)
```

### 高级平滑运动

#### 1. 圆形轨迹
```python
smooth_circle(
    center=[200, 0, 200],    # 圆心坐标 (mm)
    radius=50,               # 半径 (mm)
    plane='XY',              # 平面: 'XY', 'XZ', 'YZ'
    duration=5.0,
    clockwise=False          # 顺时针/逆时针
)
```

#### 2. 圆弧运动（中心点定义）
```python
smooth_arc_center(
    end_pose=[250, 50, 200, 0, 0, 90],  # 终点姿态
    center=[200, 0, 200],                # 圆弧中心
    duration=3.0
)
```

#### 3. 圆弧运动（参数定义）
```python
smooth_arc_parametric(
    end_pose=[250, 50, 200, 0, 0, 90],
    radius=50,           # 半径 (mm)
    arc_angle=90,        # 圆弧角度 (度)
    duration=3.0
)
```

#### 4. 样条曲线
```python
waypoints = [
    [200, 0, 100, 0, 0, 0],
    [250, 50, 150, 0, 15, 45],
    [200, 100, 200, 0, 30, 90]
]
smooth_spline(waypoints, duration=8.0)
```

#### 5. 螺旋运动
```python
smooth_helix(
    center=[200, 0, 150],   # 螺旋中心
    radius=30,              # 半径
    pitch=20,               # 螺距（每圈高度）
    height=100,             # 总高度
    duration=10.0
)
```

#### 6. 混合运动（多段衔接）
```python
segments = [
    {'type': 'LINE', 'end': [250, 0, 200, 0, 0, 0], 'duration': 2.0},
    {'type': 'CIRCLE', 'center': [250, 0, 200], 'radius': 50,
     'plane': 'XY', 'duration': 4.0},
    {'type': 'LINE', 'end': [200, 0, 200, 0, 0, 0], 'duration': 2.0}
]
smooth_blend(segments, blend_time=0.5, duration=10.0)
```

### 夹爪控制

#### 电动夹爪
```python
# 校准夹爪
control_electric_gripper(action='calibrate')

# 移动到指定位置
control_electric_gripper(
    action='move',
    position=200,    # 位置 (0-255)
    speed=150,       # 速度 (0-255)
    current=500      # 最大电流 (100-1000 mA)
)

# 查询夹爪状态
status = get_electric_gripper_status(verbose=True)
# 返回: [ID, Position, Speed, Current, StatusByte, ObjectDetected]
```

#### 气动夹爪
```python
# 打开夹爪
control_pneumatic_gripper(action='open', port=1)

# 关闭夹爪
control_pneumatic_gripper(action='close', port=1)
```

### 查询命令

```python
# 获取当前位置姿态
pose = get_robot_pose()  # 返回 [x, y, z, Rx, Ry, Rz]

# 获取当前关节角度
angles = get_robot_joint_angles()  # 返回 6个角度值

# 获取关节速度
speeds = get_robot_joint_speeds()  # 步/秒

# 获取 IO 状态
io_status = get_robot_io(verbose=True)
# 返回: [IN1, IN2, OUT1, OUT2, ESTOP]

# 检查是否停止
is_stopped = is_robot_stopped(threshold_speed=2.0)

# 检查急停状态
estop = is_estop_pressed()

# 获取完整状态
status = get_robot_status()
# 包含: pose, angles, speeds, IO, gripper, stopped, estop
```

### 辅助功能

```python
# 延时
delay_robot(2.5)  # 延时 2.5 秒

# 停止运动
stop_robot_movement()  # 立即停止并清空命令队列

# 回零
home_robot()

# 等待停止
wait_for_robot_stopped(timeout=10.0, poll_rate=0.1)

# 带重试的安全移动
safe_move_with_retry(
    move_robot_joints,
    [90, -45, 90, 0, 45, 180],
    duration=5.0,
    max_retries=3,
    retry_delay=1.0
)
```

---

## 🎓 实际应用示例

### 示例 1: 简单的抓放任务

```python
from Headless.robot_api import *

# 1. 初始化
home_robot()

# 2. 移动到观察位置
move_robot_joints([0, -90, 180, 0, -45, 180], duration=3.0)

# 3. 打开夹爪
control_electric_gripper(action='open')

# 4. 移动到抓取位置
move_robot_cartesian([200, 0, 100, 180, 0, 0], duration=2.0)

# 5. 关闭夹爪抓取
control_electric_gripper(action='move', position=200, speed=100, current=500)
delay_robot(1.0)

# 6. 抬起
move_robot_cartesian([200, 0, 200, 180, 0, 0], duration=2.0)

# 7. 移动到放置位置
move_robot_cartesian([300, 100, 200, 180, 0, 0], duration=3.0)

# 8. 下降
move_robot_cartesian([300, 100, 110, 180, 0, 0], duration=2.0)

# 9. 松开
control_electric_gripper(action='open')

# 10. 返回安全位置
home_robot()
```

### 示例 2: 使用视觉引导抓取

```python
from Headless.Gemini.robot_vision_tools import RobotVisionTools
from Headless.Vision.vision_controller import VisionController

# 初始化
vision = VisionController()
vision.start()
robot_vision = RobotVisionTools(vision)

# 执行抓取序列
objects_to_pick = ["红色方块", "蓝色圆柱", "绿色三角形"]

for obj_name in objects_to_pick:
    # 检测并抓取
    result = robot_vision.pick_object(
        object_description=obj_name,
        approach_style="vertical"
    )

    if result['success']:
        print(f"成功抓取 {obj_name}")

        # 放置到分类区域
        robot_vision.place_at_position(
            target_position=[300, 150, 50],
            approach_height=50
        )
        print(f"{obj_name} 已放置")
    else:
        print(f"抓取 {obj_name} 失败: {result['message']}")

vision.stop()
```

### 示例 3: 绘制图案

```python
from Headless.robot_api import *

# 绘制正方形
def draw_square(center, size, height, duration):
    half_size = size / 2
    corners = [
        [center[0] + half_size, center[1] + half_size, height, 0, 0, 0],
        [center[0] - half_size, center[1] + half_size, height, 0, 0, 0],
        [center[0] - half_size, center[1] - half_size, height, 0, 0, 0],
        [center[0] + half_size, center[1] - half_size, height, 0, 0, 0],
        [center[0] + half_size, center[1] + half_size, height, 0, 0, 0]
    ]
    execute_trajectory(corners, timing_mode='duration',
                      timing_value=duration, motion_type='linear')

# 绘制圆形
def draw_circle(center, radius, height, duration):
    smooth_circle(center=[center[0], center[1], height],
                 radius=radius,
                 plane='XY',
                 duration=duration)

# 执行
home_robot()
draw_square(center=[200, 0], size=80, height=150, duration=8.0)
delay_robot(1.0)
draw_circle(center=[200, 0], radius=40, height=150, duration=5.0)
```

### 示例 4: 组装任务

```python
from Headless.robot_api import *

def pick_and_stack(objects_positions, stack_position, stack_height_increment=30):
    """
    从多个位置拾取物体并堆叠
    """
    current_stack_height = 50  # 初始高度

    for i, obj_pos in enumerate(objects_positions):
        print(f"拾取物体 {i+1}/{len(objects_positions)}")

        # 打开夹爪
        control_electric_gripper(action='open')

        # 移动到物体上方
        approach_pos = obj_pos.copy()
        approach_pos[2] += 100
        move_robot_cartesian(approach_pos, duration=2.0)

        # 下降抓取
        move_robot_cartesian(obj_pos, duration=1.5)
        control_electric_gripper(action='move', position=200, speed=80)
        delay_robot(1.0)

        # 抬起
        move_robot_cartesian(approach_pos, duration=1.5)

        # 移动到堆叠位置上方
        stack_approach = stack_position.copy()
        stack_approach[2] = current_stack_height + 100
        move_robot_cartesian(stack_approach, duration=3.0)

        # 下降放置
        place_pos = stack_position.copy()
        place_pos[2] = current_stack_height
        move_robot_cartesian(place_pos, duration=1.5)

        # 松开
        control_electric_gripper(action='open')
        delay_robot(0.5)

        # 抬起
        move_robot_cartesian(stack_approach, duration=1.5)

        # 更新堆叠高度
        current_stack_height += stack_height_increment

    print("堆叠完成！")

# 使用示例
objects = [
    [200, 100, 50, 180, 0, 0],
    [200, -100, 50, 180, 0, 0],
    [250, 0, 50, 180, 0, 0]
]
stack_location = [150, 0, 50, 180, 0, 0]

home_robot()
pick_and_stack(objects, stack_location, stack_height_increment=25)
home_robot()
```

---

## 🔧 手眼标定流程

视觉引导操作需要先进行手眼标定，将相机坐标系与机器人坐标系对齐。

### 1. 生成标定板

```bash
cd Headless/Vision
python generate_charuco_board.py
```
打印生成的 ChArUco 标定板（A4 纸）

### 2. 执行标定

```bash
python calibrate_hand_eye.py
```

按照提示：
1. 将标定板放在相机视野内
2. 机器人自动移动到多个位置采集数据
3. 自动计算手眼变换矩阵
4. 保存标定结果到 `Results/calibration/` 目录

### 3. 验证标定

```bash
python validate_calibration.py
```

检查标定精度，应该达到 ±3mm 以内

### 4. 优化标定（可选）

```bash
python calibration_refinement.py
```

通过迭代优化提高精度

---

## ⚙️ 配置说明

### 网络配置

如果客户端和服务器不在同一台电脑：

编辑 `Headless/robot_api.py`:
```python
SERVER_IP = "192.0.2.10"     # 示例地址；改为你的机器人控制器 IP
SERVER_PORT = 5001           # 默认端口
```

确保防火墙允许 UDP 5001 和 5002 端口

### 相机配置

编辑 `Headless/Vision/vision_controller.py`:
```python
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
```

### 运动参数调整

编辑 `Headless/Gemini/robot_vision_tools.py`:
```python
APPROACH_HEIGHT = 50      # 接近高度 (mm)
GRASP_HEIGHT = 100        # 抓取后抬升高度 (mm)
SPEED_APPROACH = 40       # 接近速度 (%)
SPEED_GRASP = 20          # 抓取速度 (%)
```

---

## 🛡️ 安全注意事项

### 硬件安全
1. ⚠️ **急停按钮**: 确保急停按钮随时可触及
2. ⚠️ **工作空间**: 保持工作区域无障碍物
3. ⚠️ **电源**: 使用稳定的电源供应
4. ⚠️ **固定**: 确保机器人底座固定牢固

### 软件安全
- ✓ 系统自动检查关节限位
- ✓ 运动前验证逆运动学解
- ✓ 速度同步防止超速
- ✓ 急停触发立即停止所有运动
- ✓ 命令队列自动清空

### 操作规范
1. 首次使用先在低速度测试
2. 确认工作空间范围
3. 观察首次运动路径
4. 保持监控运行状态
5. 异常情况立即按急停

---

## 🐛 故障排除

### 常见问题

#### 1. 串口连接失败
**现象**: `Serial port not found`

**解决**:
- Windows: 检查设备管理器中的 COM 端口号
- Linux: 检查 `/dev/ttyUSB*` 或 `/dev/ttyACM*`
- 更新 `com_port.txt` 文件中的端口号
- 检查串口权限: `sudo usermod -a -G dialout $USER`

#### 2. UDP 通信失败
**现象**: 命令无响应

**解决**:
- 检查服务器是否启动
- 确认 IP 地址和端口正确
- 检查防火墙设置
- 测试网络连通性: `ping 服务器IP`

#### 3. 运动学求解失败
**现象**: `IK solution not found`

**解决**:
- 目标位置可能超出工作空间范围
- 检查姿态角度是否合理
- 尝试从当前位置分步移动
- 使用 `get_robot_pose()` 查看可达范围

#### 4. 相机无法启动
**现象**: `RealSense camera not found`

**解决**:
- 检查 USB 连接（建议 USB 3.0）
- 更新 RealSense 驱动
- 运行 `realsense-viewer` 测试相机
- 检查相机权限

#### 5. Gemini API 错误
**现象**: `API key invalid`

**解决**:
- 检查 `.env` 文件中的 API 密钥
- 确认 API 密钥未过期
- 检查网络连接
- 验证 API 配额

#### 6. 标定精度差
**现象**: 抓取位置偏移

**解决**:
- 重新进行手眼标定
- 增加标定采样点数量
- 确保标定板打印比例准确
- 使用 `validate_calibration.py` 验证
- 尝试 `calibration_refinement.py` 优化

---

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| **控制频率** | 100 Hz |
| **定位精度** | ±1 mm |
| **重复精度** | ±0.5 mm |
| **最大负载** | 1.2 kg |
| **最大速度** | 100%（关节相关）|
| **工作半径** | 100-400 mm |
| **响应延迟** | <50 ms (局域网) |
| **视觉帧率** | 30 fps |
| **Gemini 响应** | 1-3 秒 |

---

## 🎯 使用场景

### 适用场景
✅ 桌面物体分拣
✅ 视觉引导装配
✅ 自动化测试
✅ 教学演示
✅ 研究实验
✅ 原型开发

### 不适用场景
❌ 高速生产线（速度限制）
❌ 精密微米级加工
❌ 重载工业应用
❌ 长时间连续运行（建议<4小时/次）

---

## 🔗 相关资源

- **PAROL6 官方**: https://github.com/PCrnjak/PAROL-commander-software
- **Discord 社区**: https://discord.com/invite/prjUvjmGpZ
- **Gemini API**: https://ai.google.dev/gemini-api/docs
- **RealSense SDK**: https://github.com/IntelRealSense/librealsense
- **问题反馈**: 项目 Issues 页面

---

## 📄 许可证

请参考项目根目录的 LICENSE 文件

---

## 🙏 致谢

- PAROL6 创建者及社区
- Google Gemini AI 团队
- Intel RealSense 团队
- 所有贡献者和测试者

---

## 📮 联系方式

如有问题或建议：
1. 提交 GitHub Issue
2. 加入 PAROL6 Discord 频道
3. 查看详细 API 文档：`API-Specific-README.md`

---

**祝使用愉快！🤖**
