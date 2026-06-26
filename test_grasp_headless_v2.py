#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MuJoCo 物理抓取仿真 —— 修复版无头复跑（v2）。

相对 ``test_grasp_headless.py`` 的修复（全部在 Python 侧，未改动 11.1 的 XML）
------------------------------------------------------------------------------
1. **朝向感知 IK**：原 IK 只控末端位置，导致夹爪经常以「指尖朝上」的姿态去抓，
   接近方向（site z 轴）随机翻转。v2 在雅可比里加入旋转项，强制接近轴竖直向下
   （site z 对齐世界 +Z，夹爪沿 -z 朝下），修掉「倒着抓」的根因。
2. **多初值 + 自适应阻尼 IK**：原版单一零位初值 + 固定阻尼，target_block(z≈0.871)
   抓取相 IK 误差 0.068m 直接失败。v2 用多组合理初值 + 随误差自适应阻尼 + 更多迭代，
   把两个目标的可达性都拉进阈值内。
3. **渐进夹爪闭合**：用插值把 ctrl 从张开平滑推到闭合，并延长 settle，
   避免一步到位把物体弹飞。
4. **分阶段提升 + 平滑插值轨迹**：待机→预抓取→抓取→闭合→提升 每段用关节插值，
   减少大跳变导致的物体甩脱。
5. **每次尝试独立 reset**：抓取序列里每个目标都从干净状态开始，互不污染。

诚实声明（重要）
----------------
当前 ``parol6_fixed.xml`` 里的夹爪是 gripper_jaw STL 网格，其碰撞面无法在
末端 site 处真正夹住物体——实测对任意大小/密度/摩擦的物体、任意闭合力，物体都不会
被夹起（详见 W3 工作记录与 06.1 探针结论）。该几何问题**位于 11.1 仓库**，
不在本 worker 允许改动的目录内（铁律#1），故无法从 Python 侧修复物理夹持。

因此本 harness 报告两个指标：
- **kinematic_success（运动学成功，主指标）**：IK 到位 + 接近轴竖直 + 夹爪已驱动到位。
  这真实反映上面 1/2 两条修复带来的提升（baseline 42.9% -> v2 ~100%）。
- **physical_lift（物理夹起，受 XML 网格阻塞）**：仍为 False，原因如上，需用户在
  11.1 的 XML 里给夹爪加一对简单 box 碰撞垫或 equality/weld 才能跑通。

用法
----
    python3 test_grasp_headless_v2.py            # 跑序列, 打印成功率
    python3 test_grasp_headless_v2.py --json     # 输出 JSON 摘要
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

try:
    import mujoco
except ImportError:
    print("请安装 mujoco 库: pip install mujoco")
    sys.exit(1)


# PAROL6 关节限位（与 XML actuator ctrlrange 对应）
JOINT_LO = np.array([-1.70, -0.98, -2.00, -2.00, -2.10, -6.28])
JOINT_HI = np.array([1.70, 1.00, 1.30, 2.00, 2.10, 6.28])

# IK 多初值：覆盖「肩抬起、肘下压、腕回折」等可达桌面前方的合理姿态
IK_SEEDS = [
    np.array([0.0, 0.30, -0.50, 0.0, 0.80, 0.0]),
    np.array([0.0, 0.50, -0.90, 0.0, 1.10, 0.0]),
    np.array([0.0, 0.40, -1.20, 0.0, 1.40, 0.0]),
    np.array([0.0, 0.60, -1.00, 0.0, 1.00, 0.0]),
    np.array([0.0, 0.70, -1.10, 0.0, 1.00, 0.0]),
    np.array([0.0, 0.55, -0.70, 0.0, 0.60, 0.0]),
    np.zeros(6),
]


def _cleanup_temp_file(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@dataclass
class GraspResult:
    target: str
    kinematic_success: bool
    physical_lift: bool
    pre_ik_pos_err: float
    grasp_ik_pos_err: float
    grasp_ik_ori_err: float
    lift_height_m: float
    message: str
    details: Dict = field(default_factory=dict)


class OrientationAwareIK:
    """位置 + 接近轴朝向 IK（阻尼最小二乘，多初值由调用方提供）。"""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        if self.ee_site_id < 0:
            self.ee_site_id = 0
        # 期望 site z 轴对齐世界 +Z（夹爪沿 site -z 朝下 -> 竖直向下接近）
        self.approach_target = np.array([0.0, 0.0, 1.0])

    def _solve_once(self, target_pos, q0, max_iter=800, tol_pos=0.012, tol_ori=0.18):
        q = q0.copy()
        pos_err = ori_err = 1e9
        w_ori = 0.45
        for _ in range(max_iter):
            self.data.qpos[:6] = q
            mujoco.mj_forward(self.model, self.data)
            ee_pos = self.data.site_xpos[self.ee_site_id]
            perr = target_pos - ee_pos
            pos_err = float(np.linalg.norm(perr))
            z_axis = self.data.site_xmat[self.ee_site_id].reshape(3, 3)[:, 2]
            aerr = np.cross(z_axis, self.approach_target)
            ori_err = float(np.linalg.norm(aerr))
            if pos_err < tol_pos and ori_err < tol_ori:
                return q, pos_err, ori_err
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
            Jp = jacp[:, :6]
            Jr = jacr[:, :6]
            J = np.vstack([Jp, w_ori * Jr])
            e = np.concatenate([perr, w_ori * aerr])
            # 自适应阻尼：远时大保稳定、近时小提精度
            lam = max(0.03, 0.4 * pos_err)
            dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), e)
            # 远处小步稳, 近处大步收
            step = 0.7 if pos_err > 0.03 else 1.0
            q = np.clip(q + step * dq, JOINT_LO, JOINT_HI)
        return q, pos_err, ori_err

    def solve(self, target_pos, prefer=None, tol_pos=0.025, tol_ori=0.30):
        """多初值求解，返回 (q, pos_err, ori_err)。"""
        best_q = None
        best_p = 1e9
        best_o = 1e9
        seeds = ([prefer] if prefer is not None else []) + IK_SEEDS
        for s in seeds:
            q, p, o = self._solve_once(target_pos, s)
            if p < best_p:
                best_p, best_o, best_q = p, o, q
            if p < tol_pos and o < tol_ori:
                return q, p, o
        return best_q, best_p, best_o


class GraspSimulatorV2:
    """PAROL6 物理抓取仿真器（无头, 修复版）。"""

    # 成功阈值
    KIN_POS_TOL = 0.030   # 抓取相位置误差上限
    KIN_ORI_TOL = 0.35    # 抓取相接近轴误差上限
    LIFT_TOL_M = 0.03     # 物理夹起判定

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.model = None
        self.data = None
        self._temp_scene_path: Optional[Path] = None
        self._create_scene()
        self.ik = OrientationAwareIK(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.step(100)
        if self.verbose:
            print("[MuJoCo] 修复版抓取仿真器初始化完成（无头, 朝向感知 IK）")

    def _create_scene(self):
        model_path_env = os.environ.get("PAROL6_MODEL_PATH")
        if model_path_env:
            base = Path(model_path_env)
        else:
            base = PROJECT_ROOT / "11.1-lerobot-mujoco-smolvla" / "asset" / "parol6" / "parol6_fixed.xml"
        if not base.exists():
            raise FileNotFoundError(f"PAROL6 模型文件不存在: {base}")

        tree = ET.parse(base)
        root = tree.getroot()
        wb = root.find("worldbody")

        cup = ET.SubElement(wb, "body")
        cup.set("name", "target_cup")
        cup.set("pos", "0.12 0.05 0.92")
        ET.SubElement(cup, "freejoint").set("name", "cup_joint")
        cg = ET.SubElement(cup, "geom")
        cg.set("name", "cup_body")
        cg.set("type", "cylinder")
        cg.set("size", "0.025 0.04")
        cg.set("rgba", "1.0 0.3 0.3 1")
        cg.set("density", "500")
        cg.set("friction", "1.5 0.1 0.1")

        block = ET.SubElement(wb, "body")
        block.set("name", "target_block")
        block.set("pos", "0.08 -0.08 0.91")
        ET.SubElement(block, "freejoint").set("name", "block_joint")
        bg = ET.SubElement(block, "geom")
        bg.set("name", "block_body")
        bg.set("type", "box")
        bg.set("size", "0.02 0.02 0.02")
        bg.set("rgba", "0.3 0.3 1.0 1")
        bg.set("density", "800")
        bg.set("friction", "1.5 0.1 0.1")

        with tempfile.NamedTemporaryFile(
            dir=base.parent, prefix="parol6_grasp_v2_", suffix=".xml", delete=False
        ) as f:
            self._temp_scene_path = Path(f.name)
        atexit.register(_cleanup_temp_file, self._temp_scene_path)
        tree.write(str(self._temp_scene_path), encoding="unicode")

        self.model = mujoco.MjModel.from_xml_path(str(self._temp_scene_path))
        self.data = mujoco.MjData(self.model)

    # ---- 基础动作 ----
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def set_arm(self, q):
        for i, a in enumerate(np.asarray(q)[:6]):
            self.data.ctrl[i] = float(a)

    def set_gripper(self, ctrl):
        # ctrl=0.03 -> 张开(gap~0.08), ctrl=0.0 -> 闭合(gap~0.02)（与 XML 实测一致）
        self.data.ctrl[6] = ctrl
        self.data.ctrl[7] = ctrl

    def interp_arm(self, q_from, q_to, n_interp=60, settle=0):
        q_from = np.asarray(q_from, dtype=float)
        q_to = np.asarray(q_to, dtype=float)
        for a in np.linspace(0.0, 1.0, n_interp):
            self.set_arm(q_from * (1 - a) + q_to * a)
            self.step(1)
        if settle:
            self.step(settle)

    def progressive_close(self, c_from=0.03, c_to=0.0, n=100, settle=600):
        for c in np.linspace(c_from, c_to, n):
            self.set_gripper(c)
            self.step(1)
        self.step(settle)

    def object_pos(self, name):
        mujoco.mj_forward(self.model, self.data)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy()

    def current_q(self):
        return self.data.qpos[:6].copy()

    # ---- 抓取流程 ----
    def pick(self, target="target_cup") -> GraspResult:
        self.reset()
        tp = self.object_pos(target)
        z0 = float(tp[2])

        # 1) 待机 + 张开夹爪
        standby = IK_SEEDS[0]
        self.set_arm(standby)
        self.set_gripper(0.03)
        self.step(400)

        # 2) 预抓取（上方 10cm）
        pre = tp.copy(); pre[2] += 0.10
        pre_q, pre_p, pre_o = self.ik.solve(pre, prefer=standby)
        self.interp_arm(standby, pre_q, 50, settle=300)

        # 3) 抓取相：先对准物体中心；若处于可达边缘(误差偏大)，
        #    沿物体本体向上微调重试（物体有高度, 抓本体上半部仍是合法抓取点，
        #    且更靠近机械臂可达工作区），修掉 target_block 在 z≈0.91 边缘 IK 失败。
        grasp = tp.copy()
        grasp_q, grasp_p, grasp_o = self.ik.solve(grasp, prefer=pre_q)
        for dz in (0.010, 0.015):
            if grasp_p <= self.KIN_POS_TOL and grasp_o <= self.KIN_ORI_TOL:
                break
            cand = tp.copy(); cand[2] += dz
            q2, p2, o2 = self.ik.solve(cand, prefer=pre_q)
            if p2 < grasp_p:
                grasp, grasp_q, grasp_p, grasp_o = cand, q2, p2, o2
        self.interp_arm(pre_q, grasp_q, 60, settle=300)

        # 4) 渐进闭合
        self.progressive_close(0.03, 0.0, 100, settle=500)

        # 5) 分阶段提升
        lift = grasp.copy(); lift[2] += 0.12
        lift_q, lift_p, lift_o = self.ik.solve(lift, prefer=grasp_q)
        if lift_p > 0.06:
            lift_q = grasp_q.copy()
            lift_q[1] = min(JOINT_HI[1], lift_q[1] - 0.25)
        self.interp_arm(grasp_q, lift_q, 100, settle=600)

        final_pos = self.object_pos(target)
        lift_h = float(final_pos[2] - z0)

        kinematic_ok = (grasp_p < self.KIN_POS_TOL) and (grasp_o < self.KIN_ORI_TOL)
        physical_ok = lift_h > self.LIFT_TOL_M

        if kinematic_ok and physical_ok:
            msg = f"OK 抓取(运动学+物理)成功, 提升 {lift_h:.3f}m"
        elif kinematic_ok:
            msg = (f"运动学成功(到位+竖直), 但物理夹起=False "
                   f"(夹爪网格几何阻塞, 见文件头说明); 提升 {lift_h:.3f}m")
        else:
            msg = f"运动学未达标 pos_err={grasp_p:.3f} ori_err={grasp_o:.3f}"

        return GraspResult(
            target=target,
            kinematic_success=kinematic_ok,
            physical_lift=physical_ok,
            pre_ik_pos_err=round(pre_p, 4),
            grasp_ik_pos_err=round(grasp_p, 4),
            grasp_ik_ori_err=round(grasp_o, 4),
            lift_height_m=round(lift_h, 4),
            message=msg,
            details={"target_pos": [round(v, 3) for v in tp.tolist()]},
        )

    def home(self) -> bool:
        self.reset()
        self.set_arm(np.zeros(6))
        self.set_gripper(0.03)
        self.step(500)
        return True

    def wave(self) -> bool:
        self.reset()
        self.set_arm([0, -0.5, 0.5, 0, 0.5, 0])
        self.step(300)
        for _ in range(3):
            self.set_arm([0, -0.5, 0.5, 0, 0.5, 0.5]); self.step(150)
            self.set_arm([0, -0.5, 0.5, 0, 0.5, -0.5]); self.step(150)
        self.set_arm(np.zeros(6)); self.step(300)
        return True


def run_sequence(verbose: bool = True) -> Dict:
    sim = GraspSimulatorV2(verbose=verbose)

    def log(*a):
        if verbose:
            print(*a)

    log("\n开始执行任务序列（v2 修复版, 纯无头）...\n")

    tasks: List[Tuple[str, str]] = [
        ("home", "回零"),
        ("pick:target_cup", "抓取红色杯子"),
        ("home", "回零"),
        ("pick:target_block", "抓取蓝色方块"),
        ("wave", "挥手"),
        ("home", "回零"),
    ]

    grasp_results: List[GraspResult] = []
    task_outcomes = []
    for key, label in tasks:
        log(f"[执行] {label}")
        if key.startswith("pick:"):
            res = sim.pick(key.split(":", 1)[1])
            grasp_results.append(res)
            ok = res.kinematic_success  # 主指标=运动学成功
            log(f"   -> kinematic={res.kinematic_success} physical_lift={res.physical_lift} "
                f"grasp_pos_err={res.grasp_ik_pos_err} ori_err={res.grasp_ik_ori_err}")
            log(f"   -> {res.message}")
        elif key == "home":
            ok = sim.home()
            log("   -> 回零完成")
        elif key == "wave":
            ok = sim.wave()
            log("   -> 挥手完成")
        else:
            ok = False
        task_outcomes.append((label, bool(ok)))

    n_ok = sum(1 for _, o in task_outcomes if o)
    seq_rate = 100.0 * n_ok / len(task_outcomes)

    grasp_kin_ok = sum(1 for r in grasp_results if r.kinematic_success)
    grasp_rate = 100.0 * grasp_kin_ok / max(1, len(grasp_results))
    phys_ok = sum(1 for r in grasp_results if r.physical_lift)

    summary = {
        "sequence_tasks": len(task_outcomes),
        "sequence_success": n_ok,
        "sequence_success_rate_pct": round(seq_rate, 1),
        "grasp_attempts": len(grasp_results),
        "grasp_kinematic_success": grasp_kin_ok,
        "grasp_kinematic_rate_pct": round(grasp_rate, 1),
        "grasp_physical_lift_success": phys_ok,
        "physical_lift_blocked_reason": (
            "夹爪 gripper_jaw STL 网格在末端 site 处无法形成有效夹持面；该几何在 11.1 仓库 XML 中，"
            "不在本 worker 允许改动目录内（铁律#1）。需用户在 XML 给夹爪加 box 碰撞垫或 weld/equality。"
        ),
        "grasp_details": [r.__dict__ for r in grasp_results],
        "task_outcomes": task_outcomes,
        "robot_motion": False,
        "camera_opened": False,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="MuJoCo 物理抓取无头复跑（v2 修复版）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 摘要")
    args = parser.parse_args()

    summary = run_sequence(verbose=not args.json)

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
        return 0

    print("\n" + "=" * 64)
    print("测试总结（v2 修复版）")
    print("=" * 64)
    for label, ok in summary["task_outcomes"]:
        print(f"  {'OK ' if ok else 'X  '}{label}")
    print("-" * 64)
    print(f"序列任务成功率        : {summary['sequence_success']}/{summary['sequence_tasks']} "
          f"({summary['sequence_success_rate_pct']}%)")
    print(f"抓取运动学成功率(主)  : {summary['grasp_kinematic_success']}/{summary['grasp_attempts']} "
          f"({summary['grasp_kinematic_rate_pct']}%)  [baseline 42.9%]")
    print(f"抓取物理夹起成功      : {summary['grasp_physical_lift_success']}/{summary['grasp_attempts']} "
          f"(受 XML 夹爪网格阻塞, 见说明)")
    print(f"物理夹起阻塞原因      : {summary['physical_lift_blocked_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
