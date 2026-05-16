"""
双桨推进器分配与饱和处理模块

仿真器与船舶动力学模型使用广义力 τ = [τ_u, τ_v, τ_r] 描述控制输入，
而 CS2 的物理推进系统为差速双桨布局。本模块负责：
    1. 将纵荡/偏航指令 (τ_u, τ_r) 转换为左/右推进器推力 (T_L, T_R)；
    2. 对单个推进器施加饱和限幅（|T_i| ≤ T_max）；
    3. 将饱和后的可实现推力映射回广义力，以便动力学模型使用。

推力分配公式（论文式 3）：
    τ_u = T_L + T_R
    τ_r = (b/2) · (T_R - T_L)

其中 b = 0.30 m 为两推进器横向间距，T_max = 30 N 为单桨最大推力。
联合约束（论文式 4）：|τ_u| + (2/b)·|τ_r| ≤ 2·T_max
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, Union

class TwinThrusterAllocator:
    """
    差速双桨推进器分配器。

    CS2 的推进系统由对称安装的左、右两个推进器组成，
    分配关系（论文式 3）：
        τ_u = T_L + T_R          纵荡推力 = 两桨合力
        τ_r = (b/2)·(T_R - T_L) 偏航力矩 = 两桨差力 × 力臂

    每个推进器受单桨饱和约束 |T_i| ≤ T_max，
    二者联合产生的联合约束为 |τ_u| + (2/b)·|τ_r| ≤ 2·T_max（论文式 4）。

    参数:
        T_max : 单桨最大推力（N），默认 30 N
        b     : 推进器横向间距（m），默认 0.30 m
    """

    def __init__(self, T_max: float = 30.0, b: float = 0.30):
        self.T_max = float(T_max)
        self.b = float(b)

    def tau_to_thrusters(self, tau_u: float, tau_r: float) -> np.ndarray:
        """
        将广义力 (τ_u, τ_r) 转换为左/右推进器推力。

        逆映射公式：
            T_L = τ_u/2 - τ_r/b
            T_R = τ_u/2 + τ_r/b

        注意：此步骤不施加饱和限幅，可能输出超过 T_max 的值，
        需后续由 allocate() 截断。

        参数:
            tau_u : 纵荡推力指令（N）
            tau_r : 偏航力矩指令（N·m）

        返回:
            np.ndarray [T_L, T_R]，形状 (2,)
        """
        tau_u = float(tau_u)
        tau_r = float(tau_r)
        t_left  = 0.5 * tau_u - tau_r / max(self.b, 1e-9)
        t_right = 0.5 * tau_u + tau_r / max(self.b, 1e-9)
        return np.array([t_left, t_right], dtype=float)

    def thrusters_to_tau(self, t_left: float, t_right: float, tau_v: float = 0.0) -> np.ndarray:
        """
        将左/右推进器推力映射回广义力向量 [τ_u, τ_v, τ_r]。

        正向映射公式（论文式 3）：
            τ_u = T_L + T_R
            τ_r = (b/2)·(T_R - T_L)
            τ_v = tau_v（CS2 欠驱动，无横荡力，通常为 0）

        参数:
            t_left  : 左推进器推力（N）
            t_right : 右推进器推力（N）
            tau_v   : 横荡力（N），CS2 为欠驱动艇，该值应为 0

        返回:
            np.ndarray [τ_u, τ_v, τ_r]，形状 (3,)
        """
        t_left  = float(t_left)
        t_right = float(t_right)
        tau_v   = float(tau_v)
        tau_u = t_left + t_right
        tau_r = 0.5 * self.b * (t_right - t_left)
        return np.array([tau_u, tau_v, tau_r], dtype=float)

    def feasible_tau_u(self, tau_r: float) -> float:
        """
        计算在指定偏航需求下，联合约束允许的最大纵荡推力。

        由联合约束（论文式 4）推导：
            τ_{u,avail} = max(0, 2·T_max - (2/b)·|τ_r|)

        用于速度调度层的保守速度上限计算（论文 3.3 节）。

        参数:
            tau_r : 偏航力矩需求（N·m）

        返回:
            可用最大纵荡推力（N）
        """
        tau_r = abs(float(tau_r))
        return max(0.0, 2.0 * self.T_max - (2.0 / max(self.b, 1e-9)) * tau_r)

    def allocate(self, tau_cmd: Union[Sequence[float], np.ndarray]) -> dict:
        """
        执行完整推力分配流程（广义力 → 推进器推力 → 饱和截断 → 回映广义力）。

        流程：
            1. 将指令广义力 [τ_u, τ_r] 逆映射为 [T_L, T_R]（可能超过 T_max）
            2. 对每个推进器施加单桨限幅：T_i = clip(T_i, -T_max, T_max)
            3. 将截断后的 [T_L, T_R] 正向映射回可实现广义力

        步骤 1→2 的差值即为"分配截断误差"，反映指令力超出物理能力的程度。

        参数:
            tau_cmd : 控制器原始指令 [τ_u, τ_v, τ_r]，形状 (3,)

        返回:
            字典，包含以下键：
                tau_cmd_raw      : 原始指令广义力 [τ_u, τ_v, τ_r]
                tau_cmd_alloc    : 分配截断后可实现的广义力
                thruster_raw     : 分配前推进器推力 [T_L, T_R]（可能超限）
                thruster_alloc   : 饱和截断后推进器推力 [T_L, T_R]
                thruster_error   : 截断误差 = thruster_raw - thruster_alloc
                saturated        : bool，任一推进器是否饱和
                sat_left         : bool，左推进器是否饱和
                sat_right        : bool，右推进器是否饱和
                utilization_left : 左推进器利用率（0~1）
                utilization_right: 右推进器利用率
                utilization_max  : 最大推进器利用率（饱和时为 1.0）
        """
        tau_cmd = np.asarray(tau_cmd, dtype=float).reshape(3,)
        tau_u_raw, tau_v_raw, tau_r_raw = tau_cmd

        # 步骤 1：逆映射（可能超限）
        thruster_raw = self.tau_to_thrusters(tau_u_raw, tau_r_raw)
        # 步骤 2：单桨饱和限幅
        thruster_sat = np.clip(thruster_raw, -self.T_max, self.T_max)
        # 步骤 3：正向映射回广义力（含原始 τ_v）
        tau_alloc = self.thrusters_to_tau(
            thruster_sat[0],
            thruster_sat[1],
            tau_v=tau_v_raw,
        )

        thruster_error  = thruster_raw - thruster_sat
        saturated_mask  = np.abs(thruster_error) > 1e-9

        return {
            "tau_cmd_raw":       tau_cmd,
            "tau_cmd_alloc":     tau_alloc,
            "thruster_raw":      thruster_raw,
            "thruster_alloc":    thruster_sat,
            "thruster_error":    thruster_error,
            "saturated":         bool(np.any(saturated_mask)),
            "sat_left":          bool(saturated_mask[0]),
            "sat_right":         bool(saturated_mask[1]),
            "utilization_left":  abs(float(thruster_sat[0])) / max(self.T_max, 1e-9),
            "utilization_right": abs(float(thruster_sat[1])) / max(self.T_max, 1e-9),
            "utilization_max":   max(abs(float(thruster_sat[0])), abs(float(thruster_sat[1]))) / max(self.T_max, 1e-9),
        }
