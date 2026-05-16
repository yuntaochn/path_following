"""
USV LOS-PID 控制器模块

将 LOS 导引律（生成期望航向）和 PID 控制器（跟踪期望值）
组合为完整的 USV 路径跟踪控制器。

控制架构（分层）：

    无整形层（原始）：
        路径点 → LOS 导引律 → ψ_d → PID_PSI → τ_r

    有整形层（本文方法）：
        路径点 → LOS 导引律 → ψ_d → [HeadingReferenceShaper] → ψ_ref → PID_PSI → τ_r

    两种配置均支持，通过 shaper 参数控制（None 表示无整形层）。

状态定义：
    η = [x, y, ψ]   位置坐标（m）和航向角（rad）
    ν = [u, v, r]   纵荡速度（m/s）、横荡速度（m/s）、偏航角速度（rad/s）
    τ = [τ_u, 0, τ_r]  控制力向量（N 和 N·m）

说明：
    sway（横荡）方向力 τ_v = 0，适合无横向推进器的 USV（baseline 配置）
"""
import numpy as np
from typing import Union
from baseline.pid import PID
from baseline.los import LOSGuidance, AdaptiveLOSGuidance, ILOSGuidance

class USVLOSController:
    """
    USV LOS-PID 路径跟踪控制器（支持可选的航向参考整形层和速度调度层）。

    控制逻辑（无整形层）：
        1. LOS 导引律根据当前位置和路径点计算几何期望航向 ψ_d
        2. PID_PSI 直接跟踪 ψ_d，输出偏航力矩 τ_r
        3. PID_U 跟踪期望速度 u_d，输出纵荡推力 τ_u

    控制逻辑（有整形层，shaper 不为 None）：
        1. LOS 导引律计算几何期望航向 ψ_d
        2. HeadingReferenceShaper 将 ψ_d 整形为满足偏航动态约束的 ψ_ref
        3. PID_PSI 跟踪 ψ_ref（而非直接跟踪 ψ_d），输出偏航力矩 τ_r
        4. PID_U 跟踪期望速度 u_d，输出纵荡推力 τ_u

    控制逻辑（SHCS：整形层 + 速度调度层，shaper 和 velocity_scheduler 均不为 None）：
        1. LOS 导引律计算几何期望航向 ψ_d
        2. HeadingReferenceShaper 将 ψ_d 整形为 ψ_ref，同时得到整形误差 e_s
        3. VelocityScheduler 根据 e_s 动态调整期望速度 u_d(t)
        4. PID_PSI 跟踪 ψ_ref，输出偏航力矩 τ_r
        5. PID_U 跟踪动态调整后的 u_d(t)，输出纵荡推力 τ_u

    参数:
        pid_u              : 速度 PID 控制器实例（PID 类），控制纵荡速度 u
        pid_psi            : 航向 PID 控制器实例（PID 类），控制航向角 ψ
        los                : LOS 导引律实例（LOSGuidance / AdaptiveLOSGuidance / ILOSGuidance）
        u_d                : 基准期望纵荡速度（m/s），默认 1.0 m/s
        shaper             : 航向参考整形器实例（HeadingReferenceShaper），None 表示不使用整形层
        velocity_scheduler : 速度调度器实例（VelocityScheduler），None 表示固定速度；
                             仅在 shaper 不为 None 时有效（需要整形误差 e_s）
    """
    def __init__(
        self,
        pid_u: PID,
        pid_psi: PID,
        los: Union[LOSGuidance, AdaptiveLOSGuidance, ILOSGuidance],
        u_d: float = 1.0,
        shaper=None,
        velocity_scheduler=None,
        tau_r_filter_tau: float = 0.15,
        tau_r_rate_limit: float | None = 120.0,
    ):
        self.pid_u = pid_u      # 速度 PID（输出纵荡推力 τ_u）
        self.pid_psi = pid_psi  # 航向 PID（输出偏航力矩 τ_r）
        self.los = los          # LOS 导引律（输出几何期望航向 ψ_d）
        self.u_d = u_d          # 基准期望纵荡速度（m/s）
        self.shaper = shaper    # 航向参考整形器（可选）
        self.velocity_scheduler = velocity_scheduler  # 速度调度器（可选，SHCS）
        self.tau_r_filter_tau = max(0.0, float(tau_r_filter_tau))
        self.tau_r_rate_limit = None if tau_r_rate_limit is None else max(0.0, float(tau_r_rate_limit))
        self._tau_r_cmd_prev = 0.0

    def reset(self):
        """重置控制器所有内部状态（PID 积分、LOS 航点索引、整形器状态、速度调度器状态）"""
        self.pid_u.reset()
        self.pid_psi.reset()
        self.los.reset()
        if self.shaper is not None:
            self.shaper.reset()
        if self.velocity_scheduler is not None:
            self.velocity_scheduler.reset()
        self._tau_r_cmd_prev = 0.0

    def _smooth_tau_r(self, tau_r_raw: float, dt: float) -> float:
        """
        对偏航力矩指令做一阶平滑 + 斜率限制，减少高频抖动与尖峰。
        """
        tau_prev = float(self._tau_r_cmd_prev)
        tau_cmd = float(tau_r_raw)

        if self.tau_r_filter_tau > 0.0:
            alpha = min(1.0, dt / max(self.tau_r_filter_tau, dt))
            tau_cmd = tau_prev + alpha * (tau_cmd - tau_prev)

        if self.tau_r_rate_limit is not None:
            max_step = self.tau_r_rate_limit * dt
            tau_cmd = float(np.clip(tau_cmd, tau_prev - max_step, tau_prev + max_step))

        if self.pid_psi.output_limit is not None:
            lo, hi = self.pid_psi.output_limit
            tau_cmd = float(np.clip(tau_cmd, lo, hi))

        self._tau_r_cmd_prev = tau_cmd
        return tau_cmd

    def __call__(self, eta, nu, waypoints, dt):
        """
        执行一步控制计算。

        参数:
            eta      : 当前状态 [x, y, ψ]
            nu       : 当前速度 [u, v, r]
            waypoints: 路径点列表，格式 [[x1,y1], [x2,y2], ...]
            dt       : 控制周期（秒）

        返回:
            tau  : 控制力向量 np.array([τ_u, 0.0, τ_r])
            info : 调试信息字典，包含：
                   - psi_d             : LOS 几何期望航向（rad）
                   - psi_ref           : 整形后参考航向（无整形时等于 psi_d）（rad）
                   - cross_track_error : 横向误差（m）
                   - along_track_error : 纵向误差（m）
                   - path_angle        : 路径段方位角（rad）
                   - lookahead         : 前视距离（m）
                   - curvature         : 路径曲率（1/m）
                   - wp_idx            : 当前路径段索引
                   - u_d               : 期望速度（m/s）
                   - u                 : 当前纵荡速度（m/s）
                   - psi               : 当前航向角（rad）
                   - tau_u, tau_r      : 控制输出值
                   - shaper_r_lim      : 整形器速率上限（rad/s，无整形时为 inf）
                   - shaper_method     : 整形方法名称
        """
        x, y, psi = eta
        u, v, r = nu

        waypoints = np.asarray(waypoints, dtype=float)

        # Step 1: LOS 导引律计算几何期望航向 ψ_d
        psi_d, los_info = self.los(
            position=[x, y],
            waypoints=waypoints,
            eta=eta,
            nu=nu,
            dt=dt,
        )

        # Step 2: 航向参考整形（可选）
        # 将几何命令 ψ_d 整形为满足偏航动态约束的可执行参考 ψ_ref
        if self.shaper is not None:
            psi_ref, shaper_info = self.shaper(psi_d=psi_d, r=r, dt=dt)
        else:
            # 无整形层：直通
            psi_ref = psi_d
            shaper_info = {
                "psi_d_raw": psi_d,
                "psi_ref": psi_d,
                "r_lim": float("inf"),
                "delta_psi_raw": 0.0,
                "delta_psi_shaped": 0.0,
                "shaper_method": "none",
            }

        # Step 3: 速度调度（SHCS，可选）
        # 根据整形误差 e_s 动态调整期望速度，为偏航控制让出推进器资源
        tau_r_preview = 0.0
        scheduler_mode = "none"
        if self.shaper is not None and self.velocity_scheduler is not None:
            e_s = shaper_info.get("delta_psi_raw", 0.0)
            scheduler_mode = getattr(self.velocity_scheduler, "mode", "unknown")
            # coupled 模式需要先估计本步偏航力矩需求，再据此进行速度调度。
            # 使用 PID.preview 只做预测，不改动真实 PID 的积分/微分内部状态。
            if scheduler_mode == "coupled":
                tau_r_preview = self.pid_psi.preview(
                    setpoint=psi_ref,
                    measurement=psi,
                    dt=dt,
                )
            u_d_scheduled, vs_info = self.velocity_scheduler(
                e_s=e_s,
                tau_r_cmd=tau_r_preview,
                dt=dt,
            )
        else:
            u_d_scheduled = self.u_d
            vs_info = {"u_d": self.u_d, "speed_reduction_pct": 0.0}

        # Step 4: 速度 PID 控制（纵荡推力）
        tau_u = self.pid_u(setpoint=u_d_scheduled, measurement=u, dt=dt)

        # Step 5: 航向 PID 控制（偏航力矩）
        # 跟踪整形后的参考 ψ_ref（而非直接跟踪 ψ_d）
        tau_r_raw = self.pid_psi(setpoint=psi_ref, measurement=psi, dt=dt)
        tau_r = self._smooth_tau_r(tau_r_raw, dt)

        # 组合控制力向量：[τ_u, 0, τ_r]
        tau = np.array([tau_u, 0.0, tau_r], dtype=float)

        # 整合调试信息
        info = {
            **los_info,               # LOS 信息（psi_d, e_ct, s_along, 等）
            "psi_ref": psi_ref,       # 整形后参考航向（关键新增字段）
            "u_d": u_d_scheduled,     # 动态调整后的期望速度（SHCS 时随转弯变化）
            "u_d_nominal": self.u_d,  # 基准额定速度（固定值）
            "u": u,
            "psi": psi,
            "tau_u": tau_u,
            "tau_r": tau_r,
            "tau_r_raw": tau_r_raw,
            # 整形器调试信息
            "shaper_r_lim": shaper_info.get("r_lim", float("inf")),
            "shaper_method": shaper_info.get("shaper_method", "none"),
            "shaper_delta_psi_raw": shaper_info.get("delta_psi_raw", 0.0),
            # 速度调度信息（SHCS 新增）
            "tau_r_preview": tau_r_preview,
            "speed_scheduler_mode": scheduler_mode,
            "speed_reduction_pct": vs_info.get("speed_reduction_pct", 0.0),
        }

        return tau, info
