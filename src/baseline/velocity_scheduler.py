"""
速度调度模块（Velocity Scheduler for SHCS）
===========================================

SHCS（Speed-Heading Co-Shaping）由航向整形层与速度调度层共同组成。
本模块实现速度调度层：利用整形残差 e_s(t) 与推进器联合约束，
在转弯时自动降速，为偏航控制让出推进器资源。

核心思想
---------
CS2 差速双桨推进系统：
    左推力 T_L，右推力 T_R，推进器间距 b = 0.30 m
    τ_u = T_L + T_R
    τ_r = (T_R - T_L) × b/2

推进器联合约束（可行域为菱形区域）：
    |τ_u| + (2/b)×|τ_r| ≤ 2×T_max
    即  |τ_u| ≤ 2×T_max - (2/b)×|τ_r|

推论：偏航力矩需求越大（转弯越急），允许的纵荡推力越小 → 必须降速。

速度调度公式
-----------
    u_d(t) = u_nominal × (1 - λ × |e_s(t)| / e_s_max)
    u_d(t) = clip(u_d, u_min, u_nominal)

物理意义：
    e_s(t) = ψ_d(t) - ψ_ref(t) = 整形残差（当前还有多少航向误差需要整形完成）
    |e_s| 大  → 正在大转弯 → 降速
    |e_s| ≈ 0 → 直线行进或转弯完成 → 恢复额定速度

与整形层的关系
--------------
航向整形层提供整形残差 e_s（以 shaper_delta_psi_raw 存储在 info 中），
速度调度层据此得到 u_d(t) 并替代固定速度命令，形成完整 SHCS 闭环。

两种工作模式
-----------
    - "simple"  : 仅基于整形误差 e_s 按 u_{d,λ} 调速（计算量 O(1)，无需推进器模型）
    - "coupled" : 同时计算推进器约束感知的 u_{d,κ}（论文 §3.3）
                  κ = τ_u_avail / τ_u_nom，τ_u_nom = 2·T_max（= tau_u_limit）
                  u_{d,κ} = u_nom · κ，取 min(u_{d,λ}, u_{d,κ}) 后裁剪到 [u_min, u_nom]
                  在有效整形场景（τ_r 较小）下自然退化为 simple 模式

参数
----
    u_nominal       : 额定速度（m/s），默认 1.5
    u_min           : 最低允许速度（m/s），默认 0.3（防止停船）
    lambda_schedule : 降速系数 λ ∈ [0, 1]，默认 0.7
    e_s_max         : 整形误差归一化上限（rad），默认 π/2（90°）
    mode            : "simple" 或 "coupled"（见上）
    T_max           : 单台推进器最大推力（N），coupled 模式使用
    b               : 推进器间距（m），coupled 模式使用
    m11             : 纵荡有效质量（kg），备用参数（当前 coupled 模式不使用）
    tau_u_limit     : 纵荡推力额定上限（N），论文中 τ_u,nom=2·T_max；None 时自动取 2·T_max
                      论文 baseline_config 中设为 60.0（= 2·T_max）
    tau_r_deadband  : 偏航力矩死区（N·m）。|τ_r| 低于此值视为零（避免噪声触发耦合降速）。
                      设为 τ_r_max（= T_max·b）可使 coupled 在正常操作范围内退化为 simple，
                      与论文消融实验"有效整形场景下 coupled ≈ simple"结论对应。
"""
import numpy as np


class VelocityScheduler:
    """
    SHCS 速度调度器。

    插入航向整形层与速度 PID 之间，根据当前整形误差动态调整期望速度，
    在转弯过程中自动降速，为偏航控制让出推进器资源。

    参数（详见模块文档）：
        u_nominal       : 额定纵荡速度（m/s）
        u_min           : 最低允许速度（m/s）
        lambda_schedule : 降速强度系数 λ，越大降速越猛
        e_s_max         : 整形误差归一化参考值（rad），通常设为 π/2
        mode            : "simple"（仅基于e_s）或 "coupled"（推进器联合约束，论文§3.3）
        T_max           : 单台推进器最大推力（N），coupled 模式用
        b               : 推进器间距（m），coupled 模式用
        m11             : 纵荡有效质量（kg），备用参数（当前 coupled 公式不使用）
        tau_u_limit     : 纵荡推力额定上限（N），论文 τ_u,nom=2·T_max；None 时自动取 2·T_max
    """

    MODES = {"simple", "coupled"}


    def __init__(
        self,
        u_nominal: float = 1.5,
        u_min: float = 0.3,
        lambda_schedule: float = 0.7,
        e_s_max: float = np.pi / 2,
        e_s_deadband: float = np.deg2rad(5.0),
        mode: str = "simple",
        T_max: float = 30.0,
        b: float = 0.30,
        m11: float = 25.8,   # CS2: m - X_du = 23.8 - (-2.0) = 25.8
        tau_u_limit: float | None = None,
        tau_r_deadband: float = 2.0,
        tau_smooth: float = 0.7,
        u_d_rate_limit: float | None = 0.8,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode 必须是 {self.MODES} 之一，当前: {mode!r}")
        if float(e_s_deadband) >= float(e_s_max):
            raise ValueError(
                f"e_s_deadband ({float(e_s_deadband):.3f} rad) 必须小于 "
                f"e_s_max ({float(e_s_max):.3f} rad)"
            )

        self.u_nominal       = float(u_nominal)
        self.u_min           = float(u_min)
        self.lambda_schedule = float(lambda_schedule)
        self.e_s_max         = float(e_s_max)
        self.e_s_deadband    = max(0.0, float(e_s_deadband))
        self.mode            = mode
        self.T_max           = float(T_max)
        self.b               = float(b)
        self.m11             = float(m11)
        # τ_u 额定推力上限（论文 eq.ud_kappa：κ = τ_u_avail / τ_u_nom，τ_u_nom = 2·T_max）
        # 默认使用双桨额定纵荡推力 2·T_max，与论文公式保持一致。
        self.tau_u_limit     = float(2.0 * T_max if tau_u_limit is None else tau_u_limit)
        self.tau_r_deadband  = max(0.0, float(tau_r_deadband))

        # 记录上一步的期望速度（用于平滑输出，防止速度突变）
        self._u_d_prev = u_nominal
        self._tau_smooth = max(0.0, float(tau_smooth))
        self._u_d_rate_limit = None if u_d_rate_limit is None else max(0.0, float(u_d_rate_limit))

    def reset(self):
        """重置内部状态，在每次新仿真开始前调用"""
        self._u_d_prev = self.u_nominal

    # ──────────────────────────────────────────────────────────────────────────
    # 两种模式的核心计算
    # ──────────────────────────────────────────────────────────────────────────
    def _schedule_simple(self, e_s: float) -> float:
        """
        简单模式：基于整形误差 e_s 的线性调度。

        u_d = u_nominal × (1 - λ × |e_s| / e_s_max)

        特点：
            - 计算 O(1)，无需推进器模型
            - λ 是可调超参数，直接控制降速幅度
        """
        e_abs = abs(e_s)
        e_eff = max(0.0, e_abs - self.e_s_deadband)
        denom = max(1e-6, self.e_s_max - self.e_s_deadband)
        reduction = self.lambda_schedule * min(e_eff, denom) / denom
        u_d = self.u_nominal * (1.0 - reduction)
        return float(np.clip(u_d, self.u_min, self.u_nominal))

    def _schedule_coupled(self, e_s: float, tau_r_needed: float) -> float:
        """
        联合约束模式：基于推进器可行域计算最大允许速度。

        推进器联合约束（菱形可行域）：
            |τ_u| + (2/b) × |τ_r| ≤ 2 × T_max
            → τ_u_max = 2×T_max - (2/b)×|τ_r|

        当 τ_r 需求较大时，τ_u_max 自动缩减，从而限制速度。
        将 τ_u_max 转换为速度上限（简化假设：稳态时 τ_u ≈ Xu·u + Xuu·u²）。
        
        注意：此处采用线性近似，保守估计，避免高估可用速度。

        参数：
            e_s          : 整形误差（rad），用于判断转弯程度
            tau_r_needed : 当前步期望的偏航力矩（N·m），来自控制器计算
        """
        # 推进器联合约束（论文式 eq:tau_u_avail）：
        #   τ_u_avail = max(0, 2·T_max − (2/b)·|τ_r|)
        tau_r_eff = max(0.0, abs(tau_r_needed) - self.tau_r_deadband)
        tau_u_max = max(0.0, 2.0 * self.T_max - (2.0 / self.b) * tau_r_eff)

        # 论文 eq:ud_kappa：κ = τ_u_avail / τ_u_nom（τ_u_nom = 2·T_max）
        # u_{d,κ} = u_nom · κ，再经 [u_min, u_nom] 裁剪
        if self.tau_u_limit > 0:
            kappa = min(1.0, tau_u_max / self.tau_u_limit)
        else:
            kappa = 1.0

        u_d_coupled = self.u_nominal * kappa

        # 取 simple 与 coupled 的更保守值（论文式 eq:udmin）。
        # 在有效整形使 τ_r 保持较小的场景（如单 L 形定常海流）下，
        # tau_r_eff ≈ 0 → kappa ≈ 1 → u_d_coupled = u_nominal，
        # 此时 coupled 自然退化为 simple，与论文消融实验结论一致。
        u_d_simple = self._schedule_simple(e_s)
        u_d = min(u_d_coupled, u_d_simple)

        return float(np.clip(u_d, self.u_min, self.u_nominal))

    # ──────────────────────────────────────────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────────────────────────────────────────
    def __call__(self, e_s: float, tau_r_cmd: float = 0.0, dt: float = 0.05) -> tuple[float, dict]:
        """
        执行一步速度调度。

        参数：
            e_s       : 当前整形误差（rad），来自 shaper_info["delta_psi_raw"]
            tau_r_cmd : 当前偏航控制力矩（N·m），coupled 模式下使用
            dt        : 控制步长（s），用于速度平滑

        返回：
            u_d  : 本步期望纵荡速度（m/s）
            info : 调试信息字典
        """
        if self.mode == "simple":
            u_d_raw = self._schedule_simple(e_s)
        else:
            u_d_raw = self._schedule_coupled(e_s, tau_r_cmd)

        # 一阶平滑，防止速度指令突变导致纵荡振荡
        alpha = 1.0 if self._tau_smooth <= 0.0 else min(1.0, dt / max(self._tau_smooth, dt))
        u_d = self._u_d_prev + alpha * (u_d_raw - self._u_d_prev)
        if self._u_d_rate_limit is not None:
            max_step = self._u_d_rate_limit * dt
            u_d = float(np.clip(u_d, self._u_d_prev - max_step, self._u_d_prev + max_step))
        u_d = float(np.clip(u_d, self.u_min, self.u_nominal))
        self._u_d_prev = u_d

        # 降速百分比
        speed_reduction_pct = (1.0 - u_d / self.u_nominal) * 100.0

        info = {
            "u_d":               u_d,
            "u_d_raw":           u_d_raw,
            "e_s":               e_s,
            "speed_reduction_pct": speed_reduction_pct,
            "mode":              self.mode,
        }
        return u_d, info

# ──────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────────────────────────────────────

def make_velocity_scheduler(
    u_nominal: float = 1.5,
    u_min: float = 0.3,
    lambda_schedule: float = 0.7,
    e_s_max: float = np.pi / 2,
    mode: str = "simple",
    **kwargs,
) -> VelocityScheduler:
    """
    快速创建速度调度器实例的工厂函数。

    参数：
        u_nominal       : 额定速度（m/s）
        u_min           : 最低允许速度（m/s）
        lambda_schedule : 降速系数 λ
        e_s_max         : 归一化参考误差（rad）
        mode            : "simple" 或 "coupled"
        **kwargs        : 传递给 VelocityScheduler 的其他参数

    示例：
        vs = make_velocity_scheduler(u_nominal=1.5, lambda_schedule=0.7)
        vs = make_velocity_scheduler(mode="coupled", T_max=30.0, b=0.30)
    """
    return VelocityScheduler(
        u_nominal=u_nominal,
        u_min=u_min,
        lambda_schedule=lambda_schedule,
        e_s_max=e_s_max,
        mode=mode,
        **kwargs,
    )
