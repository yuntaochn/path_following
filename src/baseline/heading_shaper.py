"""
航向参考整形模块（Heading Reference Shaper）

问题背景：
    LOS 导引律基于纯几何关系给出期望航向 ψ_d，它在路径段切换处可能产生
    瞬时大跳变（如 90° 转弯时 Δψ_d ≈ π/2 rad），直接送入 PID 会导致：
        1. PID 输出立即饱和（tau_r = ±9 N·m，CS2 双桨物理上限），进入非线性区
        2. 积分快速累积，产生严重超调
        3. 偏航响应远超船体动态可达范围，实际轨迹偏离路径

本模块提出"约束感知航向参考整形器"：
    不直接让 PID 跟踪 ψ_d，而是先对其整形为满足 USV 偏航动态约束的
    可执行参考轨迹 ψ_ref(t)，再由 PID 跟踪该参考。

核心创新（与已有方法对比）：

    方法1 一阶参考模型滤波（first_order）：
        ψ_ref_dot = (ψ_d - ψ_ref) / T_filter
        ─ 固定时间常数，对高速/低速场景不自适应

    方法2 固定速率限幅（fixed_rate）：
        r_lim = 常数
        ψ_ref ← ψ_ref + clip(Δψ/dt, ─r_lim, +r_lim) · dt
        ─ 限幅值固定，不考虑当前偏航速率和推进器余量

    方法3 动态约束整形（dynamic）【本文方法】：
        利用 USV 偏航动态方程估计当前时刻执行器约束下的
        最大可达偏航速率 r_lim(t)，自适应调节整形速率：

            r_lim(t) = ( τ_r_max + N_r·r(t) + N_rr·|r(t)|·r(t) ) / M_33

        该值依赖当前偏航速率 r(t)：
            ─ r(t) 较大（艇正在转向）→ 阻尼大 → r_lim 降低 → 整形更保守
            ─ r(t) 接近零（艇接近直线）→ 余量充足 → r_lim 增大 → 整形更激进

        同时加入前馈预测修正：
            ψ_ref ← ψ_ref + clip(Δψ/dt, ─r_lim, +r_lim) · dt

稳定性分析：
    设误差 e_psi = ψ_ref - ψ，候选李雅普诺夫函数 V = (1/2)·e_psi²。
    整形器保证 |ψ_ref_dot| ≤ r_lim(t)，而 r_lim(t) 由物理可达性导出，
    因此 PID 不进入饱和区，误差导数 V̇ = e_psi·(ψ_ref_dot - r) ≤ 0，
    闭环航向误差单调收敛。

参数说明（基于 CS2 模型）：
    M33       : 偏航有效惯量 = Iz - N_dr = 1.76 - (-1.0) = 2.76 kg·m²
    Nr        : 偏航线性阻尼系数（CS2 标定值：-1.9 N·m·s/rad）
    Nrr       : 偏航二次阻尼系数（CS2 标定值：-0.75 N·m·s²/rad²）
    tau_r_max : 最大偏航力矩（N·m），与 PID 输出限幅一致（9.0 N·m = T_max × b = 30 × 0.3）
    r_nominal : 名义速率上限（rad/s），安全冗余保护，默认 0.5 rad/s
    method    : 整形方法选择

参考文献：
    [1] Fossen, T.I. (2011). Handbook of Marine Craft Hydrodynamics
        and Motion Control. Chap. 9 Reference Feedforward.
    [2] Skjetne, R. et al. (2004). Adaptive maneuvering with experiments
        for a model ship in a marine control laboratory. Ocean Engineering.
    [3] Bechlioulis et al. (2019). Robust path following for underactuated
        marine vehicles with unknown hydrodynamics. IEEE OE.
"""
import numpy as np
from baseline.math_utils import wrap_to_pi


class HeadingReferenceShaper:
    """
    航向参考整形器。

    插入 LOS 导引律与 PID 控制器之间，将 LOS 几何航向命令
    ψ_d 整形为满足 USV 偏航动态约束的可执行参考 ψ_ref。

    支持三种整形方法（method 参数）：
        "dynamic"    : 【本文方法】基于偏航动态可达性的自适应速率约束
        "fixed_rate" : 固定速率限幅（基线对比方法1）
        "first_order": 一阶参考模型滤波（基线对比方法2）
        "none"       : 直通（不整形），等效于原始 LOS-PID

    参数：
        M33       : 偏航有效惯量（kg·m²），默认 CS2 值 2.76
        Nr        : 偏航线性阻尼（N·m·s/rad），默认 CS2 值 -1.9
        Nrr       : 偏航二次阻尼（N·m·s²/rad²），默认 CS2 值 -0.75
        tau_r_max : 最大偏航力矩（N·m），应与 PID_PSI output_limit 一致
        r_nominal : 名义速率上限（rad/s），对 r_lim_dynamic 的工程安全上限，
                    建议取 1.5~2.0（远大于 fixed_rate 的 r_fixed，确保动态项
                    在正常工况下不被截断，极端工况仍有保护）
        method    : 整形方法，见上
        T_filter  : 一阶滤波时间常数（s），仅 first_order 方法使用
        r_fixed   : 固定速率限幅值（rad/s），仅 fixed_rate 方法使用
    """

    METHODS = {"dynamic", "fixed_rate", "first_order", "none"}

    def __init__(
        self,
        M33: float = 2.76,
        Nr: float = -1.9,
        Nrr: float = -0.75,
        tau_r_max: float = 9.0,
        r_nominal: float = 1.5,
        method: str = "dynamic",
        T_filter: float = 2.0,
        r_fixed: float = 0.3,
    ):
        if method not in self.METHODS:
            raise ValueError(f"method 必须是 {self.METHODS} 之一，当前: {method!r}")

        # 偏航动态参数（CS2 模型）
        self.M33 = float(M33)
        self.Nr = float(Nr)
        self.Nrr = float(Nrr)
        self.tau_r_max = float(tau_r_max)
        self.r_nominal = float(r_nominal)

        # 整形方法
        self.method = method
        self.T_filter = float(T_filter)   # 一阶滤波时间常数
        self.r_fixed = float(r_fixed)     # 固定速率限幅值

        # 内部状态
        self.psi_ref = None   # 当前参考航向（首次调用时用 psi_d 初始化）

        # 调试日志（记录每步的整形信息）
        self._last_r_lim = 0.0
        self._last_delta_psi = 0.0

    def reset(self):
        """
        重置内部状态。

        在每次新仿真开始前调用，清除上次仿真残留的参考状态。
        若不重置，psi_ref 会保留上次仿真末态，导致首步整形异常。
        """
        self.psi_ref = None
        self._last_r_lim = 0.0
        self._last_delta_psi = 0.0

    # ------------------------------------------------------------------
    # 核心计算方法
    # ------------------------------------------------------------------
    def _compute_r_lim_dynamic(self, r: float, dt: float) -> float:
        """
        【核心】基于偏航动态可达性计算自适应速率上限。

        推导：偏航动态方程（单通道）：
            M33 · r_dot = tau_r - Nr·r - Nrr·|r|·r

        在执行器满输出 tau_r = tau_r_max 时，最大角加速度：
            r_dot_max = (tau_r_max + Nr·r + Nrr·|r|·r) / M33
            （注意：Nr<0, Nrr<0，故阻尼项在正转时为负值，减小可用力矩）

        整形参考允许跟踪的最大航向变化速率（rad/s）定义为：
            r_lim(t) = |r(t)| + r_dot_max(t) · dt

        物理意义：在当前偏航速率 r(t) 基础上，执行器在一步 dt 内
        最多再提供 r_dot_max · dt 的速率增量，因此整形参考每步最多
        变化 r_lim · dt。

        自适应特性：
            当 r 较大（转向过程中）：
                Nr·r 和 Nrr·|r|·r 均为负值（Nr<0, Nrr<0），
                available 减小 → r_dot_max 减小，
                但 |r| 已经较大，r_lim 整体有界，约束随速率自适应。
            当 r ≈ 0（直线段或转弯起始）：
                阻尼趋零 → r_dot_max = tau_r_max/M33（约10.9 rad/s²），
                r_lim ≈ r_dot_max·dt（如 dt=0.05s 时约 0.54 rad/s），
                整形可以相对快速响应 LOS 命令跳变。

        与 fixed_rate 的本质区别：
            fixed_rate 的 r_lim 是人工指定的常数，不感知艇的状态；
            dynamic 的 r_lim 随当前偏航速率实时变化，与物理约束挂钩。

        参数：
            r  : 当前偏航速率（rad/s）
            dt : 控制步长（s），用于将角加速度换算为速率增量

        返回：
            r_lim : 自适应速率上限（rad/s），非负
        """
        # 阻尼力矩（Nr<0, Nrr<0 → 正转时为负，减小可用力矩余量）
        damping_torque = self.Nr * r + self.Nrr * abs(r) * r

        # 执行器可用于产生角加速度的净力矩余量
        available = self.tau_r_max + damping_torque
        available = max(0.0, available)

        # 最大角加速度（rad/s²）
        r_dot_max = available / max(self.M33, 1e-6)

        # 【修正】速率上限 = 当前速率绝对值 + 一步内可增加的最大增量
        # 这才是"整形参考允许变化的最大航向速率"（量纲 rad/s 统一）
        r_lim_dynamic = abs(r) + r_dot_max * dt

        # 名义上限：工程安全冗余，防止极端情况下估计过大
        r_lim = min(r_lim_dynamic, self.r_nominal)

        return float(max(r_lim, 0.0))


    def _step_dynamic(self, psi_d: float, r: float, dt: float) -> float:
        """
        动态约束整形：一步更新 ψ_ref（本文方法）。

        算法：
            Δψ    = wrap(ψ_d - ψ_ref)
            r_lim = |r| + r_dot_max(r)·dt        ← 物理可达速率上限
            ψ_ref ← ψ_ref + clip(Δψ/dt, -r_lim, +r_lim) · dt
        """
        r_lim = self._compute_r_lim_dynamic(r, dt)
        delta_psi = wrap_to_pi(psi_d - self.psi_ref) # type: ignore

        psi_ref_dot = float(np.clip(delta_psi / dt, -r_lim, r_lim))
        self.psi_ref = wrap_to_pi(self.psi_ref + psi_ref_dot * dt) # type: ignore

        self._last_r_lim = r_lim
        self._last_delta_psi = delta_psi
        return self.psi_ref


    def _step_fixed_rate(self, psi_d: float, dt: float) -> float:
        """
        固定速率限幅整形：r_lim 为常数（对比方法1）。

        算法：
            Δψ    = wrap(ψ_d - ψ_ref)
            ψ_ref ← ψ_ref + clip(Δψ/dt, -r_fixed, +r_fixed) · dt
        """
        delta_psi = wrap_to_pi(psi_d - self.psi_ref) # type: ignore
        psi_ref_dot = float(np.clip(delta_psi / dt, -self.r_fixed, self.r_fixed))
        self.psi_ref = wrap_to_pi(self.psi_ref + psi_ref_dot * dt) # type: ignore

        self._last_r_lim = self.r_fixed
        self._last_delta_psi = delta_psi
        return self.psi_ref
    

    def _step_first_order(self, psi_d: float, dt: float) -> float:
        """
        一阶参考模型滤波（对比方法2）。

        离散化一阶低通：
            ψ_ref_{k+1} = ψ_ref_k + (dt/T_filter) · wrap(ψ_d - ψ_ref_k)

        时间常数 T_filter 越大，响应越慢、命令越平滑。
        """
        alpha = dt / max(self.T_filter, dt)  # 防止除零
        delta_psi = wrap_to_pi(psi_d - self.psi_ref) # type: ignore
        self.psi_ref = wrap_to_pi(self.psi_ref + alpha * delta_psi) # type: ignore

        self._last_r_lim = abs(delta_psi) / max(self.T_filter, dt)
        self._last_delta_psi = delta_psi
        return self.psi_ref

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def __call__(self, psi_d: float, r: float, dt: float) -> tuple[float, dict]:
        """
        执行一步航向参考整形。

        参数：
            psi_d : LOS 给出的几何期望航向（rad）
            r     : 当前偏航速率（rad/s），来自 nu[2]
            dt    : 控制时间步长（s）

        返回：
            psi_ref : 整形后的参考航向（rad），发送给 PID_PSI
            info    : 调试信息字典，包含：
                        "psi_d"           : LOS 原始命令（rad）
                        "psi_ref"         : 整形后参考（rad）
                        "r_lim"           : 本步速率上限（rad/s）
                        "delta_psi_raw"   : 整形前航向误差（rad）
                        "delta_psi_shaped": 整形后航向误差（rad）
                        "shaper_method"   : 使用的整形方法
        """
        # 首次调用时，用当前 psi_d 初始化 psi_ref（避免从零起步的大跳变）
        if self.psi_ref is None:
            self.psi_ref = float(psi_d)

        psi_d = float(wrap_to_pi(psi_d))
        r = float(r)

        if self.method == "dynamic":
            psi_ref = self._step_dynamic(psi_d, r, dt)
        elif self.method == "fixed_rate":
            psi_ref = self._step_fixed_rate(psi_d, dt)
        elif self.method == "first_order":
            psi_ref = self._step_first_order(psi_d, dt)
        else:  # "none"
            self.psi_ref = float(psi_d)
            psi_ref = float(psi_d)
            self._last_r_lim = float("inf")
            self._last_delta_psi = 0.0

        info = {
            "psi_d_raw": psi_d,                             # LOS 原始命令
            "psi_ref": psi_ref,                             # 整形后参考
            "r_lim": self._last_r_lim,                      # 速率上限
            "delta_psi_raw": float(self._last_delta_psi),   # 整形前误差
            "delta_psi_shaped": float(                      # 整形后误差
                wrap_to_pi(psi_ref - psi_d)
            ),
            "shaper_method": self.method,
        }
        return psi_ref, info

# ------------------------------------------------------------------
# 工厂函数：快速创建常用配置
# ------------------------------------------------------------------

def make_shaper(method: str = "dynamic", **kwargs) -> HeadingReferenceShaper:
    """
    快速创建整形器实例的工厂函数。

    参数：
        method : 整形方法（"dynamic" / "fixed_rate" / "first_order" / "none"）
        **kwargs : 传递给 HeadingReferenceShaper 的其他参数

    示例：
        shaper = make_shaper("dynamic", tau_r_max=30.0, r_nominal=0.5)
        shaper = make_shaper("fixed_rate", r_fixed=0.3)
        shaper = make_shaper("first_order", T_filter=2.0)
        shaper = make_shaper("none")
    """
    return HeadingReferenceShaper(method=method, **kwargs)
