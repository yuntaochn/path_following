"""
性能指标计算模块

提供 USV 路径跟踪性能评估的各类指标计算函数，
包括误差统计指标、积分指标、超调量、收敛时间和控制能量等。

主要指标说明：
    RMS（Root Mean Square）     : 均方根误差，综合反映跟踪精度
    MAE（Mean Absolute Error）  : 平均绝对误差
    IAE（Integral Absolute Error）: 绝对误差积分，考虑时间累积效应
    ISE（Integral Squared Error）: 平方误差积分，对大误差惩罚更重
    ITAE（Integral Time-weighted Absolute Error）: 时间加权绝对误差积分，
         对后期误差惩罚更重（鼓励快速收敛）
    控制能量 : ∑τ²·Δt，反映控制器的"代价"（激进控制能量大）
    收敛时间 : 误差进入目标稳定带后首次保持不超出的时刻
"""

import numpy as np
from baseline.config import make_s_curve_path

def rms(x):
    """
    计算均方根值（Root Mean Square）。
    
    RMS = sqrt(mean(x²))
    
    参数:
        x : 数据数组
    
    返回:
        均方根值（float），空数组返回 nan
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(x ** 2)))

def max_abs(x):
    """
    计算最大绝对值（峰值误差）。
    
    参数:
        x : 数据数组
    
    返回:
        最大绝对值（float），空数组返回 nan
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return float(np.max(np.abs(x)))

def mean_abs(x):
    """
    计算平均绝对误差（Mean Absolute Error, MAE）。
    
    MAE = mean(|x|)
    
    参数:
        x : 数据数组
    
    返回:
        平均绝对值（float），空数组返回 nan
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return float(np.mean(np.abs(x)))

def final_value(x):
    """
    获取数组最后一个元素（终端值）。
    
    用于获取仿真结束时的状态值，如最终横向误差。
    
    参数:
        x : 数据数组
    
    返回:
        最后一个元素（float），空数组返回 nan
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    return float(x[-1])

def control_energy(u, dt):
    """
    计算控制能量 E = ∑(u² · dt)。
    
    反映控制器的代价，激进的控制策略通常有更大的能量消耗。
    物理上近似于力对时间的做功量（功率积分）。
    
    参数:
        u  : 控制输入序列
        dt : 时间步长（s）
    
    返回:
        控制能量（float），空数组返回 nan
    """
    u = np.asarray(u, dtype=float)
    if u.size == 0:
        return np.nan
    return float(np.sum(u ** 2) * dt)

def itae(error, t, use_abs=True):
    """
    计算时间加权绝对误差积分（ITAE，Integral Time-weighted Absolute Error）。
    
    ITAE = ∫ t · |e(t)| dt ≈ ∑ t_k · |e_k| · Δt
    
    特点：对后期（时间较大时）的误差惩罚更重，
    鼓励控制系统快速收敛到稳态。
    
    参数:
        error   : 误差序列
        t       : 时间序列（s）
        use_abs : True 时计算 ITAE（|e|），False 时计算时间加权误差积分
    
    返回:
        ITAE 值（float），空数组返回 nan
    """
    error = np.asarray(error, dtype=float)
    t = np.asarray(t, dtype=float)

    if error.size == 0 or t.size == 0 or error.size != t.size:
        return np.nan

    y = np.abs(error) if use_abs else error
    # 使用梯形积分法（np.trapezoid）计算数值积分
    return float(np.trapezoid(t * y, t))

def ise(error, t):
    """
    计算平方误差积分（ISE，Integral Squared Error）。
    
    ISE = ∫ e²(t) dt
    
    特点：对大误差惩罚更重（平方放大效果），
    比 IAE 对瞬时大误差更敏感。
    
    参数:
        error : 误差序列
        t     : 时间序列（s）
    
    返回:
        ISE 值（float），空数组返回 nan
    """
    error = np.asarray(error, dtype=float)
    t = np.asarray(t, dtype=float)

    if error.size == 0 or t.size == 0 or error.size != t.size:
        return np.nan
    return float(np.trapezoid(error ** 2, t))

def iae(error, t):
    """
    计算绝对误差积分（IAE，Integral Absolute Error）。
    
    IAE = ∫ |e(t)| dt
    
    特点：比 RMS 更直观，反映误差的总累积量，
    对正负误差一视同仁。
    
    参数:
        error : 误差序列
        t     : 时间序列（s）
    
    返回:
        IAE 值（float），空数组返回 nan
    """
    error = np.asarray(error, dtype=float)
    t = np.asarray(t, dtype=float)

    if error.size == 0 or t.size == 0 or error.size != t.size:
        return np.nan

    return float(np.trapezoid(np.abs(error), t))


def overshoot(y, y_ref, relative=False):
    """
    计算超调量。
    
    超调量 = max(y) - y_ref_final（以末态参考值为基准）
    
    参数:
        y        : 被控量时间序列（如速度）
        y_ref    : 参考值（标量）或参考序列
        relative : False 时返回绝对超调量（工程单位），
                   True 时返回相对超调量（百分比/100）
    
    返回:
        超调量（float），非正值时返回 0.0，空数组返回 nan
    
    示例：
        y_ref = 1.5 m/s，max(y) = 1.8 m/s
        绝对超调 = 0.3 m/s
        相对超调 = 0.3 / 1.5 = 0.2（即 20%）
    """
    y = np.asarray(y, dtype=float)

    if y.size == 0:
        return np.nan

    if np.isscalar(y_ref):
        ref_arr = np.full_like(y, float(y_ref), dtype=float) # type: ignore
    else:
        ref_arr = np.asarray(y_ref, dtype=float)
        if ref_arr.size != y.size:
            return np.nan

    ref = float(ref_arr[-1])      # 以末态参考值为基准
    peak = float(np.max(y))       # 峰值
    os_abs = peak - ref           # 绝对超调

    if not relative:
        return float(max(0.0, os_abs))  # 超调量不能为负

    if abs(ref) < 1e-12:
        return np.nan  # 参考值接近零时相对超调无意义

    return float(max(0.0, os_abs) / abs(ref))


def settling_time(signal, target=0.0, t=None, tol=0.02, abs_band=None):
    """
    计算收敛时间（settling time）。

    定义：从仿真开始，误差首次进入稳定带后，此后不再超出的最早时刻。

    稳定带确定规则（两种模式）：
        1. 若提供 abs_band（绝对阈值，单位与 signal 相同）：
               band = abs_band
           适合横向误差（CTE），直接指定如 0.5m 稳定带。
        2. 否则使用相对模式：
               band = tol × max(peak_abs, 1.0)
           其中 peak_abs = max(|signal - target|)，
           用信号峰值的百分比定义稳定带（传统 2% 稳定时间定义）。

    参数:
        signal   : 信号序列（如横向误差）
        target   : 目标稳态值，默认 0.0
        t        : 时间序列（s）；若为 None，使用索引作为时间
        tol      : 稳定带比例，默认 0.02（相对模式有效）
        abs_band : 绝对稳定带宽度（单位同 signal），提供时优先使用

    返回:
        收敛时间（float），未能收敛时返回 nan
    """
    signal = np.asarray(signal, dtype=float)

    if signal.size == 0:
        return np.nan

    if t is None:
        t = np.arange(signal.size, dtype=float)
    else:
        t = np.asarray(t, dtype=float)
        if t.size != signal.size:
            return np.nan

    err = np.abs(signal - target)

    if abs_band is not None:
        # 绝对稳定带模式（适合 CTE 单位为 m）
        band = float(abs_band)
    else:
        # 相对稳定带模式：基于信号峰值
        peak = float(np.nanmax(err)) if err.size > 0 else 1.0
        band = tol * max(peak, 1.0)

    # 从左到右找首个"之后全部在稳定带内"的位置
    for i in range(len(err)):
        if np.all(err[i:] <= band):
            return float(t[i])

    return np.nan  # 未能收敛


def path_length(x, y):
    """
    计算实际行驶轨迹长度（弧长）。
    
    轨迹长度 = ∑ √((Δx_k)² + (Δy_k)²)
    
    参数:
        x : x 坐标序列（m）
        y : y 坐标序列（m）
    
    返回:
        轨迹总长度（m），空数组或长度不一致时返回 nan
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 2 or y.size < 2 or x.size != y.size:
        return np.nan

    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))


def switch_event_times(t, wp_idx):
    """
    根据航点索引变化提取路径段切换时刻。

    参数:
        t      : 时间序列
        wp_idx : 当前路径段索引序列

    返回:
        发生切换的时间点数组；若无切换返回空数组
    """
    t = np.asarray(t, dtype=float)
    wp_idx = np.asarray(wp_idx, dtype=float)
    if t.size == 0 or wp_idx.size == 0 or t.size != wp_idx.size:
        return np.asarray([], dtype=float)
    if t.size < 2:
        return np.asarray([], dtype=float)
    idx = np.where(np.diff(wp_idx) > 0)[0]
    if idx.size == 0:
        return np.asarray([], dtype=float)
    return t[idx]


def event_window_mask(t, event_times, pre_window=1.0, post_window=8.0):
    """
    为一组事件时刻生成并集时间窗掩码。
    """
    t = np.asarray(t, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    if t.size == 0 or event_times.size == 0:
        return np.zeros_like(t, dtype=bool)
    mask = np.zeros_like(t, dtype=bool)
    for te in event_times:
        mask |= (t >= te - pre_window) & (t <= te + post_window)
    return mask


def max_consecutive_true_duration(mask, dt):
    """
    计算布尔序列中连续 True 的最长持续时间。
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0.0
    max_run = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return float(max_run * dt)



def summarize_tracking_log(
    log,
    tau_r_limit=None,
    tau_r_limit_config=9.0,
    analysis_start_time=0.0,
    turn_pre_window=1.0,
    turn_post_window=8.0,
    cross_track_settling_band=0.5,
    heading_settling_band=np.deg2rad(5.0),
):
    """
    从仿真日志中计算完整的性能指标摘要。
    
    将 SimulationLogger 记录的原始数据汇总为关键性能指标，
    供实验分析、参数优化和论文数据表使用。
    
    参数:
        log : 仿真日志字典（来自 SimulationLogger.as_arrays()）
    
    返回:
        指标字典，包含以下键：
        
        基础信息：
            t_final              : 仿真总时长（s）
            num_steps            : 总步数
            final_dist_to_goal   : 最终到目标点距离（m）
        
        横向误差指标（核心路径跟踪性能）：
            cross_track_rms      : 横向误差 RMS（m）
            cross_track_mae      : 横向误差 MAE（m）
            cross_track_max_abs  : 横向误差峰值（m）
            cross_track_iae      : 横向误差 IAE
            cross_track_ise      : 横向误差 ISE
            cross_track_itae     : 横向误差 ITAE
        
        航向误差指标：
            heading_error_rms    : 航向误差 RMS（rad）
                                   【注意】使用 e_psi = psi_ref - psi。
                                   对有整形层（SHCS）的方法，psi_ref 是整形后的平滑参考，
                                   因此此指标反映"PID 跟踪整形目标的残差"，而非 LOS 几何跟踪精度。
                                   论文中应写"航向参考跟踪误差 RMS"并在图注中说明。
            heading_error_mae    : 航向误差 MAE（rad）
            heading_error_max_abs: 航向误差峰值（rad）
        
        速度误差指标：
            speed_error_rms      : 速度误差 RMS（m/s）
            speed_error_mae      : 速度误差 MAE（m/s）
            speed_error_max_abs  : 速度误差峰值（m/s）
        
        控制能量：
            control_energy_tau_u     : 总纵荡力能量（含扰动）
            control_energy_tau_r     : 总偏航力矩能量（含扰动）
            control_energy_tau_u_cmd : 控制器指令纵荡力能量
            control_energy_tau_r_cmd : 控制器指令偏航力矩能量（论文中常用）
            disturbance_energy_tau_u : 扰动力能量（反映扰动强度）
            disturbance_energy_tau_r : 扰动力矩能量
        
        其他指标：
            trajectory_length        : 实际轨迹长度（m）
            cross_track_settling_time: 横向误差收敛时间（s）
            heading_error_settling_time: 航向误差收敛时间（s）
            final_cross_track_error  : 最终横向误差（m）
            final_heading_error      : 最终航向误差（rad）
            speed_overshoot          : 速度超调量（m/s）
    """
    t = np.asarray(log.get("t", []), dtype=float)
    if t.size == 0:
        return {}

    dt = float(np.mean(np.diff(t))) if t.size > 1 else 0.0

    full_t = t
    analysis_mask = full_t >= float(analysis_start_time)
    if not np.any(analysis_mask):
        analysis_mask = np.ones_like(full_t, dtype=bool)
    t = full_t[analysis_mask]

    def series(key):
        values = np.asarray(log.get(key, []), dtype=float)
        if values.size == full_t.size:
            return values[analysis_mask]
        return values

    # 提取各个误差和控制量序列。全局指标默认排除初始收敛段，避免
    # 初始 -10 m 误差把所有方法的 max/RMS 都“压成一样”。
    e_ct = series("e_ct")        # 横向误差
    e_psi = series("e_psi")      # 控制参考航向误差（psi_ref - psi）
    e_psi_los = series("e_psi_los")  # 几何LOS航向误差（psi_d - psi，对所有方法公平）
    e_u = series("e_u")          # 速度误差
    tau_u = series("tau_u")      # 总纵荡力
    tau_r = series("tau_r")      # 总偏航力矩
    tau_u_cmd = series("tau_u_cmd")      # 指令纵荡力（分配后）
    tau_r_cmd = series("tau_r_cmd")      # 指令偏航力矩（分配后，≤ tau_r_limit）
    tau_r_cmd_raw = series("tau_r_cmd_raw")  # PID 原始输出（分配前，可超出物理上限）
    tau_u_env = series("tau_u_env")      # 扰动纵荡力
    tau_r_env = series("tau_r_env")      # 扰动偏航力矩
    dist_to_goal = series("dist_to_goal")
    x = series("x")
    y = series("y")
    u = series("u")
    u_d = series("u_d") if "u_d" in log else None
    wp_idx = series("wp_idx") if "wp_idx" in log else np.asarray([])
    speed_reduction_pct = series("speed_reduction_pct")
    shaper_delta = series("shaper_delta_psi_raw")

    if tau_r_limit is None:
        tau_r_limit = np.nanmax(np.abs(tau_r_cmd)) if tau_r_cmd.size > 0 else np.nan
    tau_r_limit = float(tau_r_limit) if np.isfinite(tau_r_limit) else np.nan
    sat_eps = max(0.1, 0.005 * tau_r_limit) if np.isfinite(tau_r_limit) else 0.1

    full_wp_idx = np.asarray(log.get("wp_idx", []), dtype=float) if "wp_idx" in log else np.asarray([])
    full_e_ct = np.asarray(log.get("e_ct", []), dtype=float)
    full_e_psi = np.asarray(log.get("e_psi", []), dtype=float)
    full_e_psi_los = np.asarray(log.get("e_psi_los", []), dtype=float)
    full_tau_r_cmd = np.asarray(log.get("tau_r_cmd", []), dtype=float)
    # 使用分配前原始值（tau_r_cmd_raw）检测饱和，更准确：
    # 双桨分配器会因联合约束压低 tau_r_cmd，导致用分配后值永远检测不到饱和。
    full_tau_r_cmd_raw_for_turn = np.asarray(log.get("tau_r_cmd_raw", log.get("tau_r_cmd", [])), dtype=float)
    event_times = switch_event_times(full_t, full_wp_idx)
    turn_mask = event_window_mask(
        full_t,
        event_times,
        pre_window=turn_pre_window,
        post_window=turn_post_window,
    )
    turn_t = full_t[turn_mask]
    turn_e_ct = full_e_ct[turn_mask]
    turn_e_psi = full_e_psi[turn_mask]
    turn_e_psi_los = full_e_psi_los[turn_mask]
    turn_tau_r_cmd = full_tau_r_cmd[turn_mask]
    turn_tau_r_cmd_raw = full_tau_r_cmd_raw_for_turn[turn_mask] if full_tau_r_cmd_raw_for_turn.size == full_t.size else turn_tau_r_cmd
    sat_turn_mask = (
        np.abs(turn_tau_r_cmd_raw) >= (tau_r_limit_config * 0.97)
        if turn_tau_r_cmd_raw.size > 0
        else np.zeros(0, dtype=bool)
    )


    summary = {
        # 基础信息
        "t_final": final_value(t),
        "num_steps": int(len(t)),
        "analysis_start_time": float(analysis_start_time),
        "final_dist_to_goal": final_value(dist_to_goal),

        # 横向误差指标（反映路径跟踪精度，越小越好）
        "cross_track_rms": rms(e_ct),
        "cross_track_mae": mean_abs(e_ct),
        "cross_track_max_abs": max_abs(e_ct),
        "cross_track_iae": iae(e_ct, t),
        "cross_track_ise": ise(e_ct, t),
        "cross_track_itae": itae(e_ct, t),

        # 航向误差指标
        # heading_error_rms：PID 跟踪整形参考的残差（psi_ref - psi）
        #   对有整形层的方法（SHCS），psi_ref 平滑，此值会人为偏小；
        #   仅反映 PID 对整形参考的跟踪质量，不适合跨方法直接比较。
        "heading_error_rms": rms(e_psi),
        "heading_error_mae": mean_abs(e_psi),
        "heading_error_max_abs": max_abs(e_psi),
        # heading_los_error_*：几何 LOS 误差（psi_d - psi），对所有方法公平。
        #   无论有无整形层，psi_d 均为 LOS 几何期望航向，反映船舶实际跟踪路径方向的能力。
        #   论文中若要比较不同方法的航向跟踪性能，应使用此组指标。
        "heading_los_error_rms": rms(e_psi_los),
        "heading_los_error_mae": mean_abs(e_psi_los),
        "heading_los_error_max_abs": max_abs(e_psi_los),

        # 速度误差指标
        "speed_error_rms": rms(e_u),
        "speed_error_mae": mean_abs(e_u),
        "speed_error_max_abs": max_abs(e_u),

        # 控制能量（反映控制器代价）
        "control_energy_tau_u": control_energy(tau_u, dt),        # 总力能量
        "control_energy_tau_r": control_energy(tau_r, dt),
        "control_energy_tau_u_cmd": control_energy(tau_u_cmd, dt),  # 指令力能量
        "control_energy_tau_r_cmd": control_energy(tau_r_cmd, dt),  # 论文常用此项
        "disturbance_energy_tau_u": control_energy(tau_u_env, dt),  # 扰动能量
        "disturbance_energy_tau_r": control_energy(tau_r_env, dt),

        # 轨迹和时域指标
        "trajectory_length": path_length(x, y),
        # CTE 收敛时间：稳定带 0.5 m（横向误差绝对阈值，工程意义明确）
        "cross_track_settling_time": settling_time(
            e_ct,
            target=0.0,
            t=t,
            abs_band=cross_track_settling_band,
        ),
        "heading_error_settling_time": settling_time(
            e_psi,
            target=0.0,
            t=t,
            abs_band=heading_settling_band,
        ),
        "final_cross_track_error": final_value(e_ct),
        "final_heading_error": final_value(e_psi),
        # 路径切换局部瞬态指标（整形方法的主要受益区间）
        "switch_count": int(event_times.size),
        "tau_r_limit": tau_r_limit,
        "turn_window_total_time": float(turn_t[-1] - turn_t[0]) if turn_t.size >= 2 else 0.0,
        "turn_cte_peak_abs": max_abs(turn_e_ct),
        "turn_cte_rms": rms(turn_e_ct),
        "turn_cte_iae": iae(turn_e_ct, turn_t) if turn_t.size >= 2 else np.nan,
        "turn_heading_peak_abs": max_abs(turn_e_psi),
        # 转弯段几何LOS航向误差峰值（公平跨方法比较）
        "turn_heading_los_peak_abs": max_abs(turn_e_psi_los),
        "turn_heading_los_rms": rms(turn_e_psi_los),
        # 转弯饱和统计（用 tau_r_cmd_raw 分配前值，正确反映 PID 是否进入饱和）
        "turn_sat_ratio_tau_r_cmd": float(np.mean(sat_turn_mask)) if sat_turn_mask.size > 0 else 0.0,
        "turn_sat_duration_tau_r_cmd": float(np.sum(sat_turn_mask) * dt) if sat_turn_mask.size > 0 else 0.0,
        "turn_sat_max_consecutive_tau_r_cmd": max_consecutive_true_duration(sat_turn_mask, dt) if sat_turn_mask.size > 0 else 0.0,
        # SHCS/整形层相关指标。普通 LOS-PID 没有速度调度时，这些值自然为 0 或 NaN。
        "speed_reduction_mean_pct": mean_abs(speed_reduction_pct),
        "speed_reduction_max_pct": max_abs(speed_reduction_pct),
        "min_u_d": float(np.nanmin(u_d)) if u_d is not None and u_d.size > 0 else np.nan,
        "mean_u_d": float(np.nanmean(u_d)) if u_d is not None and u_d.size > 0 else np.nan,
        "max_abs_shaper_delta_psi": max_abs(shaper_delta),

        # ── 新增：总控制能耗（纵荡 + 偏航）────────────────────────────────────
        "total_control_energy": (
            (control_energy(tau_r_cmd, dt) or 0.0) +
            (control_energy(tau_u_cmd, dt) or 0.0)
        ),
        # 到达时间（np.nan 为占位符，由 Simulator.run() 在确认到达后覆写为实际值）
        "reach_time": np.nan,

        # ── 饱和统计（使用 PID 原始输出 tau_r_cmd_raw，分配前）───────────────
        # 这才能真实反映 PID 是否进入饱和，而非分配后的截断结果
        "sat_ratio_raw": float(
            np.mean(np.abs(tau_r_cmd_raw) >= tau_r_limit_config * 0.97)
        ) if tau_r_cmd_raw.size > 0 else 0.0,
        "sat_time_raw": float(
            np.sum(np.abs(tau_r_cmd_raw) >= tau_r_limit_config * 0.97) * dt
        ) if tau_r_cmd_raw.size > 0 else 0.0,
        "sat_peak_raw": float(
            np.nanmax(np.abs(tau_r_cmd_raw))
        ) if tau_r_cmd_raw.size > 0 else 0.0,
    }

    # 速度超调量（仅当参考速度序列存在时计算）
    if u.size > 0 and u_d is not None and u_d.size == u.size:
        summary["speed_overshoot"] = overshoot(u, u_d)
    else:
        summary["speed_overshoot"] = np.nan

    return summary


def to_paper_table_rows(
    summaries,
    fields=None,
    rename_map=None,
    precision=4,
):
    """
    将仿真摘要列表转换为论文表格行（支持字段筛选、重命名和数值精度控制）。

    参数:
        summaries  : 仿真摘要字典列表
        fields     : 要包含的字段列表（None 时使用默认集合）
        rename_map : 字段名到表格列名的映射（None 时使用默认映射）
        precision  : 浮点数保留小数位数

    返回:
        行字典列表（列名已按 rename_map 重命名）
    """
    if fields is None:
        fields = [
            "case_name",
            "guidance_type",
            "disturbance_case",
            "cross_track_rms",
            "cross_track_max_abs",
            "heading_error_rms",
            "final_dist_to_goal",
            "control_energy_tau_r_cmd",
        ]

    if rename_map is None:
        rename_map = {
            "case_name": "Case",
            "guidance_type": "Guidance",
            "disturbance_case": "Disturbance",
            "cross_track_rms": "CTE_RMS_m",
            "cross_track_max_abs": "CTE_MaxAbs_m",
            "heading_error_rms": "Heading_RMS_rad",
            "final_dist_to_goal": "FinalGoalDist_m",
            "control_energy_tau_r_cmd": "YawCmdEnergy",
        }

    rows = []
    for summary in summaries:
        row = {}
        for field in fields:
            key = rename_map.get(field, field)
            value = summary.get(field, "")
            if isinstance(value, float):
                row[key] = round(value, precision)
            else:
                row[key] = value
        rows.append(row)
    return rows


# =========================================================================
# 统计检验工具
# =========================================================================

def ci95(values):
    """
    计算均值的 95% 置信区间（t 分布）。

    参数:
        values : 数据序列（list 或 array）

    返回:
        (mean, std, half_width)
        - mean       : 样本均值
        - std        : 样本标准差
        - half_width : 95% CI 半宽；单样本时为 0.0
    """
    from scipy import stats as scipy_stats

    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))

    if n < 2:
        return mean, 0.0, 0.0

    std = float(np.std(values, ddof=1))
    se = std / np.sqrt(n)
    t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
    return mean, std, t_crit * se


def evaluate_metric(values, alpha_normality=0.05, alpha_significance=0.05):
    """
    对单组差值序列做显著性检验（自动选择 t 检验或 Wilcoxon 符号秩检验）。

    检验逻辑：
        - 若样本量 >= 3 且 Shapiro-Wilk 正态性检验 p > alpha_normality：
              使用单样本 t 检验（paired_t_on_delta）
        - 否则：
              使用 Wilcoxon 符号秩检验（wilcoxon）

    参数:
        values             : 差值数组（如 method_CTE - baseline_CTE）
        alpha_normality    : 正态性检验的显著性水平（默认 0.05）
        alpha_significance : 显著性判断阈值（默认 0.05）

    返回:
        字典，包含：
            test_name    : "paired_t_on_delta" 或 "wilcoxon"
            statistic    : 检验统计量
            p_value      : p 值
            significant  : bool，p < alpha_significance
            mean_delta   : 差值均值
    """
    from scipy import stats as scipy_stats

    values = np.asarray(values, dtype=float)
    n = len(values)
    mean_delta = float(np.mean(values))

    use_ttest = False
    if n >= 3:
        _, p_normality = scipy_stats.shapiro(values)
        if p_normality > alpha_normality:
            use_ttest = True

    if use_ttest:
        stat, p = scipy_stats.ttest_1samp(values, popmean=0.0)
        test_name = "paired_t_on_delta"
    else:
        if np.all(values == 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = scipy_stats.wilcoxon(values)
        test_name = "wilcoxon"

    return {
        "test_name": test_name,
        "statistic": float(stat), # type: ignore
        "p_value": float(p), # type: ignore
        "significant": bool(p < alpha_significance), # type: ignore
        "mean_delta": mean_delta,
    }
