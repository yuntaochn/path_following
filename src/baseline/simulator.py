"""
仿真引擎模块

提供 USV 闭环仿真功能，将船舶模型、控制器和扰动系统组合为
完整的仿真回路（simulation loop）。

仿真回路流程（每个时间步 k）：
    1. 传感器测量：对真实状态 η, ν 加入测量噪声 → η_meas, ν_meas
    2. 控制计算：controller(η_meas, ν_meas) → τ_cmd（基于含噪测量值）
    3. 扰动叠加：τ_env（环境力偏置 + 随机噪声）+ τ_cmd → τ_total
    4. 动力学积分：model.derivatives(η, ν, τ_total) + 海流叠加 → η_{k+1}, ν_{k+1}
    5. 日志记录：记录 40+ 个状态/控制/扰动变量
    6. 终止判断：若到达目标点 → 停止

扰动模型（5类）：
    current_velocity  : 海流速度 [vx, vy]，直接叠加到 η̇ 的位置分量
    force_bias        : 恒定环境力偏置 [Fu, Fv, Fn]（如定常风/流体力）
    force_noise_std   : 随机力噪声标准差（模拟波浪等随机扰动）
    control_noise_std : 控制力执行噪声标准差（推进器效率波动）
    eta_noise_std     : 位置/航向测量噪声标准差（GPS/IMU 误差）
    nu_noise_std      : 速度测量噪声标准差（DVL/加速度计 误差）
"""

import numpy as np
from baseline.math_utils import wrap_to_pi
from baseline.metrics import summarize_tracking_log
from baseline.actuator import TwinThrusterAllocator

class SimulationLogger:
    """
    仿真数据日志记录器。
    
    在仿真的每个时间步记录系统全部状态、控制量和扰动量，
    用于后续分析、可视化和性能指标计算。
    
    记录变量（共 40 个）：
    
    时间：
        t           : 仿真时刻（s）
    
    真实状态 η = [x, y, ψ]：
        x, y        : 位置（m）
        psi         : 艏向角（rad）
    
    真实速度 ν = [u, v, r]：
        u           : 纵荡速度（m/s）
        v           : 横荡速度（m/s）
        r           : 偏航角速度（rad/s）
    
    导引/参考信息：
        psi_d       : 期望航向（rad）
        u_d         : 期望速度（m/s）
        path_angle  : 路径段方位角（rad）
        lookahead   : LOS 前视距离（m）
        curvature   : 路径曲率（1/m）
        wp_idx      : 当前路径段索引
    
    误差：
        e_ct        : 横向误差（cross-track error，m）
        e_at        : 纵向误差（along-track error，m）
        e_psi       : 控制参考航向误差 psi_ref - psi（rad）
                      【重要】对有整形层的方法（如 SHCS），psi_ref 是整形后的平滑值，
                      因此 e_psi 反映的是 PID 跟踪整形参考的残差，而非 LOS 几何误差。
                      metrics.py 的 heading_error_rms 使用此字段。
                      论文中应写"航向参考跟踪误差（heading reference tracking error）"。
        e_psi_los   : 几何 LOS 航向误差 psi_d - psi（rad）
                      【重要】对所有方法均为 LOS 几何期望航向与实际航向之差，
                      可用于公平比较各方法的实际 LOS 跟踪能力。
                      当前 metrics.py 未直接计算此字段的 RMS，但日志中有记录。
        e_u         : 速度误差（m/s）
        dist_to_goal: 到目标点距离（m）
    
    控制力（总力 = 指令力 + 扰动力）：
        tau_u/v/r   : 总控制力（N 或 N·m）
        tau_u/v/r_cmd : 控制器指令力
        tau_u/v/r_env : 扰动力/噪声力
    
    状态导数：
        x_dot, y_dot, psi_dot : η̇
        u_dot, v_dot, r_dot   : ν̇
    
    扰动/测量：
        current_x/y         : 海流速度分量（m/s）
        eta_meas_x/y/psi    : 带噪声的位置测量值
        nu_meas_u/v/r       : 带噪声的速度测量值
    """
    def __init__(self):
        self.data = {
            # 时间
            "t": [],

            # 状态：η = [x, y, ψ]
            "x": [],
            "y": [],
            "psi": [],

            # 状态：ν = [u, v, r]
            "u": [],
            "v": [],
            "r": [],

            # 导引/参考信息
            "psi_d": [],         # 期望航向（rad）
            "psi_ref": [],       # 整形后参考航向（rad）
            "u_d": [],           # 期望速度（m/s）
            "u_d_nominal": [],   # 额定速度命令（m/s）
            "path_angle": [],    # 路径段方位角（rad）
            "lookahead": [],     # LOS 前视距离（m）
            "curvature": [],     # 路径曲率（1/m）
            "wp_idx": [],        # 当前路径段索引

            # 误差
            "e_ct": [],          # cross-track error：横向误差（m）
            "e_at": [],          # along-track error：纵向误差（m）
            "e_ct_meas": [],     # 基于含噪测量的横向误差（m）
            "e_at_meas": [],     # 基于含噪测量的纵向误差（m）
            "e_ct_true": [],     # 基于真实状态的横向误差（m）
            "e_at_true": [],     # 基于真实状态的纵向误差（m）
            "e_psi": [],         # 控制参考航向误差 psi_ref - psi（rad）
            "e_psi_los": [],     # 几何 LOS 航向误差 psi_d - psi（rad）
            "e_u": [],           # surge speed error：速度误差（m/s）
            "dist_to_goal": [],  # 到目标点的距离（m）
            "shaper_delta_psi_raw": [],  # 整形误差 e_s = psi_d - psi_ref（rad）
            "shaper_r_lim": [],          # 整形速率上限（rad/s）
            "shaper_method": [],         # 整形方法名
            "speed_scheduler_mode": [],  # 速度调度模式
            "speed_reduction_pct": [],   # 相对额定速度降速比例（%）
            "tau_r_preview": [],         # 调度器使用的偏航力矩预览（N*m）

            # 控制力
            "tau_u": [],         # 总纵荡力（指令 + 扰动）
            "tau_v": [],         # 总横荡力
            "tau_r": [],         # 总偏航力矩
            "tau_u_cmd": [],        # 执行器分配后可实现的纵荡力
            "tau_v_cmd": [],        # 执行器分配后可实现的横荡力
            "tau_r_cmd": [],        # 执行器分配后可实现的偏航力矩
            "tau_u_cmd_raw": [],    # PID 原始指令纵荡力（分配前）
            "tau_v_cmd_raw": [],    # PID 原始指令横荡力（分配前）
            "tau_r_cmd_raw": [],    # PID 原始指令偏航力矩（分配前）
            "tau_u_alloc_error": [],# 分配截断误差：raw - alloc（纵荡）
            "tau_r_alloc_error": [],# 分配截断误差：raw - alloc（偏航）
            "thruster_left": [],    # 左推进器分配推力（N）
            "thruster_right": [],   # 右推进器分配推力（N）
            "thruster_sat_left": [],    # 左推进器饱和标志（0/1）
            "thruster_sat_right": [],   # 右推进器饱和标志（0/1）
            "thruster_utilization": [], # 最大推进器利用率（0~1）
            "tau_u_env": [],     # 扰动纵荡力（环境力 + 噪声）
            "tau_v_env": [],     # 扰动横荡力
            "tau_r_env": [],     # 扰动偏航力矩

            # 状态导数
            "x_dot": [],
            "y_dot": [],
            "psi_dot": [],
            "u_dot": [],
            "v_dot": [],
            "r_dot": [],

            # 扰动/测量噪声信息
            "current_x": [],       # 海流 x 分量（m/s）
            "current_y": [],       # 海流 y 分量（m/s）
            "eta_meas_x": [],      # 含噪 x 测量值
            "eta_meas_y": [],      # 含噪 y 测量值
            "eta_meas_psi": [],    # 含噪 ψ 测量值
            "nu_meas_u": [],       # 含噪 u 测量值
            "nu_meas_v": [],       # 含噪 v 测量值
            "nu_meas_r": [],       # 含噪 r 测量值
        }

    def append(self, **kwargs):
        """
        记录一个时间步的所有数据。
        
        未传入的键会自动填充 np.nan，确保所有数组等长。
        """
        for key in self.data:
            self.data[key].append(kwargs.get(key, np.nan))

    def as_arrays(self):
        """将所有列表转换为 numpy 数组，便于后续计算"""
        return {k: np.asarray(v) for k, v in self.data.items()}

class Simulator:
    """
    USV 闭环仿真引擎。
    
    将船舶动力学模型、控制器和扰动系统组合为完整的仿真回路，
    支持多种积分方法、全类型扰动注入和详细的性能指标汇总。
    
    依赖接口（鸭子类型）：
        model.derivatives(eta, nu, tau) → (eta_dot, nu_dot)
        model.step(eta, nu, tau, dt, method=...) → (eta_next, nu_next)
        controller(eta, nu, waypoints, dt) → (tau, info)
        controller.reset()
    
    参数:
        model              : 动力学模型（SimpleUSV3DOF 或 USV3DOF）
        controller         : 控制器（USVLOSController）
        dt                 : 仿真时间步长（s），默认 0.05s（20Hz）
        t_final            : 最长仿真时间（s），默认 100s
        integration_method : 积分方法，"euler" 或 "rk4"，默认 "rk4"
        disturbance_config : 扰动参数字典，支持以下键：
            "current_velocity"  : 海流速度 [vx, vy]（m/s）
            "force_bias"        : 恒定扰动力 [Fu, Fv, Fn]（N 或 N·m）
            "force_noise_std"   : 随机力噪声标准差 [std_u, std_v, std_r]
            "control_noise_std" : 控制噪声标准差 [std_u, std_v, std_r]
            "eta_noise_std"     : 位置测量噪声标准差 [std_x, std_y, std_psi]
            "nu_noise_std"      : 速度测量噪声标准差 [std_u, std_v, std_r]
        random_seed        : 随机数种子（保证蒙特卡洛实验可复现）
    
    返回结果（run 方法返回字典）：
        "log"               : 所有时间步的状态/控制/扰动数据（numpy 数组字典）
        "summary"           : 性能指标汇总（RMS误差、控制能量等）
        "waypoints"         : 使用的路径点
        "dt"                : 仿真步长
        "t_final"           : 最长仿真时间
        "integration_method": 使用的积分方法
    """
    def __init__(self,
                 model,
                 controller,
                 dt=0.05,
                 t_final=100.0,
                 integration_method="rk4",
                 disturbance_config=None,
                 random_seed=None,
                 metrics_config=None,
                 actuator_allocator=None):
        self.model = model
        self.controller = controller
        self.dt = dt
        self.t_final = t_final
        self.integration_method = integration_method
        self.disturbance_config = {} if disturbance_config is None else dict(disturbance_config)
        self.random_seed = random_seed
        self.metrics_config = {} if metrics_config is None else dict(metrics_config)
        self.actuator_allocator = actuator_allocator
        self.rng = np.random.default_rng(random_seed)

    def _allocate_control(self, tau_cmd_raw: np.ndarray) -> dict:
        """将控制指令通过双桨分配器映射到物理可实现推力，施加单桨饱和约束。

        若未设置 actuator_allocator，则直接透传（不施加饱和约束）。
        """
        tau_cmd_raw = np.asarray(tau_cmd_raw, dtype=float).reshape(3,)
        if self.actuator_allocator is None:
            return {
                "tau_cmd_raw": tau_cmd_raw,
                "tau_cmd_alloc": tau_cmd_raw.copy(),
                "thruster_raw": np.array([np.nan, np.nan]),
                "thruster_alloc": np.array([np.nan, np.nan]),
                "saturated": False,
                "sat_left": False,
                "sat_right": False,
                "utilization_max": np.nan,
            }
        return self.actuator_allocator.allocate(tau_cmd_raw)


    def _noise_vector(self, key, dim):
        """
        从扰动配置中读取噪声标准差，生成对应维度的高斯噪声向量。
        
        参数:
            key : 扰动配置字典中的键名（如 "force_noise_std"）
            dim : 噪声向量维度
        
        返回:
            噪声向量（均值为 0，标准差由配置决定）
            若标准差全为 0，直接返回零向量（提高效率）
        """
        std = np.asarray(self.disturbance_config.get(key, np.zeros(dim)), dtype=float)
        if std.shape == ():
            std = np.full(dim, float(std), dtype=float)
        std = std.reshape(dim,)
        if not np.any(std):
            return np.zeros(dim, dtype=float)
        return self.rng.normal(loc=0.0, scale=std, size=dim)


    def _current_velocity(self):
        """
        获取海流速度向量 [vx, vy]（大地坐标系）。
        
        海流以直接叠加到 η̇ 的位置分量方式实现，
        即不论船体速度如何，海流都会使船漂移。
        """
        current = np.asarray(
            self.disturbance_config.get("current_velocity", np.zeros(2)),
            dtype=float,
        )
        return current.reshape(2,)


    def _environment_force(self):
        """
        计算环境扰动力（偏置力 + 随机噪声力）。
        
        返回 3 维向量 [F_u, F_v, F_r]，叠加到控制力上，
        模拟风、波浪等环境干扰。
        """
        # 恒定偏置力（如风力、固定流体力）
        bias = np.asarray(
            self.disturbance_config.get("force_bias", np.zeros(3)),
            dtype=float,
        ).reshape(3,)
        # 随机噪声力（如波浪扰动）
        noise = self._noise_vector("force_noise_std", 3)
        return bias + noise

    def _control_disturbance(self):
        """
        计算控制力执行噪声（推进器效率波动等）。
        
        返回 3 维噪声向量，叠加到控制指令上，
        模拟推进器响应的不确定性。
        """
        return self._noise_vector("control_noise_std", 3)


    def _measurement(self, eta, nu):
        """
        对真实状态加入传感器测量噪声。
        
        模拟 GPS/IMU/DVL 等传感器的测量误差，
        控制器基于含噪测量值（而非真实状态）进行计算，
        更贴近实际系统。
        
        参数:
            eta : 真实姿态 [x, y, ψ]
            nu  : 真实速度 [u, v, r]
        
        返回:
            eta_meas : 含噪姿态测量值
            nu_meas  : 含噪速度测量值
        """
        eta_meas = np.asarray(eta, dtype=float).copy()
        nu_meas = np.asarray(nu, dtype=float).copy()

        # 加入测量噪声
        eta_meas += self._noise_vector("eta_noise_std", 3)
        nu_meas += self._noise_vector("nu_noise_std", 3)
        # 艏向角归一化，防止加噪后超出 [-π, π)
        eta_meas[2] = wrap_to_pi(eta_meas[2])
        return eta_meas, nu_meas


    def _derivatives_with_disturbance(self, eta, nu, tau_total):
        """
        计算含海流扰动的状态导数。
        
        海流叠加方式：直接加到 η̇ 的位置分量（x 和 y）：
            η̇[:2] += current_velocity
        
        物理意义：海流以恒定速度拖曳船体位置，与船体速度独立。
        
        参数:
            eta       : 当前姿态
            nu        : 当前速度
            tau_total : 总控制力（指令 + 扰动）
        
        返回:
            eta_dot : 含海流的姿态导数
            nu_dot  : 速度导数（不受海流影响）
        """
        eta_dot, nu_dot = self.model.derivatives(eta, nu, tau_total)
        current = self._current_velocity()
        eta_dot = np.asarray(eta_dot, dtype=float).copy()
        # 海流叠加到位置导数（仅影响 x, y，不影响 ψ）
        eta_dot[:2] += current
        return eta_dot, nu_dot
    

    def _step_dynamics(self, eta, nu, tau_total):
        """
        执行一步含扰动的动力学积分。
        
        支持 Euler 法和 RK4 法，均考虑海流扰动的影响。
        
        参数:
            eta       : 当前姿态
            nu        : 当前速度
            tau_total : 总控制力
        
        返回:
            eta_next : 下一时刻姿态
            nu_next  : 下一时刻速度
        """
        if self.integration_method == "euler":
            # 一阶 Euler 积分
            eta_dot, nu_dot = self._derivatives_with_disturbance(eta, nu, tau_total)
            eta_next = np.asarray(eta, dtype=float) + eta_dot * self.dt
            nu_next = np.asarray(nu, dtype=float) + nu_dot * self.dt
        elif self.integration_method == "rk4":
            # RK4 四阶龙格-库塔积分（每步调用 4 次导数函数）
            def f(state):
                eta_s = state[:3]
                nu_s = state[3:]
                eta_dot_s, nu_dot_s = self._derivatives_with_disturbance(
                    eta_s, nu_s, tau_total,
                )
                return np.concatenate([eta_dot_s, nu_dot_s])

            state = np.concatenate([eta, nu])
            k1 = f(state)
            k2 = f(state + 0.5 * self.dt * k1)
            k3 = f(state + 0.5 * self.dt * k2)
            k4 = f(state + self.dt * k3)
            state_next = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            eta_next = state_next[:3]
            nu_next = state_next[3:]
        else:
            raise ValueError("integration_method must be 'euler' or 'rk4'")

        # 艏向角归一化
        eta_next[2] = wrap_to_pi(eta_next[2])
        return eta_next, nu_next


    @staticmethod
    def _compute_path_errors(position, waypoints, wp_idx):
        """
        基于真实位置和当前路径段索引，计算几何误差（s_along, e_ct）。

        说明：
            该误差不含测量噪声，用于更真实地评价跟踪质量和绘图可读性。
        """
        wps = np.asarray(waypoints, dtype=float)
        pos = np.asarray(position, dtype=float).reshape(2,)
        if wps.shape[0] < 2:
            return np.nan, np.nan

        idx = int(np.clip(int(wp_idx), 0, wps.shape[0] - 2))
        p0 = wps[idx]
        p1 = wps[idx + 1]
        seg = p1 - p0
        seg_norm = float(np.linalg.norm(seg))
        if seg_norm < 1e-9:
            return np.nan, np.nan

        path_angle = float(np.arctan2(seg[1], seg[0]))
        c, s = np.cos(path_angle), np.sin(path_angle)
        rel = pos - p0
        s_along = c * rel[0] + s * rel[1]
        e_ct = -s * rel[0] + c * rel[1]
        return float(s_along), float(e_ct)


    @classmethod
    def _terminal_reached(
        cls,
        position,
        waypoints,
        wp_idx,
        dist_to_goal,
        goal_tolerance,
    ):
        """
        判断是否到达终点。

        论文实验采用严格的欧氏距离半径判定：只有真实位置进入终点
        goal_tolerance 圆内，才认为任务完成。旧版本还允许“末段投影接近
        终点且横向误差较小”时提前停止，这会造成 reached_goal=True 但
        final_dist_to_goal 大于目标半径，图上看起来像离目标还很远。
        """
        return bool(dist_to_goal <= goal_tolerance)


    @classmethod
    def _terminal_crossed_goal_line(
        cls,
        position,
        waypoints,
        wp_idx,
        goal_tolerance,
    ):
        """
        判断是否已经越过终点截面。

        这是开放路径仿真的“停止事件”，不是“到达目标”事件。若对比方法
        在海流下从终点旁边擦过但没有进入目标圆，继续仿真会沿末段无限
        远离目标，导致图形被拉伸且指标被无关的末端漂移主导。因此：

        - `reached_goal` 只由目标圆半径决定；
        - `path_completed` 可由终点截面触发，用于停止绘图和统计时间窗；
        - 论文表格仍保留 `final_dist_to_goal` 和 `reached_goal`，不把擦过终点
          包装成真正到达。
        """
        wps = np.asarray(waypoints, dtype=float)
        if wps.shape[0] < 2:
            return False

        final_seg_idx = wps.shape[0] - 2
        if int(wp_idx) < final_seg_idx:
            return False

        p0 = wps[-2]
        p1 = wps[-1]
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len < 1e-9:
            return False

        s_along, e_ct = cls._compute_path_errors(
            position=position,
            waypoints=waypoints,
            wp_idx=final_seg_idx,
        )
        crossing_band = 1.5 * float(goal_tolerance)
        return bool(s_along >= seg_len and abs(e_ct) <= crossing_band)


    def run(self, eta0, nu0, waypoints, goal_tolerance=3.0, stop_when_reached=True):
        """
        运行完整闭环仿真。
        
        参数:
            eta0             : 初始姿态 [x0, y0, ψ0]
            nu0              : 初始速度 [u0, v0, r0]（通常为零向量）
            waypoints        : 路径点列表，shape (N, 2)
            goal_tolerance   : 目标到达判断阈值（m），默认 3.0m
            stop_when_reached: True 时到达目标点后立即停止仿真
        
        返回:
            字典，包含以下键：
                "log"               : 仿真数据（numpy 数组字典）
                "summary"           : 性能指标（cross_track_rms, heading_error_rms 等）
                "waypoints"         : 路径点
                "dt"                : 仿真步长
                "t_final"           : 最长仿真时间
                "integration_method": 积分方法名称
        """
        # 初始化状态
        eta = np.asarray(eta0, dtype=float).copy()
        nu = np.asarray(nu0, dtype=float).copy()
        waypoints = np.asarray(waypoints, dtype=float)

        # 重置控制器（清除 PID 积分和 LOS 航点索引）
        self.controller.reset()
        logger = SimulationLogger()

        steps = int(self.t_final / self.dt)  # 最大仿真步数
        reached_goal = False
        path_completed = False
        reach_time = None
        completion_time = None
        completion_reason = "t_final"

        for k in range(steps):
            t = k * self.dt

            # Step 1: 传感器测量（加入测量噪声）
            eta_meas, nu_meas = self._measurement(eta, nu)

            # Step 2: 控制计算（基于含噪测量值）
            tau_cmd_raw, info = self.controller(eta_meas, nu_meas, waypoints, self.dt)

            # Step 3: 执行器分配（双桨 + 单桨饱和）
            alloc = self._allocate_control(tau_cmd_raw)
            tau_cmd = np.asarray(alloc["tau_cmd_alloc"], dtype=float)

            # Step 4: 扰动力叠加（环境力 + 控制噪声）
            tau_env = self._environment_force() + self._control_disturbance()
            tau = tau_cmd + tau_env  # 总控制力

            # 计算当前步的状态导数（仅用于日志记录 ν̇/η̇，不推进状态）
            eta_dot, nu_dot = self._derivatives_with_disturbance(eta, nu, tau)

            # 计算误差
            psi_d = info["psi_d"]
            psi_ref = info.get("psi_ref", psi_d)
            u_d = info["u_d"]
            # 对有整形层的方法，PID 实际跟踪的是 psi_ref，而不是可能跳变的 psi_d。
            # 因此 e_psi 记录控制参考误差；e_psi_los 额外保留几何 LOS 误差供分析。
            e_psi = wrap_to_pi(psi_ref - eta[2])
            e_psi_los = wrap_to_pi(psi_d - eta[2])
            e_u = u_d - nu[0]                     # 速度误差
            dist_to_goal = np.linalg.norm(eta[:2] - waypoints[-1])  # 到终点距离
            e_at_meas = info.get("along_track_error", np.nan)
            e_ct_meas = info.get("cross_track_error", np.nan)
            e_at_true, e_ct_true = self._compute_path_errors(
                position=eta[:2],
                waypoints=waypoints,
                wp_idx=info.get("wp_idx", 0),
            )

            # Step 5: 记录当前时刻所有数据
            logger.append(
                t=t,

                # 真实状态
                x=eta[0],
                y=eta[1],
                psi=eta[2],
                u=nu[0],
                v=nu[1],
                r=nu[2],

                # 导引/参考信息
                psi_d=psi_d,
                psi_ref=info.get("psi_ref", np.nan),
                u_d=u_d,
                u_d_nominal=info.get("u_d_nominal", np.nan),
                # info.get() 的 np.nan 默认值：若键不存在（如固定LOS无曲率），记录为 NaN
                path_angle=info.get("path_angle", np.nan),
                lookahead=info.get("lookahead", np.nan),
                curvature=info.get("curvature", np.nan),
                wp_idx=info.get("wp_idx", np.nan),

                # 误差
                # e_ct/e_at 统一记录为“真实几何误差”，避免测量噪声污染性能指标。
                # 同时保留 *_meas 作为调试用数据。
                e_ct=e_ct_true,
                e_at=e_at_true,
                e_ct_meas=e_ct_meas,
                e_at_meas=e_at_meas,
                e_ct_true=e_ct_true,
                e_at_true=e_at_true,
                e_psi=e_psi,
                e_psi_los=e_psi_los,
                e_u=e_u,
                dist_to_goal=dist_to_goal,
                shaper_delta_psi_raw=info.get("shaper_delta_psi_raw", np.nan),
                shaper_r_lim=info.get("shaper_r_lim", np.nan),
                shaper_method=info.get("shaper_method", "none"),
                speed_scheduler_mode=info.get("speed_scheduler_mode", "none"),
                speed_reduction_pct=info.get("speed_reduction_pct", 0.0),
                tau_r_preview=info.get("tau_r_preview", 0.0),

                # 控制力分解
                tau_u=tau[0],              # 总纵荡力
                tau_v=tau[1],              # 总横荡力
                tau_r=tau[2],              # 总偏航力矩
                tau_u_cmd=tau_cmd[0],      # 执行器分配后可实现力
                tau_v_cmd=tau_cmd[1],
                tau_r_cmd=tau_cmd[2],
                tau_u_cmd_raw=tau_cmd_raw[0],  # PID 原始指令（分配前）
                tau_v_cmd_raw=tau_cmd_raw[1],
                tau_r_cmd_raw=tau_cmd_raw[2],
                tau_u_alloc_error=float(tau_cmd_raw[0] - tau_cmd[0]),
                tau_r_alloc_error=float(tau_cmd_raw[2] - tau_cmd[2]),
                thruster_left=float(alloc["thruster_alloc"][0]) if not np.isnan(alloc["thruster_alloc"][0]) else np.nan,
                thruster_right=float(alloc["thruster_alloc"][1]) if not np.isnan(alloc["thruster_alloc"][1]) else np.nan,
                thruster_sat_left=float(alloc["sat_left"]),
                thruster_sat_right=float(alloc["sat_right"]),
                thruster_utilization=float(alloc["utilization_max"]),
                tau_u_env=tau_env[0],  # 扰动力
                tau_v_env=tau_env[1],
                tau_r_env=tau_env[2],

                # 状态导数
                x_dot=eta_dot[0],
                y_dot=eta_dot[1],
                psi_dot=eta_dot[2],
                u_dot=nu_dot[0],
                v_dot=nu_dot[1],
                r_dot=nu_dot[2],

                # 扰动/测量信息
                current_x=self._current_velocity()[0],
                current_y=self._current_velocity()[1],
                eta_meas_x=eta_meas[0],
                eta_meas_y=eta_meas[1],
                eta_meas_psi=eta_meas[2],
                nu_meas_u=nu_meas[0],
                nu_meas_v=nu_meas[1],
                nu_meas_r=nu_meas[2],
            )

            # Step 6: 终止判断
            terminal_reached = self._terminal_reached(
                position=eta[:2],
                waypoints=waypoints,
                wp_idx=info.get("wp_idx", 0),
                dist_to_goal=dist_to_goal,
                goal_tolerance=goal_tolerance,
            )
            if terminal_reached and not reached_goal:
                reached_goal = True
                path_completed = True
                reach_time = t
                completion_time = t
                completion_reason = "goal_radius"
                if stop_when_reached:
                    break

            terminal_crossed = self._terminal_crossed_goal_line(
                position=eta[:2],
                waypoints=waypoints,
                wp_idx=info.get("wp_idx", 0),
                goal_tolerance=goal_tolerance,
            )
            if terminal_crossed and not path_completed:
                path_completed = True
                completion_time = t
                completion_reason = "crossed_goal_line"
                if stop_when_reached:
                    break

            # Step 7: 状态推进（积分到下一步） （关键步骤）
            eta, nu = self._step_dynamics(eta, nu, tau)
            # 再次归一化（防止累积误差）
            eta[2] = wrap_to_pi(eta[2])

        # 将日志列表转换为 numpy 数组
        log = logger.as_arrays()
        tau_r_limit = None
        pid_psi = getattr(self.controller, "pid_psi", None)
        output_limit = getattr(pid_psi, "output_limit", None)
        if output_limit is not None:
            tau_r_limit = max(abs(float(output_limit[0])), abs(float(output_limit[1])))

        # 计算性能指标摘要
        summary = summarize_tracking_log(
            log,
            tau_r_limit=tau_r_limit,
            **self.metrics_config,
        )
        # 添加到达状态信息
        summary["reached_goal"] = reached_goal
        summary["path_completed"] = path_completed
        summary["reach_time"] = np.nan if reach_time is None else float(reach_time)
        summary["completion_time"] = np.nan if completion_time is None else float(completion_time)
        summary["completion_reason"] = completion_reason

        return {
            "log": log,                               # 完整仿真数据
            "summary": summary,                       # 性能指标摘要
            "waypoints": waypoints,                   # 路径点
            "dt": self.dt,                            # 仿真步长
            "t_final": self.t_final,                  # 最长仿真时间
            "integration_method": self.integration_method,  # 积分方法
        }
