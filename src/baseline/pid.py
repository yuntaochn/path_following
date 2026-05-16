"""
PID 控制器模块

实现了一个通用的 PID（比例-积分-微分）控制器，
支持积分限幅、输出限幅、微分低通滤波和角度包络模式。

在 USV 路径跟踪中，本模块被用于：
    - 速度通道（PID_U）：控制纵荡力 tau_u，使实际速度 u 跟踪期望速度 u_d
    - 航向通道（PID_PSI）：控制偏航力矩 tau_r，使实际航向 psi 跟踪 LOS 给出的期望航向 psi_d
"""

import numpy as np
from baseline.math_utils import wrap_to_pi

class PID:
    """
    简单实用的 PID 控制器

    PID 输出公式:
        u = Kp * e + Ki * ∫e dt + Kd * de/dt

    其中：
        e = setpoint - measurement（误差）
        ∫e dt 为积分累积误差（带限幅）
        de/dt 为误差微分（带低通滤波）

    用法:
        pid = PID(kp=1.0, ki=0.1, kd=0.05, output_limit=(-10, 10))
        u = pid(setpoint=1.0, measurement=0.8, dt=0.05)

    参数:
        kp, ki, kd           : PID 参数（比例、积分、微分增益）
        integral_limit       : 积分项限幅，None 表示不限幅
                               防止积分饱和（integrator windup）
        output_limit         : 输出限幅，格式 (min, max)，None 表示不限幅
                               约束控制量在执行机构允许范围内
        derivative_filter    : 微分低通滤波系数，范围[0, 1)
                               0 表示不过滤，越接近1滤波效果越强（响应越慢）
                               滤波公式：d_filtered = α * d_prev + (1-α) * d_raw
        angle_wrap           : 是否开启角度包络模式
                               True 时，误差会通过 wrap_to_pi 归一化到 [-π, π)
                               用于航向控制，避免跨越 ±π 时方向计算错误
    """

    def __init__(
        self,
        kp=0.0,
        ki=0.0,
        kd=0.0,
        integral_limit=None,
        output_limit=None,
        derivative_filter=0.0,
        angle_wrap=False,
        aw_gain=0.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral_limit = integral_limit    # 积分限幅值（对称限幅，即 ±integral_limit）
        self.output_limit = output_limit        # 输出限幅元组 (min, max)
        self.derivative_filter = derivative_filter  # 微分低通滤波系数 α
        self.angle_wrap = angle_wrap            # 是否开启角度包络（航向控制时设为 True）
        # 反算抗积分卷绕增益：aw_gain > 0 时启用 back-calculation anti-windup。
        # 当输出饱和时，以 aw_gain*(u_sat - u_unsat)*dt 修正积分项，使其
        # 在下一步更快退出饱和区，相比纯限幅（clamping）抗卷绕效果更强。
        self.aw_gain = float(aw_gain)

        # 内部状态
        self.integral = 0.0          # 积分累积值
        self.prev_error = None       # 上一步误差（None 表示第一步）
        self.prev_derivative = 0.0   # 上一步滤波后的微分值

    def reset(self):
        """重置控制器内部状态，在每次新仿真开始前调用"""
        self.integral = 0.0
        self.prev_error = None
        self.prev_derivative = 0.0

    def _compute_output(self, setpoint, measurement, dt, update_state):
        """
        执行一步 PID 计算。

        参数:
            setpoint     : 期望值
            measurement  : 测量值
            dt           : 控制周期（s）
            update_state : 是否写回积分/微分内部状态

        返回:
            control_output : PID 控制输出（float）
        """
        if dt <= 0:
            raise ValueError("dt 必须大于 0")

        error = setpoint - measurement
        if self.angle_wrap:
            error = wrap_to_pi(error)

        p = self.kp * error

        integral = self.integral + error * dt
        if self.integral_limit is not None:
            integral = np.clip(
                integral,
                -self.integral_limit,
                self.integral_limit
            )
        i = self.ki * integral

        if self.prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self.prev_error) / dt

        alpha = self.derivative_filter
        derivative = alpha * self.prev_derivative + (1 - alpha) * derivative
        d = self.kd * derivative

        output_unclipped = p + i + d
        if self.output_limit is not None:
            output = np.clip(output_unclipped, self.output_limit[0], self.output_limit[1])
        else:
            output = output_unclipped

        # Back-calculation anti-windup: 当输出饱和时，将截断误差反馈到积分项，
        # 加速退出饱和区，避免积分卷绕。仅在 update_state=True 时生效。
        if update_state and self.aw_gain > 0.0:
            clip_error = output - output_unclipped  # 负值（下饱和）或正值（上饱和），不饱和时为0
            if abs(clip_error) > 1e-9:
                integral += self.aw_gain * clip_error * dt
                if self.integral_limit is not None:
                    integral = np.clip(integral, -self.integral_limit, self.integral_limit)

        if update_state:
            self.integral = integral
            self.prev_error = error
            self.prev_derivative = derivative

        return float(output)


    def preview(self, setpoint, measurement, dt):
        """
        预估当前步 PID 输出，但不更新内部积分/微分状态。

        用途：
            供上层模块在“不打乱真实控制器状态”的前提下估计当前控制需求，
            例如 SHCS 的 coupled 模式需要先估计偏航力矩需求，再据此调度速度。
        """
        return self._compute_output(setpoint, measurement, dt, update_state=False)

    def __call__(self, setpoint, measurement, dt):
        """
        执行一步 PID 计算。

        输入:
            setpoint    : 期望值（目标）
            measurement : 当前测量值
            dt          : 控制周期（单位：秒），必须大于 0

        返回:
            control_output : PID 控制输出（标量 float）

        计算流程:
            1. 计算误差 e = setpoint - measurement
            2. 若 angle_wrap=True，对误差做 wrap_to_pi 处理
            3. 比例项 P = kp * e
            4. 积分项 I = ki * ∫e dt（含限幅）
            5. 微分项 D = kd * de/dt（含低通滤波）
            6. 总输出 u = P + I + D（含输出限幅）
        """
        return self._compute_output(setpoint, measurement, dt, update_state=True)

