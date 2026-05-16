"""
USV 船舶动力学模型模块

本模块实现了 3 自由度（3DOF）无人水面艇的运动学和动力学模型。

坐标系与状态定义：
    - 大地坐标系（NED/Earth frame）：η = [x, y, ψ]
        x   : 北向位置（m）
        y   : 东向位置（m）
        ψ   : 艏向角（rad），以北为零，顺时针为正
    
    - 船体坐标系（Body frame）：ν = [u, v, r]
        u   : 纵荡速度（surge，m/s），沿船体纵轴方向
        v   : 横荡速度（sway，m/s），沿船体横轴方向
        r   : 偏航角速度（yaw rate，rad/s），绕垂直轴

运动学（Kinematics）：
    η̇ = J(ψ) · ν
    
    其中 J(ψ) 为坐标变换矩阵（旋转矩阵）：
        J(ψ) = [[cos ψ, -sin ψ, 0],
                 [sin ψ,  cos ψ, 0],
                 [0,      0,     1]]

动力学（Dynamics）：
    M · ν̇ + C(ν) · ν + D(ν) · ν = τ

    其中：
        M = M_RB + M_A          质量矩阵（刚体质量 + 附加质量）
        C(ν)                    科里奥利向心力矩阵
        D(ν) · ν = τ_d(ν)      非线性阻尼力（直接计算）
        τ = [τ_u, τ_v, τ_r]    控制力向量

本模块提供两个模型类：
    1. SimpleUSV3DOF  : 通用参数化 3DOF USV 模型
    2. USV3DOF        : 基于 CS2 型小型无人艇实船参数的精确模型

参考文献：
    Fossen, T.I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control.
    Skjetne, R. et al. (2004). Adaptive maneuvering, with experiments, for a model ship.
"""

from dataclasses import dataclass, field
import numpy as np
from baseline.math_utils import wrap_to_pi

class _USVStepMixin:
    """
    三步积分方法 Mixin，供 SimpleUSV3DOF 和 USV3DOF 共享。

    两个模型的 step_euler / step_rk4 / step 逻辑完全相同，
    均通过多态调用 self.derivatives() 实现积分，因此提取到此 Mixin
    避免代码重复。子类只需实现 derivatives(eta, nu, tau) 即可。
    """

    def step_euler(self, eta, nu, tau, dt):
        """
        使用一阶 Euler 法进行数值积分（步长较大时精度低，仅用于快速验证）。

        η_{k+1} = η_k + η̇_k · dt
        ν_{k+1} = ν_k + ν̇_k · dt
        """
        eta_dot, nu_dot = self.derivatives(eta, nu, tau)
        eta_next = eta + eta_dot * dt
        nu_next  = nu  + nu_dot  * dt
        eta_next[2] = wrap_to_pi(eta_next[2])
        return eta_next, nu_next

    def step_rk4(self, eta, nu, tau, dt):
        """
        使用四阶龙格-库塔（RK4）法进行数值积分（推荐使用）。

        RK4 每步使用 4 个斜率估计，截断误差 O(dt⁵)（Euler 为 O(dt²)）。
        tau 在整个时间步内视为常值（零阶保持器假设）。
        """
        def f(state):
            eta_s, nu_s = state[:3], state[3:]
            eta_dot_s, nu_dot_s = self.derivatives(eta_s, nu_s, tau)
            return np.concatenate([eta_dot_s, nu_dot_s])

        state = np.concatenate([eta, nu])
        k1 = f(state)
        k2 = f(state + 0.5 * dt * k1)
        k3 = f(state + 0.5 * dt * k2)
        k4 = f(state + dt * k3)
        state_next = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        eta_next = state_next[:3]
        nu_next  = state_next[3:]
        eta_next[2] = wrap_to_pi(eta_next[2])
        return eta_next, nu_next

    def step(self, eta, nu, tau, dt, method="rk4"):
        """
        执行一步动力学积分，分派到具体积分方法。

        参数:
            method : "euler"（一阶 Euler）或 "rk4"（四阶龙格-库塔，默认）
        """
        if method == "euler":
            return self.step_euler(eta, nu, tau, dt)
        elif method == "rk4":
            return self.step_rk4(eta, nu, tau, dt)
        else:
            raise ValueError("method must be 'euler' or 'rk4'")


class SimpleUSV3DOF(_USVStepMixin):
    """
    通用 3DOF USV 动力学模型。
    
    状态方程：
        η = [x, y, ψ]         大地坐标位置和艏向
        ν = [u, v, r]         船体坐标速度
    
    运动学：η̇ = J(ψ) · ν
    动力学：M · ν̇ = τ - C(ν) · ν - τ_d(ν)
    
    质量矩阵结构：
        M = M_RB + M_A
        M_RB = [[m,    0,    0  ],    刚体质量矩阵
                [0,    m,    m·xg],
                [0,    m·xg, Iz ]]
        M_A  = [[Xu_dot, 0,      0     ],    附加质量矩阵（水动力附加质量）
                [0,      Yv_dot, Yr_dot],    注：附加质量系数均为负值
                [0,      Nv_dot, Nr_dot]]
    
    阻尼力计算（非线性阻尼）：
        τ_d[0] = Xu·u + Xuu·|u|·u    纵向阻尼（含二次项）
        τ_d[1] = Yv·v + Yvv·|v|·v    横向阻尼
        τ_d[2] = Nr·r + Nrr·|r|·r    偏航阻尼
    
    参数（均有物理意义）：
        m        : 船体质量（kg）
        Iz       : 绕垂直轴转动惯量（kg·m²）
        xg       : 重心相对船体坐标系原点的纵向偏移（m）
        Xu_dot   : 纵向附加质量系数（kg），负值
        Yv_dot   : 横向附加质量系数（kg），负值
        Nr_dot   : 偏航附加惯量系数（kg·m²），负值
        Yr_dot   : 横向-偏航耦合附加质量（通常为 0）
        Nv_dot   : 偏航-横向耦合附加质量（通常为 0）
        Xu       : 纵向线性阻尼系数（N·s/m）
        Yv       : 横向线性阻尼系数（N·s/m）
        Nr       : 偏航线性阻尼系数（N·m·s/rad）
        Xuu      : 纵向二次阻尼系数（N·s²/m²）
        Yvv      : 横向二次阻尼系数（N·s²/m²）
        Nrr      : 偏航二次阻尼系数（N·m·s²/rad²）
    """
    def __init__(
        self,
        m=30.0,        # 船体质量（kg）
        Iz=4.1,        # 转动惯量（kg·m²）
        xg=0.0,        # 重心纵向偏移（m）
        Xu_dot=-2.0,   # 纵向附加质量（kg）
        Yv_dot=-10.0,  # 横向附加质量（kg）
        Nr_dot=-1.0,   # 偏航附加惯量（kg·m²）
        Yr_dot=0.0,    # 横-偏耦合附加质量
        Nv_dot=0.0,    # 偏-横耦合附加质量
        Xu=4.0,        # 纵向线性阻尼（N·s/m）
        Yv=6.0,        # 横向线性阻尼（N·s/m）
        Nr=1.0,        # 偏航线性阻尼（N·m·s）
        Xuu=8.0,       # 纵向二次阻尼（N·s²/m²）
        Yvv=15.0,      # 横向二次阻尼（N·s²/m²）
        Nrr=3.0,       # 偏航二次阻尼
    ):
        self.m = m
        self.Iz = Iz
        self.xg = xg

        # 附加质量系数（水动力作用，均为负值）
        self.Xu_dot = Xu_dot
        self.Yv_dot = Yv_dot
        self.Nr_dot = Nr_dot
        self.Yr_dot = Yr_dot
        self.Nv_dot = Nv_dot

        # 线性和二次阻尼系数
        self.Xu = Xu
        self.Yv = Yv
        self.Nr = Nr
        self.Xuu = Xuu
        self.Yvv = Yvv
        self.Nrr = Nrr

        # 刚体质量矩阵 M_RB
        self.M_RB = np.array([
            [m, 0.0, 0.0],
            [0.0, m, m * xg],       # 重心偏置引起的惯性耦合
            [0.0, m * xg, Iz],
        ], dtype=float)

        # 附加质量矩阵 M_A（注：系数本身为负值，加入后 M 减小）
        # M_A 代表船运动时须"带动"的周围流体质量
        self.M_A = np.array([
            [Xu_dot, 0.0, 0.0],
            [0.0, Yv_dot, Yr_dot],
            [0.0, Nv_dot, Nr_dot],
        ], dtype=float)

        # 总质量矩阵 M = M_RB + M_A
        self.M = self.M_RB + self.M_A
        self.M_inv = np.linalg.inv(self.M)  # 预计算逆矩阵，提高仿真效率

        # 矩阵元素（用于 C(ν) 计算）
        self.m11 = self.M[0, 0]  # (m - Xu_dot)
        self.m22 = self.M[1, 1]  # (m - Yv_dot)
        self.m23 = self.M[1, 2]  # (m·xg - Yr_dot)
        self.m32 = self.M[2, 1]  # (m·xg - Nv_dot)
        self.m33 = self.M[2, 2]  # (Iz - Nr_dot)

    def J(self, psi):
        """
        坐标变换矩阵 J(ψ)，将船体速度变换到大地坐标系速度。
        
        η̇ = J(ψ) · ν
        
        参数:
            psi : 艏向角（rad）
        
        返回:
            3x3 旋转矩阵
        """
        c = np.cos(psi)
        s = np.sin(psi)
        return np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=float)

    def C(self, nu):
        """
        科里奥利向心力矩阵 C(ν)。
        
        来源：船在旋转参考系中运动时出现的惯性效应。
        结构（反对称矩阵）：
            C = [[0,    0,    c13],
                 [0,    0,    c23],
                 [-c13, -c23, 0  ]]
        
        其中：
            c13 = -(m22·v + m23·r)
            c23 = m11·u
        
        参数:
            nu : 速度向量 [u, v, r]
        
        返回:
            3x3 科里奥利矩阵
        """
        u, v, r = nu 
        c_13 = -(self.m22*v + self.m23*r)
        c_23 = self.m11 * u 
        return np.array([
            [0.0, 0.0, c_13], 
            [0.0, 0.0, c_23], 
            [-c_13, -c_23, 0.0]], dtype=float)

    def tau_d(self, nu):
        """
        计算非线性阻尼力（直接输出力向量，而非阻尼矩阵）。
        
        阻尼力 = 线性阻尼 + 二次（速度平方）阻尼
        
        纵向：τ_d[0] = Xu·u + Xuu·|u|·u    （阻力随速度平方增大）
        横向：τ_d[1] = Yv·v + Yvv·|v|·v
        偏航：τ_d[2] = Nr·r + Nrr·|r|·r
        
        参数:
            nu : 速度向量 [u, v, r]
        
        返回:
            阻尼力向量 [τ_d_u, τ_d_v, τ_d_r]
        """
        u, v, r = nu
        tau_1 = self.Xu*u + self.Xuu*abs(u)*u 
        tau_2 = self.Yv*v + self.Yvv*abs(v)*v 
        tau_3 = self.Nr*r + self.Nrr*abs(r)*r 

        tau_d = np.array([tau_1, tau_2, tau_3], dtype=float)
        return tau_d  

    def derivatives(self, eta, nu, tau):
        """
        计算系统状态导数（η̇ 和 ν̇）。
        
        运动学：η̇ = J(ψ) · ν
        动力学：ν̇ = M⁻¹ · (τ - C(ν)·ν - τ_d(ν))
        
        参数:
            eta : 当前姿态 [x, y, ψ]
            nu  : 当前速度 [u, v, r]
            tau : 控制力 [τ_u, τ_v, τ_r]
        
        返回:
            eta_dot : 姿态导数 [ẋ, ẏ, ψ̇]
            nu_dot  : 速度导数 [u̇, v̇, ṙ]
        """
        eta = np.asarray(eta, dtype=float).reshape(3,)
        nu = np.asarray(nu, dtype=float).reshape(3,)
        tau = np.asarray(tau, dtype=float).reshape(3,)

        _, _, psi = eta
        # 运动学：大地坐标系下的速度
        eta_dot = self.J(psi) @ nu 
        # 动力学：M · ν̇ = τ - C·ν - τ_d  →  ν̇ = M⁻¹ · (τ - C·ν - τ_d)
        nu_dot = self.M_inv @ (tau - self.C(nu)@nu - self.tau_d(nu))
        return eta_dot, nu_dot

    # step_euler / step_rk4 / step 由 _USVStepMixin 提供，无需在此重复定义。

@dataclass
class CS2Params:
    """
    CS2 型小型无人水面艇的实船参数数据类。
    
    CS2（Cybership II）是挪威科技大学（NTNU）的 1:70 缩比无人艇模型，
    广泛用于船舶控制研究。质量约 23.8 kg，是目前文献中参数最完整的无人艇模型之一。
    
    质量和惯性参数：
        m    : 23.8 kg（船体质量）
        Iz   : 1.76 kg·m²（绕垂直轴转动惯量）
        xg   : 0.046 m（重心纵向偏移量，正值表示重心在原点前方）
    
    附加质量系数（水动力，均为负值表示抵抗运动）：
        X_du : -2.0    纵向附加质量
        Y_dv : -10.0   横向附加质量
        N_dr : -1.0    偏航附加惯量
    
    线性和二次阻尼系数（D 矩阵相关）：
        X_u, X_uu  : 纵向阻尼（线性 + 二次项）
        Y_v, Y_vv  : 横向阻尼
        N_r, N_rr  : 偏航阻尼
        Y_r, Y_rr  : 横向-偏航耦合阻尼
        N_v, N_vv  : 偏航-横向耦合阻尼
    
    交叉阻尼项（速度乘积项，CS2 精确模型特有）：
        Yrv, Nrv   : r·v 速度乘积对 Y, N 力的影响
        Yvr, Nvr   : v·r 速度乘积对 Y, N 力的影响
    
    推进器参数：
        b     : 0.30 m（双推进器之间的横向间距）
        T_max : 30.0 N（单个推进器最大推力）
    """
    # M 与 C 项的基础参数
    m: float = 23.8
    Iz: float = 1.76
    xg: float = 0.046 
    # 附加质量项（计算 M 和 C 矩阵使用）
    X_du: float = -2.0
    Y_dv: float = -10.0
    N_dr: float = -1.0
    Y_dr: float = 0.0
    N_dv: float = 0.0 
    # 阻尼项（D 矩阵）
    X_u : float = -0.72253
    X_uu : float = -1.32742
    Y_v : float = -0.88965
    Y_vv : float = -36.47287
    N_r : float = -1.900
    N_rr : float = -0.750
    Y_r : float = -7.250
    Y_rr : float = -3.450
    N_v : float = 0.03130
    N_vv : float = 3.95645
    # 交叉阻尼项（速度乘积项，描述横荡和偏航的耦合效应）
    Yrv: float = -0.805
    Nrv: float = 0.130
    Yvr: float = -0.845
    Nvr: float = 0.080
    # 推进器参数
    du: float = 25.0      # 纵向推力分配增益
    dv: float = 40.0      # 横向推力分配增益
    dr: float = 10.0      # 偏航推力分配增益
    b: float = 0.30       # 推进器间距（m）
    T_max: float = 30.0   # 最大推力（N）
    # 质量矩阵元素由 __post_init__ 统一计算。
    # 对 CS2 常见符号约定，X_du/Y_dv/N_dr 是负的水动力导数，
    # 因此有效惯量使用 m - X_du、Iz - N_dr。
    m_11: float = field(init=False)
    m_22: float = field(init=False)
    m_23: float = field(init=False)
    m_32: float = field(init=False)
    m_33: float = field(init=False)

    def __post_init__(self):
        """根据当前参数计算质量矩阵元素，保证自定义参数时不会失配。"""

        self.m_11 = self.m - self.X_du
        self.m_22 = self.m - self.Y_dv
        self.m_23 = self.m * self.xg - self.Y_dr
        self.m_32 = self.m * self.xg - self.N_dv
        self.m_33 = self.Iz - self.N_dr


class USV3DOF(_USVStepMixin):
    """
    基于 CS2 参数的精确 3DOF USV 模型。
    
    与 SimpleUSV3DOF 的区别：
        1. 使用 CS2 实船标定参数（更精确）
        2. 阻尼模型包含横向-偏航耦合项（Y_r·r, N_v·v 等）
        3. 数值求解线性方程组（np.linalg.solve），比矩阵求逆更稳定
    
    注意：
        USV3DOF 的 derivatives 方法中，阻尼力符号为正（+ tau_d），
        这是因为 CS2 参数中阻尼系数本身已为负值，
        代入公式后相当于阻力项。
        与 SimpleUSV3DOF 的 - tau_d 不同，请注意符号约定。
    
    参数:
        p : CS2Params 数据类实例，包含所有物理参数
    """
    def __init__(self, p: CS2Params | None = None):
        # 不使用 CS2Params() 作为默认实参，避免多个模型实例共享同一个可变对象。
        self.p = CS2Params() if p is None else p
        p = self.p
        # 构造质量矩阵（使用预计算的 m_ij 元素）
        self.M = np.array([
            [p.m_11, 0.0, 0.0], 
            [0, p.m_22, p.m_23], 
            [0, p.m_32, p.m_33]], dtype=float)

    def J(self, psi):
        """坐标变换矩阵（同 SimpleUSV3DOF.J）"""
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [c, -s, 0], 
            [s, c, 0], 
            [0, 0, 1.0]
            ], dtype=float)
    
    def C(self, nu):
        """
        科里奥利向心力矩阵（同 SimpleUSV3DOF.C，使用 CS2 质量矩阵元素）
        """
        u, v, r = nu 
        c_13 = -(self.p.m_22*v + self.p.m_23*r)
        c_23 = self.p.m_11 * u 
        return np.array([
            [0.0, 0.0, c_13], 
            [0.0, 0.0, c_23], 
            [-c_13, -c_23, 0.0]], dtype=float)

    def tau_d(self, nu):
        """
        计算 CS2 精确阻尼力（包含耦合项）。
        
        与 SimpleUSV3DOF 的区别：
            - 横向通道包含偏航速度耦合项：Y_r·r + Y_rr·|r|·r
            - 偏航通道包含横向速度耦合项：N_v·v + N_vv·|v|·v
        
        注意：
            CS2 参数中 X_u, Y_v, N_r 等均为负值，
            所以 tau_d 的结果也为负，表示与运动方向相反的阻力。
            在 derivatives 中使用 + tau_d（加号），因为负值本身已表示阻力方向。
        """
        p = self.p 
        u, v, r = nu 
        # 纵向阻尼（二次非线性）
        tau_1 = p.X_u * u + p.X_uu * abs(u) * u 
        # 横向阻尼（含偏航耦合项和速度乘积交叉耦合项）
        tau_2 = (p.Y_v*v + p.Y_r*r + p.Y_vv*abs(v)*v + p.Y_rr*abs(r)*r
                 + p.Yrv*abs(r)*v + p.Yvr*abs(v)*r)
        # 偏航阻尼（含横向耦合项和速度乘积交叉耦合项）
        tau_3 = (p.N_v*v + p.N_r*r + p.N_vv*abs(v)*v + p.N_rr*abs(r)*r
                 + p.Nrv*abs(r)*v + p.Nvr*abs(v)*r)
        tau_d = np.array([tau_1, tau_2, tau_3], dtype=float)
        return tau_d  
        
    def derivatives(self, eta, nu, tau):
        """
        计算 CS2 模型状态导数。
        
        动力学方程（注意阻尼项符号）：
            M · ν̇ = τ - C(ν)·ν + τ_d(ν)
            
            注：这里是 + tau_d，因为 CS2 参数中阻尼系数为负，
            tau_d 的值本身为负数（阻力方向），所以加法等效于减法。
        
        参数:
            eta : 当前姿态 [x, y, ψ]
            nu  : 当前速度 [u, v, r]
            tau : 控制力 [τ_u, τ_v, τ_r]
        
        返回:
            eta_dot : 姿态导数
            nu_dot  : 速度导数
        """
        # 强制转为 float64，防止 dtype=object 导致矩阵运算报错
        nu = np.asarray(nu, dtype=float).reshape(3, )
        tau = np.asarray(tau, dtype=float).reshape(3, )
        x, y, psi = eta

        # 运动学
        eta_dot = self.J(psi) @ nu

        # 动力学：求解线性方程组 M · ν̇ = rhs
        # 比矩阵求逆（M_inv @ rhs）数值更稳定
        rhs = tau - self.C(nu) @ nu + self.tau_d(nu)
        nu_dot = np.linalg.solve(self.M, rhs) 

        return eta_dot, nu_dot   

    # step_euler / step_rk4 / step 由 _USVStepMixin 提供，无需在此重复定义。
