"""
LOS（Line-of-Sight，视线导引）导引律模块

LOS 导引律是船舶自动路径跟踪的核心算法，其思想是：
    在当前位置前方一定距离（前视距离 Δ）处取一个"前视点"，
    将船的期望航向指向该前视点，从而引导船跟踪折线路径。

期望航向计算公式：
    ψ_d = α_k - arctan(e_ct / Δ)
    
    其中：
        α_k   = 当前路径段的方位角（arctan2(dy, dx)）
        e_ct  = 横向误差（船到路径的垂直距离，左正右负）
        Δ     = 前视距离（固定值或自适应值）

本模块提供三种导引律：
    1. LOSGuidance          — 固定前视距离 LOS（Fixed LOS）
    2. AdaptiveLOSGuidance  — 自适应前视距离 LOS（根据误差/速度/曲率动态调整）
    3. ILOSGuidance         — 积分 LOS（Integral LOS，ILOS）
                              通过对横向误差积分补偿稳态误差（如海流引起的定常漂移）

参考文献：
    [1] Fossen, T.I. (2011). Handbook of Marine Craft Hydrodynamics and Motion Control.
    [2] Borhaug, E., Pavlov, A., & Pettersen, K.Y. (2008).
        Integral LOS control for path following of underactuated marine surface vessels
        in the presence of constant ocean currents. IEEE CDC.
    [3] Fossen, T.I., Pettersen, K.Y., & Galeazzi, R. (2015).
        Line-of-sight path following for dubins paths with adaptive sideslip compensation.
        IEEE Transactions on Control Systems Technology.
"""

import numpy as np
from baseline.math_utils import wrap_to_pi

class BaseLOSGuidance:
    """
    LOS 导引律基类，提供路径管理和坐标变换的公共功能。
    
    不直接使用，由 LOSGuidance 和 AdaptiveLOSGuidance 组合使用。
    
    参数:
        switch_radius : 航点切换半径（米）。
                        当船与当前目标航点的距离小于此值时，切换到下一个路径段。
    """

    def __init__(self, switch_radius=3.0):
        self.switch_radius = switch_radius
        self.wp_idx = 0  # 当前激活路径段的起点索引

    def reset(self):
        """重置航点索引，在新仿真开始时调用"""
        self.wp_idx = 0

    @staticmethod
    def _distance(p1, p2):
        """计算两点之间的欧氏距离"""
        return np.linalg.norm(np.asarray(p1) - np.asarray(p2))
    
    def _advance_waypoint(self, x, y, waypoints):
        """
        检查是否需要切换到下一个路径段。
        
        逻辑包含两个触发条件：
            1. 距离触发：船进入目标航点 switch_radius 半径内；
            2. 投影触发：船虽然没有进入半径，但沿当前路径段方向已经越过航点。

        第二个条件很重要。实际仿真中，船可能因海流或转弯惯性从航点旁边
        擦过去，如果只按距离切换，会永远停留在旧航段，导致 L 形路径无法转弯。
        
        参数:
            x, y      : 船的当前位置
            waypoints : 所有航点列表，shape (N, 2)
        """
        if len(waypoints) < 2:
            raise ValueError("waypoints 至少需要两个点")

        pos = np.array([x, y], dtype=float)

        # 使用 while 可处理一步内跨过很短航段的情况。
        while self.wp_idx < len(waypoints) - 2:
            p0 = np.asarray(waypoints[self.wp_idx], dtype=float)
            p1 = np.asarray(waypoints[self.wp_idx + 1], dtype=float)
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))

            if seg_len < 1e-9:
                self.wp_idx += 1
                continue

            unit = seg / seg_len
            rel = pos - p0
            along = float(np.dot(rel, unit))
            dist_to_target = self._distance(pos, p1)

            reached_by_distance = dist_to_target < self.switch_radius
            # Projection-based switching is a fail-safe for cases where the
            # vessel passes a waypoint without entering its radius.  The old
            # implementation used `seg_len - switch_radius`, which can skip
            # very short dense waypoints immediately.  A segment-relative
            # margin keeps smooth paths switchable without making L corners
            # unrealistically early.
            advance_margin = min(self.switch_radius, 0.25 * seg_len)
            reached_by_projection = along >= seg_len - advance_margin

            if reached_by_distance or reached_by_projection:
                self.wp_idx += 1
            else:
                break
            
    def _get_active_segment(self, waypoints):
        """
        获取当前激活路径段的两个端点。
        
        返回:
            p0 : 路径段起点坐标 [x, y]
            p1 : 路径段终点坐标 [x, y]
        """
        if len(waypoints) < 2:
            raise ValueError("waypoints 至少需要两个点")

        # 防止索引越界
        i = min(self.wp_idx, len(waypoints) - 2)
        p0 = np.asarray(waypoints[i], dtype=float)
        p1 = np.asarray(waypoints[i + 1], dtype=float)
        return p0, p1
    
    def _compute_path_frame(self, position, waypoints):
        """
        将船的全局坐标变换到路径坐标系（Frenet 坐标系）。
        
        路径坐标系定义：
            - 纵轴（s 轴）：沿路径段方向（向前为正）
            - 横轴（e 轴）：垂直于路径段（右侧为正，NED坐标系下右舷/东侧为正）
        
        参数:
            position  : 船的当前位置 [x, y]
            waypoints : 所有航点列表
        
        返回:
            p0         : 当前路径段起点
            p1         : 当前路径段终点
            path_angle : 路径段方位角 α_k（弧度）
            s_along    : 沿轨迹误差（船在路径方向上的投影距离）
            e_ct       : 横向误差（船到路径的垂直距离）
                         正值 = 船在路径右侧（NED坐标系东侧），负值 = 船在路径左侧（西侧）
        """
        x, y = position
        # 先检查是否需要切换路径段
        self._advance_waypoint(x, y, waypoints)
        p0, p1 = self._get_active_segment(waypoints)

        # 计算路径段方向角 α_k
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        path_angle = np.arctan2(dy, dx)

        # 构造旋转矩阵（将全局坐标系旋转到路径坐标系）
        # rot @ rel = [s_along, e_ct]
        rot = np.array([
            [np.cos(path_angle), np.sin(path_angle)],
            [-np.sin(path_angle), np.cos(path_angle)],
        ])

        # 船相对于路径段起点的向量
        rel = np.array([x - p0[0], y - p0[1]], dtype=float)
        # 变换到路径坐标系
        s_along, e_ct = rot @ rel
        return p0, p1, path_angle, s_along, e_ct

    def _estimate_curvature(self, waypoints):
        """
        估计当前路径段附近的路径曲率 κ（单位：1/m）。
        
        方法：通过前后两个路径段的航向变化率近似曲率。
            κ ≈ |Δψ| / segment_length
        
        参数:
            waypoints : 所有航点列表
        
        返回:
            曲率值 κ（非负标量），直线段返回 0.0
        """
        n = len(waypoints)
        i = min(self.wp_idx, n - 2)

        # 边界情况：第一段或最后一段无法估计曲率（只有一段）
        if i <= 0 or i >= n - 2:
            return 0.0

        p_prev = np.asarray(waypoints[i - 1], dtype=float)
        p_curr = np.asarray(waypoints[i], dtype=float)
        p_next = np.asarray(waypoints[i + 1], dtype=float)

        # 前一段的方位角
        heading_prev = np.arctan2(p_curr[1] - p_prev[1], p_curr[0] - p_prev[0])
        # 后一段的方位角
        heading_next = np.arctan2(p_next[1] - p_curr[1], p_next[0] - p_curr[0])
        # 航向变化量（归一化到 [-π, π)）
        delta_heading = wrap_to_pi(heading_next - heading_prev)
        # 用于归一化的段长度
        segment_length = max(self._distance(p_curr, p_next), 1e-6)

        return float(abs(delta_heading) / segment_length)


class LOSGuidance:
    """
    固定前视距离 LOS 导引律。
    
    前视距离 Δ 为固定常数，适用于路径曲率变化不大的场景。
    
    期望航向：
        ψ_d = α_k - arctan(e_ct / Δ)
    
    特点：
        - 实现简单，计算高效
        - 前视距离过小：跟踪精度高，但可能导致振荡
        - 前视距离过大：跟踪平滑，但横向误差较大，弯道处超调严重
    
    参数:
        lookahead     : 固定前视距离 Δ（米），默认 8.0m
        switch_radius : 航点切换半径（米），默认 3.0m
    """
    def __init__(self, lookahead=8.0, switch_radius=3.0):
        self.lookahead = lookahead  # 固定前视距离 Δ
        self._base = BaseLOSGuidance(switch_radius=switch_radius)

    @property
    def wp_idx(self):
        """当前航点索引（只读属性，代理到 _base）"""
        return self._base.wp_idx

    def reset(self):
        """重置内部状态"""
        self._base.reset()

    def __call__(self, position, waypoints, eta=None, nu=None, dt=None, **kwargs):
        """
        计算 LOS 期望航向。
        
        参数:
            position  : 船的当前位置 [x, y]
            waypoints : 路径点列表，shape (N, 2)
            eta       : 当前姿态状态 [x, y, ψ]（本类不使用，保持接口统一）
            nu        : 当前速度状态 [u, v, r]（本类不使用，保持接口统一）
            dt        : 时间步长（本类不使用，保持与 ILOSGuidance 接口统一）
        
        返回:
            psi_d : 期望航向角（弧度，归一化到 [-π, π)）
            info  : 调试信息字典，包含横向误差、前视距离、路径角等
        """
        x, y = position
        waypoints = np.asarray(waypoints, dtype=float)

        if len(waypoints) < 2:
            raise ValueError("waypoints must contain at least 2 points")

        # 计算路径坐标系信息
        p0, p1, path_angle, s_along, e_ct = self._base._compute_path_frame(
            position=[x, y],
            waypoints=waypoints,
        )

        # LOS 期望航向公式：ψ_d = α_k - arctan(e_ct / Δ)
        # 当 e_ct > 0（船在左侧），ψ_d 减小（向右转）
        # 当 e_ct < 0（船在右侧），ψ_d 增大（向左转）
        psi_d = path_angle - np.arctan2(e_ct, self.lookahead)
        psi_d = wrap_to_pi(psi_d)  # 归一化到 [-π, π)

        info = {
            "psi_d": psi_d,                         # 期望航向（rad）
            "cross_track_error": e_ct,               # 横向误差（m）
            "along_track_error": s_along,            # 纵向误差（m）
            "path_angle": path_angle,                # 路径段方位角（rad）
            "wp_idx": self.wp_idx,                   # 当前路径段索引
            "segment_start": p0,                     # 当前段起点坐标
            "segment_end": p1,                       # 当前段终点坐标
            "lookahead": float(self.lookahead),      # 实际使用的前视距离（m）
            "curvature": 0.0,                        # 曲率（固定LOS不计算，保持接口一致）
        }
        return psi_d, info



class AdaptiveLOSGuidance:
    """
    自适应前视距离 LOS 导引律。
    
    在固定 LOS 基础上，根据横向误差、船速和路径曲率动态调整前视距离，
    平衡跟踪精度与控制平滑性：
    
    前视距离自适应公式：
        Δ = Δ_base + k_e * |e_ct| + k_u * |u| - k_kappa * |κ|
        Δ = clip(Δ, Δ_min, Δ_max)
    
    物理意义：
        - 横向误差越大 → Δ 增大 → 控制更平滑，避免过度转向
        - 速度越大     → Δ 增大 → 前视更远，适应高速情况
        - 曲率越大     → Δ 减小 → 弯道收紧前视，提高弯道跟踪精度
    
    参数:
        lookahead_base : 基础前视距离 Δ_base（米），默认 10.0m
        lookahead_min  : 前视距离下限 Δ_min（米），默认 4.0m
        lookahead_max  : 前视距离上限 Δ_max（米），默认 20.0m
        k_e            : 横向误差增益，默认 0.8
        k_u            : 速度增益，默认 0.0（不使用速度调整）
        k_kappa        : 曲率衰减增益，默认 0.0（不使用曲率调整）
        switch_radius  : 航点切换半径（米），默认 3.0m
    """
    def __init__(
        self,
        lookahead_base=10.0,
        lookahead_min=4.0,
        lookahead_max=20.0,
        k_e=0.8,
        k_u=0.0,
        k_kappa=0.0,
        switch_radius=3.0,
    ):
        self.lookahead_base = float(lookahead_base)  # 基础前视距离
        self.lookahead_min = float(lookahead_min)    # 前视距离下限
        self.lookahead_max = float(lookahead_max)    # 前视距离上限
        self.k_e = float(k_e)                        # 横向误差调整增益
        self.k_u = float(k_u)                        # 速度调整增益
        self.k_kappa = float(k_kappa)                # 曲率衰减增益
        self._base = BaseLOSGuidance(switch_radius=switch_radius)

    @property
    def wp_idx(self):
        """当前航点索引（只读属性，代理到 _base）"""
        return self._base.wp_idx

    def reset(self):
        """重置内部状态"""
        self._base.reset()


    def _compute_adaptive_lookahead(self, e_ct, nu, curvature):
        """
        根据横向误差、速度和曲率计算自适应前视距离。
        
        公式：
            Δ = Δ_base + k_e * |e_ct| + k_u * |u| - k_kappa * |κ|
            Δ = clip(Δ, Δ_min, Δ_max)
        
        参数:
            e_ct      : 横向误差（米）
            nu        : 速度向量 [u, v, r]，取 u（纵荡速度）
            curvature : 路径曲率 κ（1/m）
        
        返回:
            自适应前视距离 Δ（米）
        """
        # 提取纵荡速度 u
        surge_speed = 0.0
        if nu is not None:
            nu = np.asarray(nu, dtype=float).reshape(-1)
            if nu.size > 0:
                surge_speed = abs(float(nu[0]))

        # 自适应公式
        lookahead = (
            self.lookahead_base
            + self.k_e * abs(float(e_ct))          # 误差越大，前视越远
            + self.k_u * surge_speed               # 速度越快，前视越远
            - self.k_kappa * abs(float(curvature)) # 曲率越大，前视越近
        )
        # 限幅，防止极端情况
        return float(np.clip(lookahead, self.lookahead_min, self.lookahead_max))

    def __call__(self, position, waypoints, eta=None, nu=None, dt=None, **kwargs):
        """
        计算自适应 LOS 期望航向。
        
        参数:
            position  : 船的当前位置 [x, y]
            waypoints : 路径点列表，shape (N, 2)
            eta       : 当前姿态 [x, y, ψ]（此类不使用）
            nu        : 当前速度 [u, v, r]（用于计算自适应前视距离）
            dt        : 时间步长（本类不使用，保持与 ILOSGuidance 接口统一）
        
        返回:
            psi_d : 期望航向角（弧度，归一化到 [-π, π)）
            info  : 调试信息字典，包含自适应前视距离、曲率等
        """
        waypoints = np.asarray(waypoints, dtype=float)

        if len(waypoints) < 2:
            raise ValueError("waypoints must contain at least 2 points")

        # 计算路径坐标系信息
        p0, p1, path_angle, s_along, e_ct = self._base._compute_path_frame(
            position=position,
            waypoints=waypoints,
        )
        # 估计当前路径曲率
        curvature = self._base._estimate_curvature(waypoints)
        # 计算自适应前视距离
        lookahead = self._compute_adaptive_lookahead(
            e_ct=e_ct,
            nu=nu,
            curvature=curvature,
        )

        # LOS 期望航向
        psi_d = path_angle - np.arctan2(e_ct, lookahead)
        psi_d = wrap_to_pi(psi_d)

        info = {
            "psi_d": psi_d,
            "cross_track_error": float(e_ct),        # 横向误差（m）
            "along_track_error": float(s_along),     # 纵向误差（m）
            "path_angle": float(path_angle),         # 路径段方位角（rad）
            "wp_idx": self.wp_idx,                   # 当前路径段索引
            "segment_start": p0,                     # 当前段起点
            "segment_end": p1,                       # 当前段终点
            "lookahead": lookahead,                  # 实际自适应前视距离（m）
            "curvature": float(curvature),           # 估计曲率（1/m）
        }
        return psi_d, info

class ILOSGuidance:
    """
    积分 LOS 导引律（Integral Line-of-Sight，ILOS）。

    标准 LOS 在定常海流或恒定侧风存在时会产生非零稳态横向误差。
    ILOS 通过引入横向误差的积分项 σ，将期望航向修正为：

        ψ_d = α_k - arctan((e_ct + k_i · σ) / Δ)

    积分动态方程（Fossen et al. 2015）：
        σ̇ = (Δ · e_ct) / (Δ² + (e_ct + k_i · σ)²)

    Lyapunov 稳定性：
        取 V = (1/2)·e_ct² + (1/(2·k_i))·(k_i·σ + β)²
        在 Δ > 0、k_i > 0 条件下，V̇ ≤ 0，保证全局一致渐近稳定（UGAS）。

    参数：
        lookahead     : 固定前视距离 Δ（m），默认 8.0m
        k_i           : 积分增益，默认 0.05
        sigma_limit   : 积分状态限幅（防饱和），默认 2.0m
        switch_radius : 航点切换半径（m），默认 3.0m

    参考：Borhaug et al. (2008) CDC；Fossen et al. (2015) IEEE TCST。
    """

    def __init__(
        self,
        lookahead: float = 8.0,
        k_i: float = 0.05,
        sigma_limit: float = 2.0,
        switch_radius: float = 3.0,
    ):
        self.lookahead = float(lookahead)
        self.k_i = float(k_i)
        self.sigma_limit = float(sigma_limit)
        self._base = BaseLOSGuidance(switch_radius=switch_radius)
        self.sigma = 0.0

    @property
    def wp_idx(self):
        return self._base.wp_idx

    def reset(self):
        """重置内部状态（仿真开始前调用）"""
        self._base.reset()
        self.sigma = 0.0


    def _update_sigma(self, e_ct: float, dt: float) -> None:
        """
        更新积分状态 σ（前向 Euler 离散化）。

        σ̇ = (Δ · e_ct) / (Δ² + (e_ct + k_i·σ)²)

        分母随误差增大而增大，内置软限幅效果。
        """
        Delta = self.lookahead
        denom = Delta ** 2 + (e_ct + self.k_i * self.sigma) ** 2
        if abs(denom) < 1e-12:
            return
        sigma_dot = (Delta * e_ct) / denom
        self.sigma += sigma_dot * dt
        self.sigma = float(np.clip(self.sigma, -self.sigma_limit, self.sigma_limit))


    def __call__(self, position, waypoints, eta=None, nu=None, dt: float = 0.05, **kwargs):
        """
        计算 ILOS 期望航向并更新积分状态。

        ILOS 期望航向：
            ψ_d = α_k - arctan((e_ct + k_i·σ) / Δ)

        参数：
            position  : 当前位置 [x, y]
            waypoints : 路径点数组，shape (N, 2)
            eta       : 当前姿态（不使用，保持接口统一）
            nu        : 当前速度（不使用，保持接口统一）
            dt        : 时间步长（s），用于更新积分状态

        返回：
            psi_d : 期望航向角（rad，归一化到 [-π, π)）
            info  : 调试信息字典
        """
        waypoints = np.asarray(waypoints, dtype=float)

        if len(waypoints) < 2:
            raise ValueError("waypoints must contain at least 2 points")

        old_wp_idx = self.wp_idx
        p0, p1, path_angle, s_along, e_ct = self._base._compute_path_frame(
            position=position,
            waypoints=waypoints,
        )
        if self.wp_idx != old_wp_idx:
            # ILOS 的 sigma 是“当前直线路径段”的横向误差积分。路径段旋转后，
            # 旧 sigma 的方向含义已经改变，继续沿用会把上一段的海流补偿错误地
            # 注入新航段，尤其会污染 L 形转角后的跟踪结果。
            self.sigma = 0.0

        self._update_sigma(float(e_ct), dt)

        compensated = e_ct + self.k_i * self.sigma
        psi_d = path_angle - np.arctan2(compensated, self.lookahead)
        psi_d = wrap_to_pi(psi_d)

        info = {
            "psi_d": psi_d,
            "cross_track_error": float(e_ct),
            "along_track_error": float(s_along),
            "path_angle": float(path_angle),
            "wp_idx": self.wp_idx,
            "segment_start": p0,
            "segment_end": p1,
            "lookahead": float(self.lookahead),
            "curvature": 0.0,
            "sigma": float(self.sigma),
            "compensated_error": float(compensated),
        }
        return psi_d, info
