"""
实验 4.6（v2，统一前视距离版）：消融实验 — 各组件贡献逐步分析

与 exp_05 的区别：
  - 消融起点改用 los_pid_short（Δ=4 m）和 anti_windup_short（Δ=4 m + AW），
    使整个消融链与 Δ=4m 主论文保持一致，避免第一步同时包含前视距离变化和组件变化。

消融方法序列（递进关系，均使用 Δ=4 m）：
  1. ILOS (Δ=4m)     — 基线：短前视 ILOS 制导 + 标准 PID，无任何扩展
  2. AW  (Δ=4m)      — 在 PID 中加入反算抗积分卷绕（Anti-Windup），其余不变
  3. DS              — 加入动态航向整形层，消除饱和，但无速度调度
  4. SHCS            — 完整方法：动态整形 + 推进器约束感知耦合调速（本文方法）

场景：L 形路径 + 定常海流

输出结构：
  results/baseline/05b_ablation_v2/
    {method}_log.npz / {method}_summary.json   — 各方法原始数据
    ablation_summary.csv                        — 完整消融表
    fig6_composite.png                          — 2×3 主组图
    subfigs/
      fig6a_traj.png      — 2-D 轨迹叠图
      fig6b_heading.png   — 航向误差时间历程
      fig6c_cte.png       — 横向误差时间历程
      fig6d_taur.png      — 偏航力矩（含饱和阴影）
      fig6e_surge.png     — 纵荡速度
      fig6f_bar.png       — 4指标分组条形图（组件贡献汇总）
"""

import sys
import os
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from baseline import get_config, get_path
from scripts.experiment_utils import (
    ABLATION_BAR_EDGES, ABLATION_BAR_FACES, PAPER_SUCCESS,
    run_trial, save_result, save_summaries_csv,
    apply_plot_style, get_method_style, print_metrics_table,
    relative_improvement, RESULTS_ROOT, get_initial_state,
    save_fig, draw_final_position, plot_waypoints,
    shade_saturation_windows, heu_figsize, apply_cjk_text_fonts,
)

# ─────────────────────────────────────────────────────────────────────────────
# 实验配置常量
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR = RESULTS_ROOT / "05b_ablation_v2"

# 消融方法序列（均使用 Δ=4m，与 v2 论文主实验保持一致）
METHOD_NAMES    = ["los_pid_short", "anti_windup_short", "dynamic_shaper", "shcs"]
PATH_NAME       = "l_shape"
DIST_NAME       = "steady_current"
SEED            = 1
EXPERIMENT_NAME = "ablation_v2"

# 消融表导出的指标字段
TABLE_FIELDS = [
    ("cross_track_rms",           "CTE_RMS_m",      4),
    ("turn_cte_rms",              "TurnCTE_RMS_m",  4),
    ("heading_los_error_rms",     "HdgLOS_RMS_rad", 5),  # 公平航向指标（psi_d - psi）
    ("heading_error_rms",         "HdgRef_RMS_rad", 5),  # PID参考跟踪（仅供参考）
    ("control_energy_tau_r_cmd",  "YawEnergy",      2),
    ("total_control_energy",      "TotalEnergy",    2),
    ("sat_time_raw",              "SatTime_s",      2),
    ("sat_ratio_raw",             "SatRatio",       4),
    ("sat_peak_raw",              "SatPeak_Nm",     2),
    ("speed_reduction_max_pct",   "MaxSpeedRed_pct", 2),
    ("reach_time",                "ReachTime_s",    2),
]

# 单个面板图尺寸（用于子图）
_HALF = heu_figsize("small", 0.72)
_FULL = heu_figsize("small", 0.95)

# x 轴短标签（将 config label 映射为图中简短标签）
_SHORT_LABELS = {
    "ILOS":              "ILOS",
    "ILOS-PID(Δ=4m)":   "ILOS",
    "AW":                "AW",
    "AW(Δ=4m)":         "AW",
    "DS":                "DS",
    "SHCS":              "SHCS",
}


# ─────────────────────────────────────────────────────────────────────────────
# 底层绘图辅助函数（接受 ax 参数，可被组图和子图复用）
# ─────────────────────────────────────────────────────────────────────────────

def _draw_traj(ax, results, labels, names, waypoints, cfg):
    """(a) 2-D 轨迹叠图。

    多条方法轨迹叠加在同一图中，显示各方法在 L 形转弯处的路径差异。
    SHCS 方法轨迹最贴近参考路径，说明完整方法的优越性。
    包含图例（组图中仅轨迹图显示图例）。
    """
    wps = np.asarray(waypoints)
    goal_tol = float(cfg["simulation"].get("goal_tolerance", 3.0))
    plot_waypoints(ax, wps, goal_tol=goal_tol)
    for idx, (res, lbl, name) in enumerate(zip(results, labels, names)):
        c, ls = get_method_style(name, idx)
        lw = 2.0 if name == "shcs" else 1.4
        ax.plot(res["log"]["x"], res["log"]["y"],
                color=c, lw=lw, ls=ls, label=lbl, zorder=4)
        draw_final_position(ax, res["log"], c)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend(loc="upper left", ncol=2)


def _draw_heading(ax, results, names):
    """(b) 几何 LOS 航向误差时间历程（单位：°）。

    使用 e_psi_los = psi_d - psi，对所有方法一致。
    加入抗积分卷绕（步骤2）后，饱和区积分累积减少，振荡降低。
    加入整形器（步骤3）后，指令本身平滑，收敛更快。
    """
    for idx, (res, name) in enumerate(zip(results, names)):
        c, ls = get_method_style(name, idx)
        key = "e_psi_los" if "e_psi_los" in res["log"] else "e_psi"
        ax.plot(res["log"]["t"], np.degrees(res["log"][key]),
                color=c, lw=1.4, ls=ls)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.text(0.98, 0.97, "e = ψ_d − ψ",
            transform=ax.transAxes, fontsize=5.5, ha="right", va="top", color="gray")
    ax.set_xlabel("t / s")
    ax.set_ylabel("LOS Heading Error / (°)")
    ax.grid(True)


def _draw_cte(ax, results, names):
    """(c) 横向误差（CTE）时间历程（单位：m）。

    转弯段 CTE 峰值体现各方法差异：
    - 基线/AW：PID 饱和导致无法快速跟踪偏航，CTE 峰值最大
    - 动态整形：整形消除饱和，CTE 峰值显著下降
    - SHCS：速度调度进一步降低横流影响，CTE 最小
    """
    for idx, (res, name) in enumerate(zip(results, names)):
        c, ls = get_method_style(name, idx)
        ax.plot(res["log"]["t"], res["log"]["e_ct"],
                color=c, lw=1.4, ls=ls)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("t / s")
    ax.set_ylabel("Cross-Track Error / m")
    ax.grid(True)


def _draw_taur(ax, results, names, tau_r_lim):
    """(d) 偏航力矩时间历程（含饱和阴影）。

    未整形方法（ILOS / AW）在转弯时力矩饱和，用各自颜色阴影标注。
    整形类方法（DS/SHCS-Simple/SHCS）不会触发饱和，故无阴影。
    """
    unshaped = {"los_pid_short", "anti_windup_short"}
    for res, name in zip(results, names):
        if name in unshaped:
            c, _ = get_method_style(name, 0)
            shade_saturation_windows(
                ax,
                np.asarray(res["log"]["t"]),
                np.asarray(res["log"]["tau_r_cmd_raw"]),
                tau_r_lim, color=c, alpha=0.12,
            )
    for idx, (res, name) in enumerate(zip(results, names)):
        c, ls = get_method_style(name, idx)
        ax.plot(res["log"]["t"], res["log"]["tau_r_cmd_raw"],
                color=c, lw=1.4, ls=ls)
    ax.axhline( tau_r_lim, color="#B24A3B", lw=0.9, ls="--", alpha=0.82)
    ax.axhline(-tau_r_lim, color="#B24A3B", lw=0.9, ls="--", alpha=0.82)
    ax.text(0.98, 0.97, f"±{tau_r_lim:.0f} N·m",
            transform=ax.transAxes, fontsize=5.5, color="#8A342B",
            ha="right", va="top")
    ax.set_xlabel("t / s")
    ax.set_ylabel("Yaw Torque / (N·m)")
    ax.grid(True)


def _draw_surge(ax, results, names):
    """(e) 纵荡速度时间历程。

    实线：实际速度；点线：期望速度指令。
    SHCS/SHCS-Simple 在转弯段的期望速度明显下降（速度调度生效），
    为偏航控制腾出推力余量。
    """
    for idx, (res, name) in enumerate(zip(results, names)):
        c, ls = get_method_style(name, idx)
        ax.plot(res["log"]["t"], res["log"]["u"],
                color=c, lw=1.4, ls=ls)
        ax.plot(res["log"]["t"], res["log"]["u_d"],
                color=c, lw=0.7, ls=":", alpha=0.5)
    ax.text(0.98, 0.05, "Solid: actual   Dotted: desired",
            transform=ax.transAxes, fontsize=5.5, ha="right", va="bottom", color="gray")
    ax.set_xlabel("t / s")
    ax.set_ylabel("Surge Speed / (m/s)")
    ax.grid(True)


def _draw_cte_bar(ax, results, labels, names):
    """(f) 横向误差 RMS 条形图（组图版，仅展示一个核心指标）。

    在 2×3 组图中嵌入单指标条形图，用于快速对比各消融步骤的最终性能。
    完整的 4 指标多面板条形图保存在 subfigs/fig6f_bar.png 中。
    """
    vals = [float(res["summary"].get("cross_track_rms", np.nan)) for res in results]
    x = np.arange(len(names))
    bar_colors = [ABLATION_BAR_FACES.get(n, "0.80") for n in names]
    bar_edges  = [ABLATION_BAR_EDGES.get(n, "#8B929E") for n in names]
    bar_lws    = [0.65 for n in names]
    ax.bar(x, vals, width=0.62, color=bar_colors, edgecolor=bar_edges,
           linewidth=bar_lws, zorder=3)
    finite = [v for v in vals if not np.isnan(v)]
    y_max = max(finite) if finite else 1.0
    if y_max <= 0:
        y_max = 1.0
    for xi, val in enumerate(vals):
        if not np.isnan(val):
            ax.text(xi, val + y_max * 0.03, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6.0,
                    color="#1F2933",
                    fontweight=("bold" if names[xi] == "shcs" else "normal"))
    ax.set_xticks(x)
    ax.set_xticklabels([_SHORT_LABELS.get(lb, lb) for lb in labels],
                       rotation=0, ha="center", fontsize=6.0)
    ax.set_ylabel("CTE RMS / m")
    ax.set_ylim(0, y_max * 1.40)
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# 各子图面板（调用底层 _draw_* 辅助，添加标题和图例后返回独立 Figure）
# ─────────────────────────────────────────────────────────────────────────────

def make_traj_panel(results, labels, names, waypoints, cfg) -> plt.Figure:
    """(a) 2-D 轨迹叠图（独立子图）"""
    apply_plot_style("panel")
    fig, ax = plt.subplots(figsize=_FULL)
    _draw_traj(ax, results, labels, names, waypoints, cfg)
    ax.set_title("(a) 二维轨迹", fontweight="bold", loc="left")
    return fig


def make_heading_panel(results, labels, names) -> plt.Figure:
    """(b) 航向误差时间历程（独立子图）"""
    apply_plot_style("panel")
    fig, ax = plt.subplots(figsize=_HALF)
    _draw_heading(ax, results, names)
    ax.set_title("(b) 航向误差", fontweight="bold", loc="left")
    # 在独立子图中额外添加图例，组图版不重复显示
    for idx, (lbl, name) in enumerate(zip(labels, names)):
        c, ls = get_method_style(name, idx)
        ax.plot([], [], color=c, lw=1.4, ls=ls, label=lbl)
    ax.legend(ncol=2)
    return fig


def make_cte_panel(results, labels, names) -> plt.Figure:
    """(c) 横向误差时间历程（独立子图）"""
    apply_plot_style("panel")
    fig, ax = plt.subplots(figsize=_HALF)
    _draw_cte(ax, results, names)
    ax.set_title("(c) 横向误差", fontweight="bold", loc="left")
    for idx, (lbl, name) in enumerate(zip(labels, names)):
        c, ls = get_method_style(name, idx)
        ax.plot([], [], color=c, lw=1.4, ls=ls, label=lbl)
    ax.legend(ncol=2)
    return fig


def make_taur_panel(results, labels, names, cfg) -> plt.Figure:
    """(d) 偏航力矩时间历程（独立子图，含饱和阴影）"""
    apply_plot_style("panel")
    fig, ax = plt.subplots(figsize=_HALF)
    tau_r_lim = cfg["actuator"]["T_max"] * cfg["actuator"]["b"]
    _draw_taur(ax, results, names, tau_r_lim)
    ax.set_title("(d) 偏航力矩（阴影为饱和）", fontweight="bold", loc="left")
    for idx, (lbl, name) in enumerate(zip(labels, names)):
        c, ls = get_method_style(name, idx)
        ax.plot([], [], color=c, lw=1.4, ls=ls, label=lbl)
    ax.plot([], [], color="#B24A3B", lw=0.9, ls="--", alpha=0.82,
            label=f"±{tau_r_lim:.0f} N·m")
    ax.legend(ncol=2)
    return fig


def make_surge_panel(results, labels, names) -> plt.Figure:
    """(e) 纵荡速度时间历程（独立子图）"""
    apply_plot_style("panel")
    fig, ax = plt.subplots(figsize=_HALF)
    _draw_surge(ax, results, names)
    ax.set_title("(e) 纵荡速度（实线实际，点线期望）", fontweight="bold", loc="left")
    for idx, (lbl, name) in enumerate(zip(labels, names)):
        c, ls = get_method_style(name, idx)
        ax.plot([], [], color=c, lw=1.4, ls=ls, label=lbl)
    ax.legend(ncol=2)
    return fig


def make_bar_panel(results, labels, names) -> plt.Figure:
    """消融组件贡献条形面板（4 指标 × 2×2 布局，单栏版）。

    各方法按消融步骤从左到右递进；SHCS 砖红色粗边凸显；
    纵轴标注指标名称与单位；子图标题置于下方（与 exp_01 风格一致）。
    """
    apply_plot_style("bar")
    # 单栏小图（7 cm），覆盖全局字号避免文字拥挤重叠
    plt.rcParams.update({
        "font.size":       6.0,
        "axes.titlesize":  6.5,
        "axes.labelsize":  6.0,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
    })

    # (field, 底部标题, 纵轴标签, 数值格式) heading_error_rms HdgRef_RMS_rad
    metrics = [
        ("cross_track_rms",          "(a) 横向误差RMS",    "CTE-RMS / m",    "{:.3f}"),
        ("heading_error_rms",        "(b) 航向误差RMS",    "Ref-RMS / rad", "{:.4f}"),
        ("control_energy_tau_r_cmd", "(c) 偏航控制能耗",    "Yaw-Energy / $(N^2\cdot m^2\cdot s)$",     "{:.0f}"),
        ("sat_time_raw",             "(d) 饱和作用时间",    "Sat-Time / s",  "{:.2f}"),
    ]

    x = np.arange(len(names))
    # x 轴显示标签（与 cfg["methods"][m]["label"] 对应，保持单行）
    _SINGLE_LINE = {
        "ILOS":              "ILOS",
        "ILOS-PID(Δ=4m)":   "ILOS",
        "AW":                "AW",
        "AW(Δ=4m)":         "AW",
        "DS":                "DS",
        "SHCS":              "SHCS",
    }

    fig, axes = plt.subplots(2, 2, figsize=heu_figsize("small", 1.15),
                             constrained_layout=True)
    axes = axes.ravel()

    for ax, (field, caption, ylabel, fmt) in zip(axes, metrics):
        is_sat = (field == "sat_time_raw")

        vals: list[float] = []
        for res in results:
            v = res["summary"].get(field, float("nan"))
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(float("nan"))

        finite = [v for v in vals if not np.isnan(v)]
        y_max  = max(finite) if finite else 1.0
        if y_max <= 0:
            y_max = 1.0

        bar_colors = [ABLATION_BAR_FACES.get(n, "0.80") for n in names]
        bar_edges  = [ABLATION_BAR_EDGES.get(n, "#8B929E") for n in names]
        bar_lws    = [0.65 for n in names]
        ax.bar(
            x, vals, width=0.58,
            color=bar_colors, edgecolor=bar_edges, linewidth=bar_lws,
            zorder=3,
        )

        # # 基线虚线
        # baseline_val = vals[0]
        # if not is_sat and not np.isnan(baseline_val) and baseline_val > 0.05 * y_max:
        #     ax.axhline(
        #         baseline_val, color="#9AA3AE",
        #         ls=(0, (4, 3)), lw=0.7, alpha=0.75, zorder=1,
        #     )

        # # 条形顶端数值标签
        # for xi, val in enumerate(vals):
        #     if np.isnan(val):
        #         continue
        #     if is_sat and val == 0.0:
        #         ax.text(xi, y_max * 0.04, "0",
        #                 ha="center", va="bottom",
        #                 fontsize=5.0, color=PAPER_SUCCESS, fontweight="bold")
        #         continue
        #     ax.text(xi, val + y_max * 0.028, fmt.format(val),
        #             ha="center", va="bottom", fontsize=5.0, color="#1F2933",
        #             fontweight=("bold" if names[xi] == "shcs" else "normal"))

        ax.set_xticks(x)
        ax.set_xticklabels(
            [_SINGLE_LINE.get(lb, lb) for lb in labels],
            rotation=0, ha="center", fontsize=5.5,
        )
        ax.set_ylabel(ylabel, fontsize=6.0)
        ax.set_ylim(0, y_max * 1.40)
        ax.grid(axis="y", alpha=0.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # 子图标题置于底部（与 exp_01 composite 风格一致）
        ax.text(0.5, -0.20, caption,
                transform=ax.transAxes, ha="center", va="top",
                fontweight="bold", fontsize=6.5, clip_on=False)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2×3 复合图
# ─────────────────────────────────────────────────────────────────────────────

def make_composite(results, labels, names, waypoints, cfg) -> plt.Figure:
    """生成 2×3 复合图（论文 Fig. 6 全宽主图）。

    布局：
      行 0：(a) 轨迹  (b) 航向误差  (c) 横向误差
      行 1：(d) 偏航力矩  (e) 纵荡速度  (f) 横向误差RMS条形图

    轨迹图包含图例，其余时序面板不重复显示图例以节省空间。
    (f) 位置展示单指标条形对比，完整 4 指标图保存在 subfigs/fig6f_bar.png。
    """
    apply_plot_style("composite")

    tau_r_lim = cfg["actuator"]["T_max"] * cfg["actuator"]["b"]

    fig = plt.figure(figsize=heu_figsize("large", 0.72), layout="constrained")
    gs  = gridspec.GridSpec(2, 3, figure=fig)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    ax_traj, ax_hdg, ax_cte, ax_taur, ax_surge, ax_bar = axes

    # 填充各子图内容
    _draw_traj(ax_traj, results, labels, names, waypoints, cfg)
    _draw_heading(ax_hdg, results, names)
    _draw_cte(ax_cte, results, names)
    _draw_taur(ax_taur, results, names, tau_r_lim)
    _draw_surge(ax_surge, results, names)
    _draw_cte_bar(ax_bar, results, labels, names)

    # 底部子图说明（紧贴 x 轴下方）
    panel_captions = [
        "(a) 轨迹", "(b) 航向误差", "(c) 横向误差",
        "(d) 偏航力矩", "(e) 纵荡速度", "(f) 横向误差RMS",
    ]
    fs = plt.rcParams["axes.titlesize"]
    for ax, caption in zip(axes, panel_captions):
        ax.text(0.5, -0.18, caption,
                transform=ax.transAxes, ha="center", va="top",
                fontweight="bold", fontsize=fs, clip_on=False)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CSV 数据处理
# ─────────────────────────────────────────────────────────────────────────────

def make_csv_rows(results, labels, names) -> list[dict]:
    """将各方法的仿真汇总指标格式化为 CSV 行列表。"""
    rows = []
    for res, lbl, name in zip(results, labels, names):
        row = {"Method": lbl, "method_key": name}
        for field, col, prec in TABLE_FIELDS:
            val = res["summary"].get(field, float("nan"))
            try:
                row[col] = round(float(val), prec)
            except (ValueError, TypeError):
                row[col] = ""
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 0. 加载配置 ──────────────────────────────────────────────────────────
    cfg       = get_config()
    waypoints = get_path(PATH_NAME, cfg)
    dist_cfg  = cfg["disturbances"][DIST_NAME]
    eta0, nu0 = get_initial_state(cfg, EXPERIMENT_NAME)
    labels    = [cfg["methods"][m]["label"] for m in METHOD_NAMES]

    print(f"{'='*65}")
    print(f"  Sec 4.6 消融实验 v2（统一前视Δ=4m）  ({PATH_NAME} + {DIST_NAME})")
    print(f"  方法序列: {' → '.join(labels)}")
    print(f"{'='*65}")

    # ── 1. 运行仿真（多种方法，顺序执行）───────────────────────────────────
    results = []
    for name, lbl in zip(METHOD_NAMES, labels):
        print(f"  [{lbl:16s}] ...", end="  ", flush=True)
        res = run_trial(name, cfg, waypoints, eta0, nu0, dist_cfg, SEED)
        results.append(res)
        s = res["summary"]
        t_show = s.get("reach_time", float("nan")) if s.get("reached_goal") else s.get("completion_time", float("nan"))
        print(f"完成={s.get('path_completed')}  到达={s.get('reached_goal')}  "
              f"t={t_show:.1f}s  CTE={s['cross_track_rms']:.3f}m  "
              f"Hdg={s['heading_error_rms']:.4f}rad  "
              f"YawE={s['control_energy_tau_r_cmd']:.1f}  "
              f"SatT={s['sat_time_raw']:.2f}s")

    # ── 2. 打印指标表 & 相对 ILOS-PID 的改善量 ───────────────────────────────
    ABLATION_METRICS = [
        ("cross_track_rms",          "横向误差RMS [m]",      ".3f"),
        ("turn_cte_rms",             "转弯横向误差RMS [m]", ".3f"),
        ("heading_error_rms",        "航向RMS [rad]",        ".4f"),
        ("control_energy_tau_r_cmd", "偏航能耗",             ".1f"),
        ("total_control_energy",     "总能耗",               ".1f"),
        ("sat_time_raw",             "饱和时间 [s]",         ".2f"),
        ("sat_ratio_raw",            "饱和比例",             ".4f"),
        ("speed_reduction_max_pct",  "最大降速 [%]",         ".1f"),
        ("reach_time",               "到达时间 [s]",         ".1f"),
    ]
    print_metrics_table(results, labels, ABLATION_METRICS)

    base = results[0]["summary"]
    print(f"\n  各方法相对 ILOS-PID 改善量：")
    print(f"  {'Method':<18}  {'ΔCTE':>8}  {'ΔHdg':>8}  {'ΔYawE':>8}  {'ΔSatT':>8}")
    for res, lbl in zip(results[1:], labels[1:]):
        s  = res["summary"]
        dc = relative_improvement(base["cross_track_rms"],          s["cross_track_rms"])
        dh = relative_improvement(base["heading_error_rms"],        s["heading_error_rms"])
        de = relative_improvement(base["control_energy_tau_r_cmd"], s["control_energy_tau_r_cmd"])
        ds = relative_improvement(base["sat_time_raw"],             s["sat_time_raw"])
        print(f"  {lbl:<18}  {dc:+7.1f}%  {dh:+7.1f}%  {de:+7.1f}%  {ds:+7.1f}%")

    # ── 3. 保存原始数据 ───────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, res in zip(METHOD_NAMES, results):
        save_result(res, OUT_DIR, name)
    save_summaries_csv(make_csv_rows(results, labels, METHOD_NAMES),
                       OUT_DIR / "ablation_summary.csv")
    print(f"\n  CSV: {OUT_DIR / 'ablation_summary.csv'}")

    # ── 4. 保存子图（6张独立面板）────────────────────────────────────────────
    print("\n  保存子图 →", OUT_DIR / "subfigs")
    subfigs_dir = OUT_DIR / "subfigs"
    subfigs_dir.mkdir(parents=True, exist_ok=True)

    fig = make_traj_panel(results, labels, METHOD_NAMES, waypoints, cfg)
    save_fig(fig, subfigs_dir / "fig6a_traj.png")

    fig = make_heading_panel(results, labels, METHOD_NAMES)
    save_fig(fig, subfigs_dir / "fig6b_heading.png")

    fig = make_cte_panel(results, labels, METHOD_NAMES)
    save_fig(fig, subfigs_dir / "fig6c_cte.png")

    fig = make_taur_panel(results, labels, METHOD_NAMES, cfg)
    save_fig(fig, subfigs_dir / "fig6d_taur.png")

    fig = make_surge_panel(results, labels, METHOD_NAMES)
    save_fig(fig, subfigs_dir / "fig6e_surge.png")

    fig = make_bar_panel(results, labels, METHOD_NAMES)
    save_fig(fig, subfigs_dir / "fig6f_bar.png")

    # ── 5. 保存组图（2×3 主图）───────────────────────────────────────────────
    print("\n  保存组图 →", OUT_DIR / "fig6_composite.png")
    fig = make_composite(results, labels, METHOD_NAMES, waypoints, cfg)
    save_fig(fig, OUT_DIR / "fig6_composite.png")

    print("\n  实验 4.6 v2 全部完成。")


if __name__ == "__main__":
    main()
