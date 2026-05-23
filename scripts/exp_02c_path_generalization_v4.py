"""
实验 4.3（v4，简化单栏版）：多路径泛化性分析

与 v3 的区别：
  - 路径精简为两条代表性路径（Z形双折、U形折返），去掉 L形（已在 3.2 节覆盖）
    和蛇形（宽长比大，单栏小图中难以呈现细节）；
  - 对比方法精简为 ILOS-PID（los_pid_short）与 SHCS，去掉 FO——FO 特性已在
    典型场景（3.2 节）得到充分刻画，泛化实验聚焦最强基线与提出方法的对比；
  - 输出图幅改为单栏宽度（7 cm），适合在双栏论文的单列中放置。

对比方法（均使用 Δ=4 m）：
  ILOS-PID（los_pid_short） / SHCS（shcs）

输出目录：
  results/baseline/02c_path_generalization_v4/
    generalization_v4_summary.csv     — 完整指标汇总（含改善量行）
    fig_traj_composite.png            — 轨迹组图（2 工况 × 2 路径，单栏宽）
    fig_metrics.png                   — 指标汇总组图（单栏宽）
    fig_path_overview.png             — 路径概览图
    subfigs/                          — 各路径×工况独立轨迹子图
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
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from baseline import get_config
from scripts.experiment_utils import (
    run_trial, save_result, save_summaries_csv,
    apply_plot_style, get_method_style, relative_improvement, RESULTS_ROOT,
    get_initial_state, save_fig,
    heu_figsize,
)

# ─────────────────────────────────────────────────────────────────────────────
# 常量 & 输出目录
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR      = RESULTS_ROOT / "02c_path_generalization_v4"
METHOD_NAMES = ["los_pid_short", "shcs"]          # v4：仅 ILOS-PID vs SHCS
DIST_SHOW    = ["calm", "steady_current"]         # 主图展示的工况
DIST_ALL     = ["calm", "steady_current", "current"]
SEED         = 1

# ─────────────────────────────────────────────────────────────────────────────
# 路径定义（仅保留 Z形双折 与 U形折返）
# ─────────────────────────────────────────────────────────────────────────────

PATHS_V4 = {
    "z_double": dict(
        wps   = np.array([[0, 0], [40, 0], [40, 55], [80, 55]], dtype=float),
        label = "Z形",
        short = "Z",
    ),
    "u_return": dict(
        wps   = np.array([[0, 0], [60, 0], [60, 65], [0, 65]], dtype=float),
        label = "U形",
        short = "U",
    ),
}
PATH_NAMES_V4 = ["z_double", "u_return"]

# 保留全部路径定义（供 summary table 参考，不参与主图绘制）
def _make_snake() -> np.ndarray:
    return np.array([
        [0, 0], [35, 0], [35, 60], [70, 60], [70, 0], [100, 0]
    ], dtype=float)

PATHS_ALL = {
    "l_single": dict(
        wps   = np.array([[0, 0], [50, 0], [50, 60]], dtype=float),
        label = "L形（单折90°）",
        short = "L",
    ),
    **PATHS_V4,
    "snake_alt": dict(
        wps   = _make_snake(),
        label = "蛇形（交替90°）",
        short = "S3",
    ),
}
PATH_NAMES_ALL = ["l_single", "z_double", "u_return", "snake_alt"]

DIST_LABELS = {
    "calm":           "无扰动",
    "steady_current": "定常海流",
    "current":        "含噪海流",
}

KEY_METRICS = [
    ("cross_track_rms",           "CTE_RMS_m",      4),
    ("cross_track_iae",           "CTE_IAE",         2),
    ("turn_cte_rms",              "TurnCTE_RMS_m",   4),
    ("heading_los_error_rms",     "HdgLOS_RMS_rad",  5),
    ("control_energy_tau_r_cmd",  "YawEnergy",       2),
    ("speed_reduction_max_pct",   "MaxSpeedRed%",    1),
]

# ─────────────────────────────────────────────────────────────────────────────
# 面板宽度比例
# ─────────────────────────────────────────────────────────────────────────────

def _compute_width_ratios() -> list:
    """各路径面板宽度按 max(x_extent, y_extent×0.75) 等比缩放。"""
    ratios = []
    for pn in PATH_NAMES_V4:
        wps  = PATHS_V4[pn]["wps"]
        xext = float(np.ptp(wps[:, 0]))
        yext = float(np.ptp(wps[:, 1]))
        ratios.append(max(xext, yext * 0.75, 1.0))
    return ratios

_WIDTH_RATIOS = _compute_width_ratios()

DIST_LABELS_EN = {
    "calm":           "Calm",
    "steady_current": "Steady",
    "current":        "Noisy",
}

TRAJ_STYLES = {
    "los_pid_short": dict(color="#263B52", ls=(0, (3.8, 2.0)), lw=1.0,
                          zorder=7, alpha=0.98),
    "shcs":          dict(color="#B24A3B", ls="-", lw=1.2,
                          zorder=6, alpha=0.92),
}


def _traj_style(method_name: str, idx: int = 0) -> dict:
    color, ls = get_method_style(method_name, idx)
    style = dict(color=color, ls=ls, lw=1.1, zorder=5, alpha=1.0)
    style.update(TRAJ_STYLES.get(method_name, {}))
    return style


# ─────────────────────────────────────────────────────────────────────────────
# 图例与面板绘制辅助
# ─────────────────────────────────────────────────────────────────────────────

def _legend_handles(cfg) -> tuple:
    """构建轨迹图例句柄（参考路径 + 各方法）。
    ILOS 用虚线以便与 SHCS 实线视觉区分（两者轨迹几乎重合时仍可辨认）。
    """
    handles = [Line2D([0], [0], color="gray", lw=0.8, ls="--", alpha=0.5)]
    labels  = ["Ref."]
    for idx, mn in enumerate(METHOD_NAMES):
        style = _traj_style(mn, idx)
        handles.append(Line2D([0], [0],
                              color=style["color"], lw=style["lw"],
                              ls=style["ls"], alpha=style["alpha"]))
        labels.append(cfg["methods"][mn]["label"])
    return handles, labels


def _plot_wps(ax, wps, goal_tol):
    ax.plot(wps[:, 0], wps[:, 1], color="gray", ls="--", lw=0.8, alpha=0.45,
            label="Reference")
    ax.plot(wps[0, 0],  wps[0, 1], marker="^", color="#3B6EA8", ms=3.2,
            linestyle="none", zorder=6)
    ax.plot(wps[-1, 0], wps[-1, 1], marker="*", color="#B24A3B", ms=4.2,
            linestyle="none", zorder=6)
    if goal_tol is not None:
        ax.add_patch(plt.Circle(
            (wps[-1, 0], wps[-1, 1]),
            goal_tol, fill=False, color="#B24A3B", ls=":", lw=0.7, alpha=0.28,
        ))


def _draw_final_pos(ax, log, color):
    ax.plot(log["x"][-1], log["y"][-1],
            marker="o", ms=2.3, mfc="white", mec=color, mew=0.65,
            linestyle="none", zorder=7)


def _draw_panel(ax, all_results, dist_name, path_name, wps, cfg,
                show_legend=False, show_xlabel=False, show_ylabel=False):
    goal_tol = float(cfg["simulation"].get("goal_tolerance", 3.0))
    _plot_wps(ax, wps, goal_tol)

    # 先画 SHCS（实线，底层），再画 ILOS（虚线，顶层），使虚线始终可见
    draw_order = sorted(enumerate(METHOD_NAMES),
                        key=lambda t: 0 if t[1] == "shcs" else 1)
    for midx, mn in draw_order:
        key = (dist_name, path_name, mn)
        if key not in all_results:
            continue
        res = all_results[key]
        style = _traj_style(mn, midx)
        line, = ax.plot(res["log"]["x"], res["log"]["y"],
                        color=style["color"], lw=style["lw"],
                        ls=style["ls"], alpha=style["alpha"],
                        zorder=style["zorder"])
        if mn == "los_pid_short":
            line.set_path_effects([
                pe.Stroke(linewidth=style["lw"] + 0.85,
                          foreground="white", alpha=0.74),
                pe.Normal(),
            ])
        _draw_final_pos(ax, res["log"], style["color"])

    ax.margins(0.06)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x / m" if show_xlabel else "", fontsize=7.0, labelpad=1.0)
    ax.set_ylabel("y / m" if show_ylabel else "", fontsize=7.0, labelpad=1.0)

    if show_legend:
        h, lbl = _legend_handles(cfg)
        ax.legend(h, lbl, loc="upper left", ncol=1,
                  fontsize=5.5, handlelength=1.45,
                  framealpha=0.88, edgecolor="#c8d0d8",
                  borderpad=0.28, handletextpad=0.35,
                  labelspacing=0.22)


# ─────────────────────────────────────────────────────────────────────────────
# 轨迹组图（2 工况 × 2 路径，单栏宽）
# ─────────────────────────────────────────────────────────────────────────────

def make_composite(all_results, cfg) -> plt.Figure:
    """生成轨迹组图（单栏宽）。

    行 = 工况（DIST_SHOW），列 = 路径（PATH_NAMES_V4）。
    图幅使用 heu_figsize("small", ...) = 7 cm 宽，适合单栏放置。
    """
    apply_plot_style("composite")

    n_rows = len(DIST_SHOW)   # 2
    n_cols = len(PATH_NAMES_V4)  # 2

    # Single-column width with extra vertical room for below-axis panel letters.
    fig = plt.figure(figsize=heu_figsize("small", 1.10))
    gs  = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        width_ratios=_WIDTH_RATIOS,
        hspace=0.48, wspace=0.24,
        left=0.19, right=0.99, bottom=0.21, top=0.90,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)]
            for r in range(n_rows)]

    panel_idx = 0
    for row_idx, dist_name in enumerate(DIST_SHOW):
        for col_idx, path_name in enumerate(PATH_NAMES_V4):
            ax  = axes[row_idx][col_idx]
            wps = PATHS_V4[path_name]["wps"]

            is_first    = (row_idx == 0 and col_idx == 0)
            is_last_row = (row_idx == n_rows - 1)
            is_left_col = (col_idx == 0)

            _draw_panel(ax, all_results, dist_name, path_name, wps, cfg,
                        show_legend=is_first,
                        show_xlabel=is_last_row,
                        show_ylabel=is_left_col)

            # Panel letters are placed below each subplot.
            letter = chr(ord("a") + panel_idx)
            label_y = -0.22 if not is_last_row else -0.48
            ax.text(0.5, label_y, f"({letter})",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=6.8, fontweight="bold", clip_on=False)

            # 列标题（路径名，仅首行）
            if row_idx == 0:
                ax.set_title(PATHS_V4[path_name]["label"],
                             fontweight="bold", fontsize=7.5, pad=5.0)

            panel_idx += 1

    # Row labels sit in the enlarged left gutter, away from the y-axis label.
    for row_idx, dist_name in enumerate(DIST_SHOW):
        ss = gs[row_idx, 0]
        x0, y0, x1, y1 = ss.get_position(fig).extents
        fig.text(0.045, (y0 + y1) / 2, DIST_LABELS[dist_name],
                 rotation=90, ha="center", va="center",
                 fontsize=6.8, fontweight="bold")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 路径概览图（辅助图）
# ─────────────────────────────────────────────────────────────────────────────

def make_path_overview(cfg) -> plt.Figure:
    apply_plot_style("composite")

    goal_tol = float(cfg["simulation"].get("goal_tolerance", 3.0))
    n_cols   = len(PATH_NAMES_V4)

    fig = plt.figure(figsize=heu_figsize("small", 0.38))
    gs  = gridspec.GridSpec(
        1, n_cols, figure=fig,
        width_ratios=_WIDTH_RATIOS,
        hspace=0, wspace=0.26,
        left=0.08, right=0.99, bottom=0.14, top=0.82,
    )

    for col_idx, pn in enumerate(PATH_NAMES_V4):
        ax  = fig.add_subplot(gs[0, col_idx])
        wps = PATHS_V4[pn]["wps"]
        _plot_wps(ax, wps, goal_tol)
        ax.margins(0.12)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(PATHS_V4[pn]["label"], fontweight="bold",
                     fontsize=7.4, pad=4.0)
        ax.set_xlabel("x / m", fontsize=6.5)
        if col_idx == 0:
            ax.set_ylabel("y / m", fontsize=6.5)

        for wi, wp in enumerate(wps):
            ax.annotate(f"P{wi}", xy=(wp[0], wp[1]),
                        fontsize=5.5, color="#555",
                        ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")

    fig.suptitle("实验路径概览（Z形双折与U形折返）",
                 fontweight="bold", fontsize=7.8, y=0.98)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 独立轨迹子图存档
# ─────────────────────────────────────────────────────────────────────────────

def save_subfigures(all_results, cfg, out_dir: Path) -> None:
    apply_plot_style("composite")
    out_dir.mkdir(parents=True, exist_ok=True)

    for dist_idx, dist_name in enumerate(DIST_ALL):
        for path_idx, path_name in enumerate(PATH_NAMES_V4):
            wps    = PATHS_V4[path_name]["wps"]
            xext   = float(np.ptp(wps[:, 0]))
            yext   = max(float(np.ptp(wps[:, 1])), 1.0)
            aspect = float(np.clip(yext / max(xext, 1.0), 0.45, 1.05))
            w, h   = heu_figsize("small", aspect)

            fig, ax = plt.subplots(figsize=(w, h), layout="constrained")
            show_legend = (dist_idx == 0 and path_idx == 0)
            _draw_panel(ax, all_results, dist_name, path_name, wps, cfg,
                        show_legend=show_legend,
                        show_xlabel=True, show_ylabel=True)
            ax.set_title(
                f"{PATHS_V4[path_name]['label']} — {DIST_LABELS[dist_name]}",
                fontweight="bold", loc="left", fontsize=7.4)
            save_fig(fig, out_dir / f"v4_{dist_name}_{path_name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 指标汇总组图（1 行 × 2 列，参考 fig6f_bar 风格）
# ─────────────────────────────────────────────────────────────────────────────

def make_metrics_figure(all_results, cfg) -> plt.Figure:
    """生成性能指标汇总组图（单栏宽，fig6f_bar 风格）。

    布局：1 行 × 2 列（CTE RMS | 偏航能耗）。
    x 轴：4 场景 = [Z-Calm, Z-Steady, U-Calm, U-Steady]，每场景 2 根柱（ILOS / SHCS）。
    配色：ILOS 浅灰底+深灰边，SHCS 砖红底+深红边，与消融图 fig6f_bar 风格一致。
    中间竖虚线区分 Z形 / U形 两组。
    """
    apply_plot_style("composite")

    # x 轴场景顺序：路径在外层，工况在内层
    scenarios = [(pn, dn) for pn in PATH_NAMES_V4 for dn in DIST_SHOW]
    x_tick_labels = [
        f"{PATHS_V4[pn]['short']}-{DIST_LABELS_EN.get(dn, dn)}"
        for pn, dn in scenarios
    ]
    n_scen = len(scenarios)          # 4
    x      = np.arange(n_scen)
    width  = 0.28
    offsets = np.array([-0.5, 0.5]) * width   # [-0.14, +0.14]

    # 与 experiment_utils.ABLATION_BAR_* 风格一致：冷灰 + 暖陶色，统一细描边
    BAR_FACE = {
        "los_pid_short": "#C8CDD5",   # cool light gray
        "shcs":          "#C27B73",   # warm muted terracotta
    }
    BAR_EDGE = {
        "los_pid_short": "#8B929E",   # medium gray
        "shcs":          "#8F4A44",   # muted dark rose
    }
    BAR_LW = {
        "los_pid_short": 0.65,
        "shcs":          0.65,        # 统一描边粗细，不再突出 SHCS 边框
    }

    metric_defs = [
        ("cross_track_rms",          "CTE-RMS / m",  "横向误差RMS"),
        ("control_energy_tau_r_cmd", "Yaw-Energy / $(N^2\cdot m^2\cdot s)$",   "偏航控制能耗"),
    ]

    # Single-column width; extra bottom room keeps rotated x labels and captions clear.
    fig, axes = plt.subplots(1, 2, figsize=heu_figsize("small", 0.68))
    fig.subplots_adjust(left=0.19, right=0.985, top=0.90, bottom=0.34,
                        wspace=0.78)

    for ax_idx, (field, ylabel, caption) in enumerate(metric_defs):
        ax = axes[ax_idx]

        # 收集各方法在各场景下的指标值
        method_data = {}
        all_vals    = []
        for mn in METHOD_NAMES:
            vals = []
            for pn, dn in scenarios:
                res = all_results.get((dn, pn, mn))
                v = float(res["summary"].get(field, float("nan"))) if res else float("nan")
                vals.append(v)
                if not np.isnan(v):
                    all_vals.append(v)
            method_data[mn] = vals
        y_max = max(all_vals) if all_vals else 1.0

        for midx, mn in enumerate(METHOD_NAMES):
            label = cfg["methods"][mn]["label"] if ax_idx == 0 else "_nolegend_"
            ax.bar(
                x + offsets[midx], method_data[mn], width,
                color=BAR_FACE[mn], edgecolor=BAR_EDGE[mn],
                linewidth=BAR_LW[mn], label=label, zorder=3,
            )
        # 竖虚线区分两条路径（Z / U）
        ax.axvline(x=1.5, color="#cccccc", lw=0.7, ls="--", zorder=1)

        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels, fontsize=5.1,
                           rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, n_scen - 0.4)
        ax.set_ylim(0, y_max * 1.28)
        ax.set_ylabel(ylabel, fontsize=6.1, labelpad=5.0)
        ax.tick_params(axis="y", labelsize=5.8, pad=1.5)
        ax.grid(axis="y", alpha=0.28, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # 子图标题（仿 fig6f_bar 风格：下方居中加粗）
        letter = chr(ord("a") + ax_idx)
        caption_text = "横向误差RMS" if ax_idx == 0 else "偏航控制能耗"
        ax.text(0.5, -0.32, f"({letter}) {caption_text}",
                transform=ax.transAxes, ha="center", va="top",
                fontweight="bold",
                fontsize=6.4,
                clip_on=False)

        # Keep the legend inside the first subplot, but make it compact.
        if ax_idx == 0:
            ax.legend(loc="best", ncol=1, fontsize=5.2,
                      handlelength=1.15, framealpha=0.90,
                      edgecolor="#c8d0d8", borderpad=0.25,
                      handletextpad=0.35, labelspacing=0.18)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CSV 汇总表（保留全部路径以供参考）
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_table(all_results, cfg) -> list:
    """生成 CSV 汇总行（仅含已运行的 2 路径 × 3 工况 × 2 方法）。"""
    rows = []
    for dist_name in DIST_ALL:
        for pn in PATH_NAMES_V4:
            base_res = all_results.get((dist_name, pn, "los_pid_short"))
            shcs_res = all_results.get((dist_name, pn, "shcs"))
            if base_res is None or shcs_res is None:
                continue
            for mn in METHOD_NAMES:
                res = all_results.get((dist_name, pn, mn))
                if res is None:
                    continue
                s = res["summary"]
                row = {
                    "Disturbance": DIST_LABELS.get(dist_name, dist_name),
                    "Path":        PATHS_V4[pn]["label"],
                    "Method":      cfg["methods"][mn]["label"],
                    "Reached":     bool(s.get("reached_goal", False)),
                    "GoalDist_m":  round(float(s.get("final_dist_to_goal",
                                                      float("nan"))), 3),
                }
                for field, col, prec in KEY_METRICS:
                    val = s.get(field, float("nan"))
                    try:
                        row[col] = round(float(val), prec)
                    except (ValueError, TypeError):
                        row[col] = ""
                rows.append(row)
            # SHCS vs ILOS-PID 改善量行
            imp = {
                "Disturbance": DIST_LABELS.get(dist_name, dist_name),
                "Path":        PATHS_V4[pn]["label"],
                "Method":      "SHCS vs ILOS-PID (%)",
            }
            for field, col, _ in KEY_METRICS:
                bv = base_res["summary"].get(field, float("nan"))
                nv = shcs_res["summary"].get(field, float("nan"))
                try:
                    pct = relative_improvement(float(bv), float(nv))
                    imp[col] = f"{pct:+.1f}%" if np.isfinite(pct) else "n/a"
                except (ValueError, TypeError):
                    imp[col] = ""
            rows.append(imp)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 打印改善量汇总（控制台）
# ─────────────────────────────────────────────────────────────────────────────

def _print_improvement_summary(all_results) -> None:
    print(f"\n{'─'*72}")
    print("  SHCS vs ILOS-PID 改善汇总（正值=改善，负值=变差）")
    print(f"{'─'*72}")
    header = f"  {'工况':<18} {'路径':<20} {'ΔCTE%':>8} {'ΔTurnCTE%':>11} {'ΔYawE%':>8}"
    print(header)
    for dist_name in DIST_ALL:
        for pn in PATH_NAMES_V4:
            bs = all_results.get((dist_name, pn, "los_pid_short"), {}).get("summary", {})
            ss = all_results.get((dist_name, pn, "shcs"), {}).get("summary", {})
            if not bs or not ss:
                continue
            cte_i  = relative_improvement(bs.get("cross_track_rms", float("nan")),
                                          ss.get("cross_track_rms", float("nan")))
            turn_i = relative_improvement(bs.get("turn_cte_rms", float("nan")),
                                          ss.get("turn_cte_rms", float("nan")))
            yaw_i  = relative_improvement(bs.get("control_energy_tau_r_cmd", float("nan")),
                                          ss.get("control_energy_tau_r_cmd", float("nan")))
            print(f"  {DIST_LABELS.get(dist_name, dist_name):<18} "
                  f"{PATHS_V4[pn]['label']:<20} "
                  f"{cte_i:>+7.1f}%  {turn_i:>+9.1f}%  {yaw_i:>+7.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 0. 加载配置 ──────────────────────────────────────────────────────────
    cfg       = get_config()
    eta0, nu0 = get_initial_state(cfg, "path_generalization")

    total = len(DIST_ALL) * len(PATH_NAMES_V4) * len(METHOD_NAMES)
    print(f"{'='*65}")
    print(f"  多路径泛化 v4  "
          f"({len(PATH_NAMES_V4)} 路径 × {len(DIST_ALL)} 工况 "
          f"× {len(METHOD_NAMES)} 方法 = {total} 组）")
    print(f"  路径：Z形双折、U形折返")
    print(f"  方法：ILOS-PID vs SHCS（FO已在3.2节刻画，本节不再重复）")
    print(f"{'='*65}")

    # ── 1. 仿真 ──────────────────────────────────────────────────────────────
    all_results: dict = {}
    done = 0
    for dist_name in DIST_ALL:
        dist_cfg = cfg["disturbances"][dist_name]
        for pn in PATH_NAMES_V4:
            wps = PATHS_V4[pn]["wps"]
            for mn in METHOD_NAMES:
                done += 1
                lbl = cfg["methods"][mn]["label"]
                print(f"  [{done:2d}/{total}] {dist_name} + {pn} + {lbl} ...",
                      end="  ", flush=True)
                res = run_trial(mn, cfg, wps, eta0, nu0, dist_cfg, SEED)
                all_results[(dist_name, pn, mn)] = res
                s = res["summary"]
                t = (s.get("reach_time", float("nan")) if s.get("reached_goal")
                     else s.get("completion_time", float("nan")))
                print(f"到达={s['reached_goal']}  t={t:.1f}s  "
                      f"CTE={s['cross_track_rms']:.3f}m  "
                      f"YawE={s['control_energy_tau_r_cmd']:.1f}")

    # ── 2. 控制台改善量汇总 ──────────────────────────────────────────────────
    _print_improvement_summary(all_results)

    # ── 3. 保存原始数据与 CSV ─────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for (dist_name, pn, mn), res in all_results.items():
        save_result(res, OUT_DIR, f"{dist_name}_{pn}_{mn}")
    csv_path = OUT_DIR / "generalization_v4_summary.csv"
    save_summaries_csv(make_summary_table(all_results, cfg), csv_path)
    print(f"\n  CSV → {csv_path}")

    # ── 4. 保存路径概览图 ─────────────────────────────────────────────────────
    ov_path = OUT_DIR / "fig_path_overview.png"
    print(f"\n  保存路径概览图 → {ov_path}")
    save_fig(make_path_overview(cfg), ov_path)

    # ── 5. 保存独立轨迹子图 ───────────────────────────────────────────────────
    subfig_dir = OUT_DIR / "subfigs"
    print(f"\n  保存子图 → {subfig_dir}")
    save_subfigures(all_results, cfg, subfig_dir)

    # ── 6. 保存轨迹组图（单栏宽）─────────────────────────────────────────────
    comp_path = OUT_DIR / "fig_traj_composite.png"
    print(f"\n  保存轨迹组图 → {comp_path}")
    save_fig(make_composite(all_results, cfg), comp_path)

    # ── 7. 保存指标组图（单栏宽）─────────────────────────────────────────────
    met_path = OUT_DIR / "fig_metrics.png"
    print(f"\n  保存指标组图 → {met_path}")
    save_fig(make_metrics_figure(all_results, cfg), met_path)

    print("\n  多路径泛化 v4 全部完成。")


if __name__ == "__main__":
    main()
