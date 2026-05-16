"""
实验 4.3（v3，统一前视距离版）：多路径-多工况泛化性分析

与 exp_02b 的区别：
  - ILOS-PID 基线改用 los_pid_short（Δ=4 m），与 SHCS/FO 保持相同前视距离，
    使方法间差异完全来自参考整形机制，而非前视距离选择。

对比方法（均使用 Δ=4 m）：
  ILOS-PID（los_pid_short） / ILOS+FO（first_order） / SHCS（shcs）

输出目录：
  results/baseline/02c_path_generalization_v3/
    generalization_v3_summary.csv     — 完整指标汇总（含改善量行）
    fig_traj_composite.png            — 轨迹组图（2 工况 × 4 路径）
    fig_metrics.png                   — 指标汇总组图
    fig_path_overview.png             — 路径概览图（辅助理解）
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

OUT_DIR      = RESULTS_ROOT / "02c_path_generalization_v3"
METHOD_NAMES = ["los_pid_short", "first_order", "shcs"]
DIST_SHOW    = ["calm", "steady_current"]             # 主图展示的工况（无扰动+定常海流对比最鲜明）
DIST_ALL     = ["calm", "steady_current", "current"]  # 全量运行
SEED         = 1

# ─────────────────────────────────────────────────────────────────────────────
# 路径定义（内联 numpy 数组，不依赖 baseline_config 路径名）
# ─────────────────────────────────────────────────────────────────────────────

def _make_snake() -> np.ndarray:
    """蛇形折线：东-北-东-南-东，3 次交替 90° 折弯。
    形成波浪状路径，测试方向交替切换时的参考整形连续性。
    """
    return np.array([
        [0, 0], [35, 0], [35, 60], [70, 60], [70, 0], [100, 0]
    ], dtype=float)

PATHS_V2 = {
    "l_single": dict(
        wps   = np.array([[0, 0], [50, 0], [50, 60]], dtype=float),
        label = "L形（单折90°）",
        short = "L",
    ),
    "z_double": dict(
        wps   = np.array([[0, 0], [40, 0], [40, 55], [80, 55]], dtype=float),
        label = "Z形（双折90°）",
        short = "Z",
    ),
    "u_return": dict(
        wps   = np.array([[0, 0], [60, 0], [60, 65], [0, 65]], dtype=float),
        label = "U形折返",
        short = "U",
    ),
    "snake_alt": dict(
        wps   = _make_snake(),
        label = "蛇形（交替90°）",
        short = "S3",
    ),
}
PATH_NAMES_V2 = ["l_single", "z_double", "u_return", "snake_alt"]

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
# 面板宽度比例（保证各列轨迹图米/英寸比例一致）
# ─────────────────────────────────────────────────────────────────────────────

def _compute_width_ratios() -> list:
    """各路径面板宽度按 max(x_extent, y_extent×0.75) 等比缩放。

    等比因子取 max 而不是单纯 x_extent，是为防止近方形路径（如 U 形）
    对应的列被压得过窄。
    """
    ratios = []
    for pn in PATH_NAMES_V2:
        wps  = PATHS_V2[pn]["wps"]
        xext = float(np.ptp(wps[:, 0]))
        yext = float(np.ptp(wps[:, 1]))
        ratios.append(max(xext, yext * 0.75, 1.0))
    return ratios

_WIDTH_RATIOS = _compute_width_ratios()


# ─────────────────────────────────────────────────────────────────────────────
# 图例与面板绘制辅助
# ─────────────────────────────────────────────────────────────────────────────

def _legend_handles(cfg) -> tuple:
    """构建轨迹图例句柄（参考路径 + 各方法）。"""
    handles = [Line2D([0], [0], color="gray", lw=0.8, ls="--", alpha=0.5)]
    labels  = ["Ref."]
    for idx, mn in enumerate(METHOD_NAMES):
        color, ls = get_method_style(mn, idx)
        lw = 1.1 if mn == "shcs" else 0.65
        handles.append(Line2D([0], [0], color=color, lw=lw, ls=ls))
        labels.append(cfg["methods"][mn]["label"])
    return handles, labels


def _plot_wps(ax, wps, goal_tol):
    """绘制参考路径与航点标记，采用更小的图标尺寸。

    刻意减小起点三角（ms=3.2）和终点星（ms=4.2），使其在组图中
    不会遮挡轨迹线，同时目标圆半径标注也更轻薄。
    """
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
    """标记仿真结束位置，使用缩小的空心圆（ms=2.3）。"""
    ax.plot(log["x"][-1], log["y"][-1],
            marker="o", ms=2.3, mfc="white", mec=color, mew=0.65,
            linestyle="none", zorder=7)


def _annotate_improvement(ax, all_results, dist_name, path_name):
    """在面板右下角标注 SHCS 相对 ILOS-PID 的 CTE 改善幅度。

    即使轨迹空间差异不易用肉眼分辨，量化标注可直接告知读者性能收益。
    正值（改善）显示绿色，负值（退化）显示红色。
    """
    ilos_r = all_results.get((dist_name, path_name, "los_pid_short"))
    shcs_r = all_results.get((dist_name, path_name, "shcs"))
    if ilos_r is None or shcs_r is None:
        return
    ilos_cte = ilos_r["summary"].get("cross_track_rms", float("nan"))
    shcs_cte = shcs_r["summary"].get("cross_track_rms", float("nan"))
    imp = relative_improvement(ilos_cte, shcs_cte)
    if not np.isfinite(imp):
        return
    sign_str = f"↓{abs(imp):.0f}%" if imp > 0 else f"↑{abs(imp):.0f}%"
    text  = f"CTE {sign_str}"
    facecolor = "#eaf6ec" if imp > 0 else "#faeaea"
    edgecolor = "#2a7a40" if imp > 0 else "#a01020"
    fontcolor = "#1a5c2e" if imp > 0 else "#8b1010"
    ax.text(0.975, 0.04, text,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.6, fontweight="bold", color=fontcolor,
            bbox=dict(facecolor=facecolor, edgecolor=edgecolor,
                      alpha=0.90, pad=0.8, linewidth=0.6,
                      boxstyle="round,pad=0.25"))


def _draw_panel(ax, all_results, dist_name, path_name, wps, cfg,
                show_legend=False, show_xlabel=False, show_ylabel=False):
    """在给定坐标轴上绘制三方法轨迹对比。

    使用自定义小尺寸标记（_plot_wps / _draw_final_pos），
    并在面板右下角标注 SHCS 相对 ILOS-PID 的 CTE 改善幅度。
    """
    goal_tol = float(cfg["simulation"].get("goal_tolerance", 3.0))
    _plot_wps(ax, wps, goal_tol)

    for midx, mn in enumerate(METHOD_NAMES):
        key = (dist_name, path_name, mn)
        if key not in all_results:
            continue
        res = all_results[key]
        c, ls = get_method_style(mn, midx)
        lw = 1.1 if mn == "shcs" else 0.65
        zo = 5 if mn == "shcs" else 3
        ax.plot(res["log"]["x"], res["log"]["y"],
                color=c, lw=lw, ls=ls, zorder=zo)
        _draw_final_pos(ax, res["log"], c)

    # # 改善量标注（仅在已有仿真结果时）
    # if all_results:
    #     _annotate_improvement(ax, all_results, dist_name, path_name)

    # 6% 均匀留白，先于 set_aspect 调用，避免"Ignoring fixed limits"警告
    ax.margins(0.06)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x / m" if show_xlabel else "")
    ax.set_ylabel("y / m" if show_ylabel else "")

    if show_legend:
        h, lbl = _legend_handles(cfg)
        ax.legend(h, lbl, loc="upper left", ncol=1,
                  fontsize=5.4, handlelength=1.5,
                  framealpha=0.88, edgecolor="#c8d0d8",
                  borderpad=0.35, handletextpad=0.4)


# ─────────────────────────────────────────────────────────────────────────────
# 轨迹组图（2 工况 × 4 路径）
# ─────────────────────────────────────────────────────────────────────────────

def make_composite(all_results, cfg) -> plt.Figure:
    """生成轨迹组图（主图）。

    行 = 工况（DIST_SHOW），列 = 路径（PATH_NAMES_V2）。
    列宽按路径空间跨度等比设置，保证各子图米/英寸比例一致。
    图例仅在左上第一格显示，行标签用 fig.text() 旋转放置。
    """
    apply_plot_style("composite")

    n_rows = len(DIST_SHOW)
    n_cols = len(PATH_NAMES_V2)

    # 每行高度因子 0.37，使子图纵横比适当
    fig = plt.figure(figsize=heu_figsize("large", 0.37 * n_rows))
    gs  = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        width_ratios=_WIDTH_RATIOS,
        hspace=0.30, wspace=0.26,
        left=0.09, right=0.99, bottom=0.07, top=0.92,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)]
            for r in range(n_rows)]

    panel_idx = 0
    for row_idx, dist_name in enumerate(DIST_SHOW):
        for col_idx, path_name in enumerate(PATH_NAMES_V2):
            ax  = axes[row_idx][col_idx]
            wps = PATHS_V2[path_name]["wps"]

            is_first    = (row_idx == 0 and col_idx == 0)
            is_last_row = (row_idx == n_rows - 1)
            is_left_col = (col_idx == 0)

            _draw_panel(ax, all_results, dist_name, path_name, wps, cfg,
                        show_legend=is_first,
                        show_xlabel=is_last_row,
                        show_ylabel=is_left_col)

            # # 子图序号（左上角白底小标签）
            # letter  = chr(ord("a") + panel_idx)
            # label_y = 0.72 if is_first else 0.98
            # ax.text(0.02, label_y, f"({letter})",
            #         transform=ax.transAxes, ha="left", va="top",
            #         fontsize=7.2, fontweight="bold",
            #         bbox=dict(facecolor="white", edgecolor="none",
            #                   alpha=0.82, pad=0.9))
            # 子图序号（底部居中，无底色，纯文字）
            letter  = chr(ord("a") + panel_idx)
            # ax.text(0.5, -0.18, f"({letter})",
            #         transform=ax.transAxes, ha="center", va="top",
            #         fontsize=7.2, fontweight="bold")
            # 关键：最后一行有 X 标签 → 序号放更低；第一行放正常
            if is_last_row:
                # 最后一行（有x轴标签）→ 放更下面，避开文字
                ax.text(0.5, -0.25, f"({letter})",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=7.2, fontweight="bold")
            else:
                # 上面几行（无x轴标签）→ 放正常位置
                ax.text(0.5, -0.15, f"({letter})",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=7.2, fontweight="bold")

            # 列标题（路径名，仅首行）
            if row_idx == 0:
                ax.set_title(PATHS_V2[path_name]["label"],
                             fontweight="bold", fontsize=7.5, pad=5.5)

            panel_idx += 1

    # 行标签：从 GridSpec 几何直接获取位置，无需渲染
    for row_idx, dist_name in enumerate(DIST_SHOW):
        ss = gs[row_idx, 0]
        x0, y0, x1, y1 = ss.get_position(fig).extents
        fig.text(0.01, (y0 + y1) / 2, DIST_LABELS[dist_name],
                 rotation=90, ha="center", va="center",
                 fontsize=7.2, fontweight="bold")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 路径概览图（辅助图，不含仿真轨迹）
# ─────────────────────────────────────────────────────────────────────────────

def make_path_overview(cfg) -> plt.Figure:
    """生成 4 条参考路径的概览图，标注航点编号。

    用于在论文补充材料或演示中直观展示路径几何特征。
    """
    apply_plot_style("composite")

    goal_tol = float(cfg["simulation"].get("goal_tolerance", 3.0))
    n_cols   = len(PATH_NAMES_V2)

    fig = plt.figure(figsize=heu_figsize("large", 0.26))
    gs  = gridspec.GridSpec(
        1, n_cols, figure=fig,
        width_ratios=_WIDTH_RATIOS,
        hspace=0, wspace=0.24,
        left=0.06, right=0.99, bottom=0.12, top=0.84,
    )

    for col_idx, pn in enumerate(PATH_NAMES_V2):
        ax  = fig.add_subplot(gs[0, col_idx])
        wps = PATHS_V2[pn]["wps"]
        _plot_wps(ax, wps, goal_tol)
        ax.margins(0.12)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(PATHS_V2[pn]["label"], fontweight="bold",
                     fontsize=7.4, pad=4.0)
        ax.set_xlabel("x / m", fontsize=6.5)
        if col_idx == 0:
            ax.set_ylabel("y / m", fontsize=6.5)

        # 航点编号标注
        for wi, wp in enumerate(wps):
            ax.annotate(f"P{wi}", xy=(wp[0], wp[1]),
                        fontsize=5.5, color="#555",
                        ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")

    fig.suptitle("实验路径概览（4 条测试路径）",
                 fontweight="bold", fontsize=7.8, y=0.98)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 独立轨迹子图存档
# ─────────────────────────────────────────────────────────────────────────────

def save_subfigures(all_results, cfg, out_dir: Path) -> None:
    """将各工况×路径的轨迹子图分别存为独立 PNG。

    图形尺寸根据路径空间跨度动态调整，使各图比例协调。
    """
    apply_plot_style("composite")
    out_dir.mkdir(parents=True, exist_ok=True)

    for dist_idx, dist_name in enumerate(DIST_ALL):
        for path_idx, path_name in enumerate(PATH_NAMES_V2):
            wps    = PATHS_V2[path_name]["wps"]
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
                f"{PATHS_V2[path_name]['label']} — {DIST_LABELS[dist_name]}",
                fontweight="bold", loc="left", fontsize=7.4)
            save_fig(fig, out_dir / f"v2_{dist_name}_{path_name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# 指标汇总组图（2 指标 × 2 工况）
# ─────────────────────────────────────────────────────────────────────────────

def make_metrics_figure(all_results, cfg) -> plt.Figure:
    """生成性能指标汇总组图。

    布局：2 行（CTE RMS / 偏航能耗）× 2 列（无扰动 / 定常海流）。
    每格：4 路径为 x 轴，3 方法分组条形图；SHCS 加粗黑边突出。
    数值标签仅标注在 SHCS 条形顶端，避免过密。
    """
    apply_plot_style("composite")

    metric_defs = [
        ("cross_track_rms",          "CTE RMS / m"),
        ("control_energy_tau_r_cmd", "Yaw Energy"),
    ]
    n_m = len(metric_defs)
    n_d = len(DIST_SHOW)

    fig = plt.figure(figsize=heu_figsize("large", 0.78))
    gs  = gridspec.GridSpec(n_m, n_d, figure=fig)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_d)]
            for r in range(n_m)]
    fig.subplots_adjust(left=0.10, right=0.99, top=0.94, bottom=0.12,
                        hspace=0.56, wspace=0.20)

    x        = np.arange(len(PATH_NAMES_V2))
    width    = 0.22
    offsets  = np.array([-1, 0, 1]) * width
    colors   = [get_method_style(mn, i)[0] for i, mn in enumerate(METHOD_NAMES)]
    mlabels  = [cfg["methods"][mn]["label"] for mn in METHOD_NAMES]
    YELLOW   = "#F0E442"
    edge_cs  = [
        "#1a1a1a" if mn == "shcs"
        else "#777" if colors[i] == YELLOW
        else "white"
        for i, mn in enumerate(METHOD_NAMES)
    ]
    edge_ws  = [1.4 if mn == "shcs" else 0.5 for mn in METHOD_NAMES]
    short_x  = [PATHS_V2[p]["short"] for p in PATH_NAMES_V2]

    panel_idx = 0
    for row_idx, (field, ylabel) in enumerate(metric_defs):
        for col_idx, dist_name in enumerate(DIST_SHOW):
            ax = axes[row_idx][col_idx]

            # 收集各方法在各路径下的指标值
            all_vals, method_vals = [], []
            for mn in METHOD_NAMES:
                vals = []
                for pn in PATH_NAMES_V2:
                    res = all_results.get((dist_name, pn, mn))
                    vals.append(
                        float(res["summary"].get(field, float("nan")))
                        if res else float("nan")
                    )
                method_vals.append(vals)
                all_vals.extend(v for v in vals if not np.isnan(v))
            y_max = max(all_vals) if all_vals else 1.0

            for midx, (mn, color, ec, ew) in enumerate(
                    zip(METHOD_NAMES, colors, edge_cs, edge_ws)):
                vals  = method_vals[midx]
                label = mlabels[midx] if panel_idx == 0 else "_nolegend_"
                ax.bar(x + offsets[midx], vals, width,
                       color=color, label=label,
                       edgecolor=ec, linewidth=ew)
                # 仅 SHCS 标注数值（避免过密）
                if mn == "shcs":
                    for xi, val in enumerate(vals):
                        if not np.isnan(val):
                            fmt = f"{val:.2f}" if y_max < 5 else f"{val:.0f}"
                            ax.text(x[xi] + offsets[midx], val + y_max * 0.03,
                                    fmt, ha="center", va="bottom",
                                    fontsize=5.2, fontweight="bold",
                                    color="#1a1a1a")

            ax.set_xticks(x)
            ax.set_xticklabels(short_x)
            ax.set_ylim(0, y_max * 1.32)
            ax.set_ylabel(ylabel if col_idx == 0 else "")
            ax.grid(axis="y", alpha=0.28)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # 首行顶部标题
            if row_idx == 0:
                ax.set_title(DIST_LABELS[dist_name],
                             fontweight="bold", fontsize=7.5)

            # 底部 caption
            letter = chr(ord("a") + panel_idx)
            ax.text(0.5, -0.24, f"({letter}) {DIST_LABELS[dist_name]}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontweight="bold",
                    fontsize=plt.rcParams.get("axes.titlesize", 7),
                    clip_on=False)

            if panel_idx == 0:
                ax.legend(loc="upper right", ncol=1, fontsize=6,
                          framealpha=0.88, edgecolor="#c8d0d8")
            panel_idx += 1

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CSV 汇总表
# ─────────────────────────────────────────────────────────────────────────────

def make_summary_table(all_results, cfg) -> list:
    """生成 CSV 汇总行：各方法数据行 + SHCS vs ILOS-PID 改善量行。"""
    rows = []
    for dist_name in DIST_ALL:
        for pn in PATH_NAMES_V2:
            base_res = all_results.get((dist_name, pn, "los_pid_short"))
            shcs_res = all_results.get((dist_name, pn, "shcs"))
            if base_res is None or shcs_res is None:
                continue
            # 各方法数据行
            for mn in METHOD_NAMES:
                res = all_results.get((dist_name, pn, mn))
                if res is None:
                    continue
                s = res["summary"]
                row = {
                    "Disturbance": DIST_LABELS.get(dist_name, dist_name),
                    "Path":        PATHS_V2[pn]["label"],
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
                "Path":        PATHS_V2[pn]["label"],
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
        for pn in PATH_NAMES_V2:
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
                  f"{PATHS_V2[pn]['label']:<20} "
                  f"{cte_i:>+7.1f}%  {turn_i:>+9.1f}%  {yaw_i:>+7.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 0. 加载配置 ──────────────────────────────────────────────────────────
    cfg       = get_config()
    eta0, nu0 = get_initial_state(cfg, "path_generalization")

    total = len(DIST_ALL) * len(PATH_NAMES_V2) * len(METHOD_NAMES)
    print(f"{'='*65}")
    print(f"  多路径泛化 v2  "
          f"({len(PATH_NAMES_V2)} 路径 × {len(DIST_ALL)} 工况 "
          f"× {len(METHOD_NAMES)} 方法 = {total} 组)")
    print(f"{'='*65}")

    # ── 1. 仿真（遍历工况 × 路径 × 方法）─────────────────────────────────
    all_results: dict = {}
    done = 0
    for dist_name in DIST_ALL:
        dist_cfg = cfg["disturbances"][dist_name]
        for pn in PATH_NAMES_V2:
            wps = PATHS_V2[pn]["wps"]
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
    csv_path = OUT_DIR / "generalization_v3_summary.csv"
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

    # ── 6. 保存轨迹组图 ───────────────────────────────────────────────────────
    comp_path = OUT_DIR / "fig_traj_composite.png"
    print(f"\n  保存轨迹组图 → {comp_path}")
    save_fig(make_composite(all_results, cfg), comp_path)

    # ── 7. 保存指标组图 ───────────────────────────────────────────────────────
    met_path = OUT_DIR / "fig_metrics.png"
    print(f"\n  保存指标组图 → {met_path}")
    save_fig(make_metrics_figure(all_results, cfg), met_path)

    print("\n  多路径泛化 v3 全部完成。")


if __name__ == "__main__":
    main()
