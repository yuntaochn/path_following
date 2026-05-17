"""
实验 4.4：随机化蒙特卡洛分析。

新版实验不再重复同一个含噪海流场景，而是构造一组随机化工况：

  - 海流速度在名义 steady/current 附近随机变化；
  - 力噪声、控制噪声、测量噪声强度按 case 随机缩放；
  - 初始横向误差和初始航向存在小扰动；
  - 每个 case 还包含小的恒定环境力偏置。

每个随机 case 对 ILOS-PID(同前视距离 Δ=4 m) 与 SHCS 成对复用，
因此统计结果反映的是同一工况下参考整形和速度调度带来的差异，而不是
单个噪声样本或前视距离差异造成的"漂亮数字"。
"""

from __future__ import annotations

import copy
import os
import sys
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

from baseline import get_config, get_path
from baseline.metrics import ci95, evaluate_metric
from scripts.experiment_utils import (
    RESULTS_ROOT,
    PAPER_TEXT,
    apply_cjk_text_fonts,
    apply_plot_style,
    get_initial_state,
    heu_figsize,
    save_fig,
    save_summaries_csv,
    run_trial,
)


OUT_DIR = RESULTS_ROOT / "03_monte_carlo"
PATH_NAME = "l_shape"
BASE_DISTURBANCE = "current"
EXPERIMENT_NAME = "monte_carlo"

# 使用同前视距离基线，避免把 Δ=10 m vs Δ=4 m 的差异写成蒙特卡洛收益。
METHOD_NAMES = ["los_pid_short", "shcs"]
METHOD_LABELS = {
    "los_pid_short": "ILOS-PID($\\Delta=4$ m)",
    "shcs": "SHCS",
}

N_CASES = 30
DESIGN_SEED = 20260514

CORE_METRICS = [
    ("cross_track_rms",           "横向误差RMS",           "CTE-RMS /m",           4),
    ("heading_los_error_rms",     "LOS航向误差RMS",           "LOS-RMS /rad",      5),  # 公平指标
    ("control_energy_tau_r_cmd",  "偏航控制能耗",           "Yaw-Energy / $(N^2\cdot m^2\cdot s)$",             2),
    ("sat_time_raw",              "饱和作用时间",           "Sat-Time /s",   3),
]


def _fmt_float(value: float, precision: int) -> str:
    return f"{value:.{precision}f}" if np.isfinite(value) else "n/a"


def _format_p_display(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def _scale_vec(values, scale: float) -> list[float]:
    return (np.asarray(values, dtype=float) * float(scale)).tolist()


def _sample_case(rng: np.random.Generator, idx: int, cfg: dict) -> dict:
    """生成一个随机化蒙特卡洛 case。

    随机范围有意保持在"可解释的工程扰动"内，而不是极端压力测试。
    """
    base_dist = cfg["disturbances"][BASE_DISTURBANCE]

    current = rng.normal(loc=[0.35, 0.18], scale=[0.08, 0.06])
    current[0] = np.clip(current[0], 0.20, 0.52)
    current[1] = np.clip(current[1], 0.04, 0.34)

    force_noise_scale = rng.uniform(0.75, 1.45)
    control_noise_scale = rng.uniform(0.75, 1.45)
    eta_noise_scale = rng.uniform(0.75, 1.35)
    nu_noise_scale = rng.uniform(0.75, 1.35)

    dist_cfg = copy.deepcopy(base_dist)
    dist_cfg["current_velocity"] = current.tolist()
    dist_cfg["force_noise_std"] = _scale_vec(base_dist.get("force_noise_std", [0, 0, 0]), force_noise_scale)
    dist_cfg["control_noise_std"] = _scale_vec(base_dist.get("control_noise_std", [0, 0, 0]), control_noise_scale)
    dist_cfg["eta_noise_std"] = _scale_vec(base_dist.get("eta_noise_std", [0, 0, 0]), eta_noise_scale)
    dist_cfg["nu_noise_std"] = _scale_vec(base_dist.get("nu_noise_std", [0, 0, 0]), nu_noise_scale)
    dist_cfg["force_bias"] = rng.normal(loc=[0.0, 0.0, 0.0], scale=[0.25, 0.0, 0.05]).tolist()

    return {
        "case_id": idx + 1,
        "seed": int(rng.integers(1, 1_000_000)),
        "initial_y_offset": float(rng.normal(0.0, 1.5)),
        "initial_heading_offset_rad": float(rng.normal(0.0, np.deg2rad(4.0))),
        "disturbance": dist_cfg,
        "current_x": float(current[0]),
        "current_y": float(current[1]),
        "force_noise_scale": float(force_noise_scale),
        "control_noise_scale": float(control_noise_scale),
        "eta_noise_scale": float(eta_noise_scale),
        "nu_noise_scale": float(nu_noise_scale),
        "force_bias_u": float(dist_cfg["force_bias"][0]),
        "force_bias_r": float(dist_cfg["force_bias"][2]),
    }


def _build_cases(cfg: dict) -> list[dict]:
    rng = np.random.default_rng(DESIGN_SEED)
    return [_sample_case(rng, i, cfg) for i in range(N_CASES)]


def _case_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        rows.append({
            "case_id": case["case_id"],
            "seed": case["seed"],
            "current_x": f"{case['current_x']:.4f}",
            "current_y": f"{case['current_y']:.4f}",
            "initial_y_offset": f"{case['initial_y_offset']:.4f}",
            "initial_heading_offset_deg": f"{np.rad2deg(case['initial_heading_offset_rad']):.3f}",
            "force_noise_scale": f"{case['force_noise_scale']:.4f}",
            "control_noise_scale": f"{case['control_noise_scale']:.4f}",
            "eta_noise_scale": f"{case['eta_noise_scale']:.4f}",
            "nu_noise_scale": f"{case['nu_noise_scale']:.4f}",
            "force_bias_u": f"{case['force_bias_u']:.4f}",
            "force_bias_r": f"{case['force_bias_r']:.4f}",
        })
    return rows


def _collect_statistics(mc_data: dict) -> list[dict]:
    rows = []
    base_name, shcs_name = METHOD_NAMES
    for field, short_label, table_label, precision in CORE_METRICS:
        base_vals = np.asarray(mc_data[base_name][field], dtype=float)
        shcs_vals = np.asarray(mc_data[shcs_name][field], dtype=float)
        delta = base_vals - shcs_vals
        change_pct = (np.mean(shcs_vals) - np.mean(base_vals)) / abs(np.mean(base_vals)) * 100

        b_mean, b_std, b_hw = ci95(base_vals.tolist())
        s_mean, s_std, s_hw = ci95(shcs_vals.tolist())
        stat = evaluate_metric(delta.tolist())
        p_value = float(stat["p_value"])

        rows.append({
            "field": field,
            "Metric": short_label,
            "TableMetric": table_label,
            "Baseline": METHOD_LABELS[base_name],
            "Baseline_mean": _fmt_float(b_mean, precision),
            "Baseline_std": _fmt_float(b_std, precision),
            "Baseline_CI95": f"±{_fmt_float(b_hw, precision)}",
            "SHCS_mean": _fmt_float(s_mean, precision),
            "SHCS_std": _fmt_float(s_std, precision),
            "SHCS_CI95": f"±{_fmt_float(s_hw, precision)}",
            "SHCS_change%": f"{change_pct:+.2f}%",
            "SHCS_lower_cases": f"{int(np.sum(shcs_vals < base_vals))}/{len(delta)}",
            "test": stat["test_name"],
            "p_value": f"{p_value:.6g}",
            "p_display": _format_p_display(p_value),
            "mean_delta_baseline_minus_shcs": _fmt_float(float(np.mean(delta)), precision),
        })
    return rows


def _print_stats(rows: list[dict]) -> None:
    print(f"\n{'-' * 112}")
    # print(f"  {'Metric':<14} {'ILOS-PID(Δ=4m)':<22} {'SHCS':<22}"
    #       f" {'change':>10} {'lower':>8} {'p':>8}")
    print(f"  {'Metric':<14} {'ILOS-PID(Δ=4m)':<22} {'SHCS':<22}"
          f" {'change':>10} {'lower':>8} {'p':>8}")
    print(f"{'-' * 112}")
    for row in rows:
        base = f"{row['Baseline_mean']}{row['Baseline_CI95']}"
        shcs = f"{row['SHCS_mean']}{row['SHCS_CI95']}"
        print(f"  {row['Metric']:<14} {base:<22} {shcs:<22}"
              f" {row['SHCS_change%']:>10} {row['SHCS_lower_cases']:>8}"
              f" {row['p_display']:>8}")
    print(f"{'-' * 112}")


def write_core_latex_table(rows: list[dict], path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\tabcaption{随机化蒙特卡洛统计结果（30 个随机工况，L 形路径 + 含噪海流）}",
        r"\label{tab:mc}",
        r"\journaltable",
        r"\begin{tabular}{@{}lCCCCC@{}}",
        r"\hline",
        r"指标 & ILOS-PID($\Delta=4$ m) & SHCS & SHCS变化 & 较低案例 & 配对检验 \\",
        r"\hline",
    ]
    for row in rows:
        change_tex = row["SHCS_change%"].replace("%", r"\%")
        p_tex = row["p_display"].replace("<", "$<$") if row["p_display"].startswith("<") else row["p_display"]
        lines.append(
            f"{row['TableMetric']} & "
            f"${row['Baseline_mean']}\\pm{row['Baseline_CI95'].lstrip('±')}$ & "
            f"${row['SHCS_mean']}\\pm{row['SHCS_CI95'].lstrip('±')}$ & "
            f"{change_tex} & "
            f"{row['SHCS_lower_cases']} & "
            f"{p_tex} \\\\"
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _metric_values(mc_data: dict, field: str) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(mc_data[METHOD_NAMES[0]][field], dtype=float)
    shcs = np.asarray(mc_data[METHOD_NAMES[1]][field], dtype=float)
    return base, shcs


def make_summary_figure(mc_data: dict, rows: list[dict]) -> plt.Figure:
    """配对斜率图（1×4 横向布局）。

    每条细线连接同一 case 在 ILOS-PID 和 SHCS 下的指标值：
      蓝色 = 该 case 中 SHCS 改善；橙色 = 该 case 中 SHCS 变差。
    粗线和菱形标记 ± 95% CI 表示均值趋势。
    """
    apply_plot_style("panel")

    panel_specs = [
        ("cross_track_rms",          "CTE-RMS",         "m"),
        ("heading_los_error_rms",    "LOS Hdg RMS",     "rad"),
        ("control_energy_tau_r_cmd", "Yaw Energy",      ""),
        ("sat_time_raw",             "Saturation time", "s"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=heu_figsize("large", 0.44))

    # 配色
    C_BETTER   = "#4E8FC0"   # 该 case SHCS 改善（蓝色线）
    C_WORSE    = "#D9704D"   # 该 case SHCS 变差（橙色线）
    C_ILOS_DOT = "#374151"   # ILOS 侧均值菱形
    C_SHCS_DOT = "#8F3F34"   # SHCS 侧均值菱形

    X_L, X_R = 0.28, 0.72   # 两侧 x 坐标（轴域归一化）
    row_by_field = {row["field"]: row for row in rows}

    for idx, (ax, (field, title, ylabel)) in enumerate(zip(axes, panel_specs)):
        base, shcs = _metric_values(mc_data, field)

        # ── 1. 30 条配对斜率细线 ──────────────────────────────────────────
        for b_val, s_val in zip(base.tolist(), shcs.tolist()):
            c = C_BETTER if s_val < b_val else C_WORSE
            ax.plot([X_L, X_R], [b_val, s_val],
                    color=c, lw=0.75, alpha=0.28, zorder=2)
            ax.scatter([X_L, X_R], [b_val, s_val],
                       s=5.5, c=c, alpha=0.38, linewidths=0, zorder=3)

        # ── 2. 均值 ± CI 菱形与均值趋势线 ────────────────────────────────
        b_mean, _, b_hw = ci95(base.tolist())
        s_mean, _, s_hw = ci95(shcs.tolist())

        mean_c = C_BETTER if s_mean < b_mean else C_WORSE
        ax.plot([X_L, X_R], [b_mean, s_mean],
                color=mean_c, lw=2.2, alpha=0.88, zorder=5,
                solid_capstyle="round")
        ax.errorbar(X_L, b_mean, yerr=b_hw, fmt="D", ms=4.8,
                    color=C_ILOS_DOT, ecolor=C_ILOS_DOT,
                    elinewidth=1.3, capsize=3.5, zorder=7)
        ax.errorbar(X_R, s_mean, yerr=s_hw, fmt="D", ms=4.8,
                    color=C_SHCS_DOT, ecolor=C_SHCS_DOT,
                    elinewidth=1.3, capsize=3.5, zorder=7)

        # ── 3. 面板标题与统计标注 ─────────────────────────────────────────
        row = row_by_field[field]
        change_c = C_BETTER if s_mean < b_mean else C_WORSE
        ax.set_title(f"({chr(ord('a') + idx)}) {title}",
                     fontsize=7.2, fontweight="bold", pad=3.5)
        ax.text(0.97, 0.97,
                f"{row['SHCS_change%']}\n$p{row['p_display']}$",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=6.0, color=change_c, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.75, pad=0.6))

        # ── 4. 坐标轴格式 ────────────────────────────────────────────────
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([X_L, X_R])
        # ax.set_xticklabels(["ILOS\n$\\Delta\\!=\\!4$m", "SHCS"], fontsize=6.5)
        ax.set_xticklabels(["ILOS", "SHCS"], fontsize=6.5)
        ax.set_ylabel(ylabel, fontsize=7.0)
        ax.tick_params(axis="x", length=0, pad=3.0)
        ax.grid(True, axis="y", alpha=0.28, lw=0.5)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # Y 轴范围：上方留足空间容纳均值菱形与标注文字
        all_vals = np.concatenate([base, shcs])
        y_min, y_max = float(np.min(all_vals)), float(np.max(all_vals))
        y_rng = max(y_max - y_min, abs(y_max) * 0.02, 1e-6)
        lower = max(0.0, y_min - y_rng * 0.06) if y_min >= 0 else y_min - y_rng * 0.06
        ax.set_ylim(lower, y_max + y_rng * 0.22)

    apply_cjk_text_fonts(fig)
    fig.tight_layout(pad=0.55, h_pad=0.5, w_pad=1.1)
    return fig


def make_summary_figure_col(mc_data: dict, rows: list[dict]) -> plt.Figure:
    """单栏版配对斜率图（2×2 布局）。

    行 1 = 代价指标（CTE / LOS Hdg），行 2 = 收益指标（Yaw Energy / Sat Time）。
    图宽 = 单栏宽度（7 cm），可直接嵌入双栏正文。
    子图标题置于底部，纵轴标签置于纵轴旁（与 exp_01 composite 风格一致）。
    """
    apply_plot_style("panel")

    panel_specs = [
        ("cross_track_rms",          "(a) 横向误差RMS",         "CTE-RMS / m"),
        ("heading_los_error_rms",    "(b) LOS航向误差RMS",     "LOS-RMS / rad"),
        ("control_energy_tau_r_cmd", "(c) 偏航控制能耗",      "Yaw-Energy / $(N^2\cdot m^2\cdot s)$"),
        ("sat_time_raw",             "(d) 饱和作用时间", "Sat-Time / s"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=heu_figsize("small", 1.15),
                             constrained_layout=True)
    axes = axes.ravel()

    C_BETTER   = "#4E8FC0"
    C_WORSE    = "#D9704D"
    C_ILOS_DOT = "#374151"
    C_SHCS_DOT = "#8F3F34"
    X_L, X_R  = 0.25, 0.75
    row_by_field = {row["field"]: row for row in rows}

    for idx, (ax, (field, caption, ylabel)) in enumerate(zip(axes, panel_specs)):
        base, shcs = _metric_values(mc_data, field)

        # 30 条配对斜率细线
        for b_val, s_val in zip(base.tolist(), shcs.tolist()):
            c = C_BETTER if s_val < b_val else C_WORSE
            ax.plot([X_L, X_R], [b_val, s_val],
                    color=c, lw=0.6, alpha=0.25, zorder=2)
            ax.scatter([X_L, X_R], [b_val, s_val],
                       s=3.5, c=c, alpha=0.32, linewidths=0, zorder=3)

        # 均值 ± CI 与趋势线
        b_mean, _, b_hw = ci95(base.tolist())
        s_mean, _, s_hw = ci95(shcs.tolist())
        mean_c = C_BETTER if s_mean < b_mean else C_WORSE
        ax.plot([X_L, X_R], [b_mean, s_mean],
                color=mean_c, lw=2.0, alpha=0.88, zorder=5,
                solid_capstyle="round")
        ax.errorbar(X_L, b_mean, yerr=b_hw, fmt="D", ms=3.8,
                    color=C_ILOS_DOT, ecolor=C_ILOS_DOT,
                    elinewidth=1.0, capsize=2.5, zorder=7)
        ax.errorbar(X_R, s_mean, yerr=s_hw, fmt="D", ms=3.8,
                    color=C_SHCS_DOT, ecolor=C_SHCS_DOT,
                    elinewidth=1.0, capsize=2.5, zorder=7)

        # 改善量标注（右上角）
        row = row_by_field[field]
        change_c = C_BETTER if s_mean < b_mean else C_WORSE
        ax.text(0.97, 0.97,
                f"{row['SHCS_change%']}\n$p${row['p_display']}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=5.5, color=change_c, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.75, pad=0.4))

        # 坐标轴格式
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks([X_L, X_R])
        # ax.set_xticklabels(["ILOS\n$\\Delta\\!=\\!4$m", "SHCS"], fontsize=5.5)
        ax.set_xticklabels(["ILOS", "SHCS"], fontsize=5.5)
        ax.set_ylabel(ylabel, fontsize=6.0)
        ax.tick_params(axis="x", length=0, pad=2.0)
        ax.grid(True, axis="y", alpha=0.28, lw=0.5)
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # 在两个数据列位置加浅灰竖线，作为斜率线的视觉锚柱
        for x_pos in [X_L, X_R]:
            ax.axvline(x_pos, color="#BBBBBB", lw=0.9, zorder=1, alpha=0.7)

        all_vals = np.concatenate([base, shcs])
        y_min, y_max = float(np.min(all_vals)), float(np.max(all_vals))
        y_rng = max(y_max - y_min, abs(y_max) * 0.02, 1e-6)
        lower = max(0.0, y_min - y_rng * 0.06) if y_min >= 0 else y_min - y_rng * 0.06
        ax.set_ylim(lower, y_max + y_rng * 0.22)

        # 子图标题置于底部（与 exp_01 composite 风格一致）
        ax.text(0.5, -0.12, caption,
                transform=ax.transAxes, ha="center", va="top",
                fontweight="bold", fontsize=6.5, clip_on=False)

    apply_cjk_text_fonts(fig)
    return fig


def make_figure_only() -> None:
    """从已保存的 NPZ 快速重新生成主图，不重新运行仿真。"""
    npz_path = OUT_DIR / "mc_raw.npz"

    if not npz_path.exists():
        raise FileNotFoundError(
            f"原始数据未找到: {npz_path}\n请先运行完整仿真（不带 --regen-fig 参数）。"
        )

    raw = np.load(npz_path)
    mc_data: dict = {method: {} for method in METHOD_NAMES}
    for method in METHOD_NAMES:
        for field, *_ in CORE_METRICS:
            mc_data[method][field] = raw[f"{method}_{field}"].tolist()

    stats_rows = _collect_statistics(mc_data)
    _print_stats(stats_rows)

    apply_plot_style("panel")

    fig_wide = make_summary_figure(mc_data, stats_rows)
    save_fig(fig_wide, OUT_DIR / "fig4_mc_paired.png")
    print(f"  双栏版 -> {OUT_DIR / 'fig4_mc_paired.png'}")

    fig_col = make_summary_figure_col(mc_data, stats_rows)
    save_fig(fig_col, OUT_DIR / "fig4_mc_paired_col.png")
    print(f"  单栏版 -> {OUT_DIR / 'fig4_mc_paired_col.png'}")


def main() -> None:
    cfg = get_config()
    waypoints = get_path(PATH_NAME, cfg)
    eta0_base, nu0 = get_initial_state(cfg, EXPERIMENT_NAME)
    cases = _build_cases(cfg)

    print("=" * 78)
    print(f"  Sec 4.4 randomized Monte Carlo ({PATH_NAME}, N={N_CASES} paired cases)")
    print("  Baseline: ILOS-PID with the same lookahead as SHCS (Δ=4 m)")
    print("=" * 78)

    mc_data: dict = {method: {field: [] for field, *_ in CORE_METRICS} for method in METHOD_NAMES}
    raw_rows: list[dict] = []

    for case in cases:
        eta0 = np.asarray(eta0_base, dtype=float).copy()
        eta0[1] += case["initial_y_offset"]
        eta0[2] += case["initial_heading_offset_rad"]

        summaries = {}
        for method in METHOD_NAMES:
            result = run_trial(
                method,
                cfg,
                waypoints,
                eta0,
                nu0,
                case["disturbance"],
                case["seed"],
            )
            summaries[method] = result["summary"]
            for field, *_ in CORE_METRICS:
                mc_data[method][field].append(float(result["summary"].get(field, np.nan)))

        b = summaries[METHOD_NAMES[0]]
        s = summaries[METHOD_NAMES[1]]
        print(
            f"  case={case['case_id']:02d} seed={case['seed']:6d} "
            f"current=({case['current_x']:.2f},{case['current_y']:.2f}) "
            f"CTE {b['cross_track_rms']:.3f}->{s['cross_track_rms']:.3f} "
            f"YawE {b['control_energy_tau_r_cmd']:.1f}->{s['control_energy_tau_r_cmd']:.1f} "
            f"Sat {b['sat_time_raw']:.2f}->{s['sat_time_raw']:.2f}",
            flush=True,
        )

        raw_row = {
            "case_id": case["case_id"],
            "seed": case["seed"],
            "current_x": f"{case['current_x']:.4f}",
            "current_y": f"{case['current_y']:.4f}",
        }
        for method in METHOD_NAMES:
            raw_row[f"{method}_reached_goal"] = summaries[method].get("reached_goal", "")
            raw_row[f"{method}_reach_time"] = summaries[method].get("reach_time", np.nan)
            raw_row[f"{method}_final_dist_to_goal"] = summaries[method].get("final_dist_to_goal", np.nan)
            for field, *_ in CORE_METRICS:
                raw_row[f"{method}_{field}"] = summaries[method].get(field, np.nan)
        raw_rows.append(raw_row)

    stats_rows = _collect_statistics(mc_data)
    _print_stats(stats_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_summaries_csv(_case_rows(cases), OUT_DIR / "mc_case_design.csv")
    save_summaries_csv(raw_rows, OUT_DIR / "mc_case_metrics.csv")
    save_summaries_csv(stats_rows, OUT_DIR / "mc_core_statistics.csv")
    # 兼容旧文档中的完整统计文件名。
    save_summaries_csv(stats_rows, OUT_DIR / "mc_statistics.csv")
    write_core_latex_table(stats_rows, OUT_DIR / "mc_core_table.tex")

    np_save_dict = {
        f"{method}_{field}": np.asarray(mc_data[method][field], dtype=float)
        for method in METHOD_NAMES for field, *_ in CORE_METRICS
    }
    np_save_dict["case_ids"] = np.asarray([c["case_id"] for c in cases], dtype=int)
    np.savez_compressed(OUT_DIR / "mc_raw.npz", **np_save_dict)

    print("\n  Save Monte Carlo paired slope figure ->", OUT_DIR / "fig4_mc_paired.png")
    fig = make_summary_figure(mc_data, stats_rows)
    save_fig(fig, OUT_DIR / "fig4_mc_paired.png")

    print("  Save single-column figure ->", OUT_DIR / "fig4_mc_paired_col.png")
    fig_col = make_summary_figure_col(mc_data, stats_rows)
    save_fig(fig_col, OUT_DIR / "fig4_mc_paired_col.png")

    print("\n  Experiment 4.4 completed.")


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument(
        "--regen-fig", action="store_true",
        help="从已保存数据快速重绘主图，不重新运行仿真",
    )
    _args = _parser.parse_args()
    if _args.regen_fig:
        make_figure_only()
    else:
        main()
