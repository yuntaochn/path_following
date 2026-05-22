"""
实验 4.2（v2）：统一前视距离（Δ=4 m）典型场景四方法对比

与 exp_01_typical_l_shape.py 的区别：
  - ILOS-PID 基线改用 los_pid_short（Δ=4 m），与 SHCS/FR/FO 保持相同前视距离，
    使方法间差异完全来自参考整形机制，而非前视距离选择。

对比方法（均使用 Δ=4 m）：
  1. ILOS-PID  (Δ=4 m)  — 短前视基线，体现航段切换处的饱和与卷绕问题
  2. ILOS+FR   (Δ=4 m)  — 固定速率限幅整形，消除阶跃但牺牲 CTE
  3. ILOS+FO   (Δ=4 m)  — 一阶参考模型整形，过渡更平滑但 CTE 代价仍大
  4. SHCS      (Δ=4 m)  — 动态整形 + 推进器约束感知调速（本文方法）

场景：L 形路径（90° 单次转弯）+ 定常海流扰动

输出结构：
  results/baseline/01b_equal_lookahead/
    {method}_log.npz / {method}_summary.json
    compare_summary.csv
    fig2_composite.png
    subfigs/
      fig2a_trajectory.png  ...  fig2f_thruster_margin.png
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
    run_trial, save_result, save_summaries_csv,
    apply_plot_style, get_method_style, print_metrics_table,
    relative_improvement, RESULTS_ROOT, get_initial_state,
    save_fig, draw_final_position, plot_waypoints, heu_figsize,
    shade_saturation_windows,
)

# ─────────────────────────────────────────────────────────────────────────────
# 实验配置
# ─────────────────────────────────────────────────────────────────────────────

OUT_DIR         = RESULTS_ROOT / "01b_equal_lookahead"
# los_pid_short = ILOS-PID with Δ=4 m（与 SHCS/FR/FO 一致）
METHOD_NAMES    = ["los_pid_short", "fixed_rate", "first_order", "shcs"]
PATH_NAME       = "l_shape"
DIST_NAME       = "steady_current"
SEED            = 1
EXPERIMENT_NAME = "compare_methods"   # 初始状态与 exp_01 相同

# 航向指标统一使用 heading_los_error_rms（几何 LOS 误差），
# 可在所有方法间公平横向比较，不受整形层影响。
TABLE_FIELDS = [
    ("cross_track_rms",           "CTE_RMS_m",          4),
    ("turn_cte_rms",              "TurnCTE_RMS_m",       4),
    ("turn_cte_peak_abs",         "TurnCTE_Peak_m",      4),
    ("heading_los_error_rms",     "HdgLOS_RMS_rad",      5),
    ("turn_heading_los_peak_abs", "TurnHdgLOS_Peak_rad", 5),
    ("control_energy_tau_r_cmd",  "YawEnergy",           2),
    ("speed_reduction_max_pct",   "MaxSpeedRed_pct",     2),
    ("sat_time_raw",              "SatTime_s",           2),
    ("reach_time",                "ReachTime_s",         2),
]

# ─────────────────────────────────────────────────────────────────────────────
# 各面板绘制函数
# ─────────────────────────────────────────────────────────────────────────────

def _lw(name: str) -> float:
    return 1.6 if name == "shcs" else 1.0

def _zo(name: str) -> int:
    return 5 if name == "shcs" else 3


def _draw_traj(ax, results, labels, names, wps, goal_tol):
    plot_waypoints(ax, wps, goal_tol=goal_tol, ref_label="Ref.")
    for idx, (res, lbl, name) in enumerate(zip(results, labels, names)):
        c, ls = get_method_style(name, idx)
        ax.plot(res["log"]["x"], res["log"]["y"],
                color=c, lw=_lw(name), ls=ls, label=lbl, zorder=_zo(name))
        draw_final_position(ax, res["log"], c)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.legend(
        loc="best", ncol=2, fontsize=5.5,
        framealpha=0.90, edgecolor="#c8d0d8",
        handlelength=1.4, handletextpad=0.4,
        borderpad=0.4, labelspacing=0.18,
    )


def _draw_heading(ax, results, names):
    """几何 LOS 误差（e_psi_los = psi_d - psi），各方法公平可比。"""
    for idx, (res, name) in enumerate(zip(results, names)):
        c, ls = get_method_style(name, idx)
        key = "e_psi_los" if "e_psi_los" in res["log"] else "e_psi"
        ax.plot(res["log"]["t"], np.degrees(res["log"][key]),
                color=c, lw=_lw(name), ls=ls, zorder=_zo(name))
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("t / s")
    ax.set_ylabel("LOS Heading Error / (°)")


def _draw_cte(ax, results, names):
    for idx, (res, name) in enumerate(zip(results, names)):
        c, ls = get_method_style(name, idx)
        ax.plot(res["log"]["t"], res["log"]["e_ct"],
                color=c, lw=_lw(name), ls=ls, zorder=_zo(name))
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel("t / s")
    ax.set_ylabel("Cross-Track Error / m")


def _draw_taur(ax, results, names, tau_r_lim):
    """偏航力矩时间历程，对 ILOS-PID(Δ=4m) 标注饱和区间。"""
    for res, name in zip(results, names):
        if name == "los_pid_short":   # Δ=4m 基线
            shade_saturation_windows(
                ax,
                np.asarray(res["log"]["t"]),
                np.asarray(res["log"]["tau_r_cmd_raw"]),
                tau_r_lim, color="#C86A5A", alpha=0.16,
            )
    for idx, (res, name) in enumerate(zip(results, names)):
        c, ls = get_method_style(name, idx)
        ax.plot(res["log"]["t"], res["log"]["tau_r_cmd_raw"],
                color=c, lw=_lw(name), ls=ls, zorder=_zo(name))
    ax.axhline( tau_r_lim, color="#B24A3B", lw=0.9, ls="--", alpha=0.82)
    ax.axhline(-tau_r_lim, color="#B24A3B", lw=0.9, ls="--", alpha=0.82)
    ax.set_xlabel("t / s")
    ax.set_ylabel("Raw Yaw Torque / (N·m)")


def _draw_surge(ax, results, names):
    for idx, (res, name) in enumerate(zip(results, names)):
        c, ls = get_method_style(name, idx)
        ax.plot(res["log"]["t"], res["log"]["u"],
                color=c, lw=_lw(name), ls=ls, zorder=_zo(name))
        ax.plot(res["log"]["t"], res["log"]["u_d"],
                color=c, lw=0.55, ls=":", alpha=0.5)
    ax.set_xlabel("t / s")
    ax.set_ylabel("Surge Speed / (m/s)")


def _draw_margin(ax, results, names, T_max, b):
    tau_u_max = 2.0 * T_max
    for idx, (res, name) in enumerate(zip(results, names)):
        c, ls = get_method_style(name, idx)
        tau_r  = np.abs(np.asarray(res["log"]["tau_r_cmd"]))
        margin = np.maximum(0.0, (tau_u_max - (2.0 / b) * tau_r) / tau_u_max * 100.0)
        ax.plot(res["log"]["t"], margin,
                color=c, lw=_lw(name), ls=ls, zorder=_zo(name))
    ax.axhline(0, color="#B24A3B", lw=0.9, ls="--", alpha=0.82)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("t / s")
    ax.set_ylabel("Thruster Margin / %")


# ─────────────────────────────────────────────────────────────────────────────
# 组图与子图
# ─────────────────────────────────────────────────────────────────────────────

def make_composite(results, labels, names, wps, cfg) -> plt.Figure:
    apply_plot_style("composite")

    goal_tol  = float(cfg["simulation"].get("goal_tolerance", 3.0))
    T_max     = cfg["actuator"]["T_max"]
    b         = cfg["actuator"]["b"]
    tau_r_lim = T_max * b
    wps       = np.asarray(wps)

    fig = plt.figure(figsize=heu_figsize("large", 0.72), layout="constrained")
    gs  = gridspec.GridSpec(2, 3, figure=fig)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    ax_traj, ax_hdg, ax_cte, ax_taur, ax_surge, ax_margin = axes

    _draw_traj(ax_traj, results, labels, names, wps, goal_tol)
    _draw_heading(ax_hdg, results, names)
    _draw_cte(ax_cte, results, names)
    _draw_taur(ax_taur, results, names, tau_r_lim)
    _draw_surge(ax_surge, results, names)
    _draw_margin(ax_margin, results, names, T_max, b)

    panel_captions = [
        "(a) 轨迹", "(b) LOS航向误差", "(c) 横向误差",
        "(d) 偏航力矩",  "(e) 纵荡速度",    "(f) 推进器余量",
    ]
    fs = plt.rcParams["axes.titlesize"]
    for ax, caption in zip(axes, panel_captions):
        ax.text(0.5, -0.25, caption,
                transform=ax.transAxes, ha="center", va="top",
                fontweight="normal", fontsize=fs, clip_on=False)

    return fig


def save_subfigures(results, labels, names, wps, cfg, out_dir: Path) -> None:
    apply_plot_style("composite")
    out_dir.mkdir(parents=True, exist_ok=True)

    goal_tol  = float(cfg["simulation"].get("goal_tolerance", 3.0))
    T_max     = cfg["actuator"]["T_max"]
    b         = cfg["actuator"]["b"]
    tau_r_lim = T_max * b
    wps       = np.asarray(wps)

    fig, ax = plt.subplots(figsize=heu_figsize("small", 0.85), layout="constrained")
    _draw_traj(ax, results, labels, names, wps, goal_tol)
    # ax.set_title("(a) 轨迹", fontweight="bold", loc="left")
    save_fig(fig, out_dir / "fig2a_trajectory.png")

    fig, ax = plt.subplots(figsize=heu_figsize("small", 0.75), layout="constrained")
    _draw_heading(ax, results, names)
    # ax.set_title("(b) LOS航向误差", fontweight="bold", loc="left")
    save_fig(fig, out_dir / "fig2b_heading_error.png")

    fig, ax = plt.subplots(figsize=heu_figsize("small", 0.75), layout="constrained")
    _draw_cte(ax, results, names)
    # ax.set_title("(c) 横向误差", fontweight="bold", loc="left")
    save_fig(fig, out_dir / "fig2c_cross_track_error.png")

    fig, ax = plt.subplots(figsize=heu_figsize("small", 0.75), layout="constrained")
    _draw_taur(ax, results, names, tau_r_lim)
    # ax.set_title("(d) 偏航力矩（阴影为饱和）", fontweight="bold", loc="left")
    save_fig(fig, out_dir / "fig2d_yaw_torque.png")

    fig, ax = plt.subplots(figsize=heu_figsize("small", 0.75), layout="constrained")
    _draw_surge(ax, results, names)
    # ax.set_title("(e) 纵荡速度", fontweight="bold", loc="left")
    save_fig(fig, out_dir / "fig2e_surge_speed.png")

    fig, ax = plt.subplots(figsize=heu_figsize("small", 0.75), layout="constrained")
    _draw_margin(ax, results, names, T_max, b)
    # ax.set_title("(f) 推进器余量", fontweight="bold", loc="left")
    save_fig(fig, out_dir / "fig2f_thruster_margin.png")


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

def make_csv_rows(results, labels) -> list[dict]:
    rows = []
    for res, lbl in zip(results, labels):
        s = res["summary"]
        row = {
            "Method":     lbl,
            "Reached":    bool(s.get("reached_goal", False)),
            "GoalDist_m": round(float(s.get("final_dist_to_goal", float("nan"))), 3),
            "Completion": s.get("completion_reason", ""),
        }
        for field, col, prec in TABLE_FIELDS:
            val = s.get(field, float("nan"))
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
    cfg       = get_config()
    waypoints = get_path(PATH_NAME, cfg)
    dist_cfg  = cfg["disturbances"][DIST_NAME]
    eta0, nu0 = get_initial_state(cfg, EXPERIMENT_NAME)
    labels    = [cfg["methods"][m]["label"] for m in METHOD_NAMES]

    print(f"{'='*65}")
    print(f"  Sec 4.2 (v2) 统一前视距离对比  ({PATH_NAME} + {DIST_NAME})")
    print(f"  所有方法均使用 Δ=4 m，差异仅来自参考整形机制")
    print(f"{'='*65}")

    # ── 1. 运行仿真 ───────────────────────────────────────────────────────────
    results = []
    for name, lbl in zip(METHOD_NAMES, labels):
        print(f"  [{lbl}] ...", end="  ", flush=True)
        res = run_trial(name, cfg, waypoints, eta0, nu0, dist_cfg, SEED)
        results.append(res)
        s = res["summary"]
        t_show = s.get("reach_time", float("nan")) if s.get("reached_goal") else s.get("completion_time", float("nan"))
        print(f"完成={s.get('path_completed')}  到达={s['reached_goal']}  "
              f"t={t_show:.1f}s  CTE={s['cross_track_rms']:.3f}m  "
              f"LOS航向RMS={s.get('heading_los_error_rms', float('nan')):.4f}rad  "
              f"YawE={s['control_energy_tau_r_cmd']:.1f}  "
              f"SatT={s.get('sat_time_raw', 0.0):.2f}s")

    # ── 2. 打印全量指标表 ─────────────────────────────────────────────────────
    print_metrics_table(results, labels)

    # ILOS-PID(Δ=4m) 为第一个方法，SHCS 为最后一个
    base = results[0]["summary"]
    shcs = results[-1]["summary"]
    print(f"\n  SHCS vs ILOS-PID(Δ=4m)：")
    print(f"    CTE RMS  : {base['cross_track_rms']:.4f} → {shcs['cross_track_rms']:.4f}  "
          f"[{relative_improvement(base['cross_track_rms'], shcs['cross_track_rms']):+.1f}%]")
    print(f"    LOS航向  : {base.get('heading_los_error_rms', float('nan')):.4f} → "
          f"{shcs.get('heading_los_error_rms', float('nan')):.4f}  "
          f"[{relative_improvement(base.get('heading_los_error_rms', float('nan')), shcs.get('heading_los_error_rms', float('nan'))):+.1f}%]")
    print(f"    偏航能耗 : {base['control_energy_tau_r_cmd']:.1f} → {shcs['control_energy_tau_r_cmd']:.1f}  "
          f"[{relative_improvement(base['control_energy_tau_r_cmd'], shcs['control_energy_tau_r_cmd']):+.1f}%]")
    print(f"    饱和时间 : {base.get('sat_time_raw', 0.0):.2f}s → {shcs.get('sat_time_raw', 0.0):.2f}s")

    # ── 3. 保存数据 ───────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, res in zip(METHOD_NAMES, results):
        save_result(res, OUT_DIR, name)
    save_summaries_csv(make_csv_rows(results, labels), OUT_DIR / "compare_summary.csv")
    print(f"\n  对比 CSV: {OUT_DIR / 'compare_summary.csv'}")

    # ── 4. 保存子图 ───────────────────────────────────────────────────────────
    print(f"  保存子图 → {OUT_DIR / 'subfigs'}")
    save_subfigures(results, labels, METHOD_NAMES, waypoints, cfg, OUT_DIR / "subfigs")

    # ── 5. 保存组图 ───────────────────────────────────────────────────────────
    print(f"  保存组图 → {OUT_DIR / 'fig2_composite.png'}")
    fig = make_composite(results, labels, METHOD_NAMES, waypoints, cfg)
    save_fig(fig, OUT_DIR / "fig2_composite.png")

    print("\n  实验 4.2 (v2) 全部完成。")


if __name__ == "__main__":
    main()
