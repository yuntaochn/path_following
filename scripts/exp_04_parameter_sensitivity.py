"""
实验 4.5：SHCS 关键参数敏感性分析。

本脚本在论文主实验一致的 L 形路径 + 定常海流工况下，扫描 SHCS 的两个
关键参数，并观察横向误差 RMS 与偏航控制能耗的变化趋势。

1. lambda_schedule，图中记为 lambda：
   速度调度器的降速强度系数。数值越大，航向整形残差较大时降速越明显。

2. r_nominal，图中记为 r_nom：
   动态航向整形器的名义偏航角速度上限。数值越小，航向参考越保守；
   数值越大，航向参考越激进、越接近未经整形的 LOS 指令。

投稿图采用“单栏窄图”布局：上下两个子图，每个子图对应一个参数，并用
双纵轴同时展示 CTE RMS 和偏航能耗。这样比横向 1 x 2 排列更适合双栏论文
中“每列放一张图”的排版方式，也比原 4 x 1 长图更紧凑。
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

# 将 matplotlib 缓存放到临时目录，避免无 GUI/服务器环境下的权限问题。
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from baseline import get_config, get_path
from scripts.experiment_utils import (
    PAPER_BLUE,
    PAPER_GRAY,
    PAPER_GRID,
    PAPER_ORANGE,
    RESULTS_ROOT,
    apply_cjk_text_fonts,
    apply_plot_style,
    get_initial_state,
    heu_figsize,
    run_trial,
    save_fig,
    save_summaries_csv,
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

OUT_DIR = RESULTS_ROOT / "04_parameter_sensitivity"

PATH_NAME = "l_shape"
DIST_NAME = "steady_current"
SEED = 1
EXPERIMENT_NAME = "sensitivity"

# 参数扫描范围：点数不宜过密，否则单栏图中标记会拥挤；这些取值覆盖了
# “过小-默认-过大”的关键区间，足够说明参数敏感性。
LAMBDA_VALS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
R_NOMINAL_VALS = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5]

# 敏感性图和 CSV 中使用的两个指标。
CTE_FIELD = "cross_track_rms"
ENERGY_FIELD = "control_energy_tau_r_cmd"
SENS_METRICS = [
    (CTE_FIELD, "横向误差RMS / m"),
    (ENERGY_FIELD, "偏航能耗 / (N^2 m^2 s)"),
]

# 与全局图件一致的低饱和色：蓝色表示误差，暖棕色表示能耗。
BLUE = PAPER_BLUE
ORANGE = PAPER_ORANGE
GRAY = PAPER_GRAY
GRID = PAPER_GRID


# ---------------------------------------------------------------------------
# 图形调节参数
# ---------------------------------------------------------------------------
#
# 这几项是后续最可能需要手动微调的参数，所以单独放在这里。论文排版后如果
# 感觉图太高、曲线太平或坐标过度放大，优先改这个区域，不必深入绘图函数。

# 单栏图宽度。3.35 inch 约等于 85 mm，接近常见双栏期刊的单栏宽。
FIG_SIZE = heu_figsize("large", 0.78)

# 自动坐标范围的比例留白。数值越小，趋势放得越大；数值越大，曲线周围留白
# 越多。
TIGHT_PAD_RATIO = 0.14

# lambda 的趋势本身很平缓。若坐标轴过紧，会在视觉上误导读者，以为 lambda
# 对性能影响很强；因此第一幅图单独使用更大的留白，让曲线更符合“不敏感”
# 的论文结论。若仍觉得过陡，可继续增大这两个值。
LAMBDA_PAD_RATIO = 0.55
LAMBDA_MIN_CTE_PAD = 0.0020
LAMBDA_MIN_ENERGY_PAD = 0.85

# 最小绝对留白，防止某个指标变化极小时被过度放大。lambda 对 CTE RMS 的
# 影响很小，所以这里专门设置下限，避免视觉上误导读者。
MIN_CTE_PAD = 0.00035
MIN_ENERGY_PAD = 0.35

# r_nominal 的低取值本来就是“过慢整形”的反例，应保留在坐标范围内以体现
# 参数敏感性；同时用较紧的留白让 0.7-2.5 rad/s 的平台区仍能看清。
MIN_RNOM_CTE_PAD = 0.045
MIN_RNOM_ENERGY_PAD = 4.0


# ---------------------------------------------------------------------------
# Simulation sweep
# ---------------------------------------------------------------------------

def sweep(
    cfg: dict,
    waypoints: np.ndarray,
    eta0: np.ndarray,
    nu0: np.ndarray,
    dist_cfg: dict,
    seed: int,
    param_name: str,
    values: Iterable[float],
    shaper_key: str | None = None,
    scheduler_key: str | None = None,
) -> dict[str, list[float]]:
    """执行单因素参数扫描。

    每轮扫描只改变一个参数，其余配置全部沿用论文基准配置。这样图中的趋势
    才能解释为“该参数导致的变化”，而不是多个控制器参数共同改变的结果。

    Args:
        cfg: 完整基准配置字典。
        waypoints: 参考路径点。
        eta0: 初始位姿 [x, y, psi]。
        nu0: 初始速度 [u, v, r]。
        dist_cfg: 当前工况的扰动配置。
        seed: 仿真随机种子。
        param_name: 需要修改的参数名。
        values: 参数扫描值。
        shaper_key: 若不为 None，则修改 cfg["shapers"][shaper_key]。
        scheduler_key: 若不为 None，则修改 cfg["velocity_schedulers"][...]。

    Returns:
        指标字典，列表顺序与 values 一致。
    """
    collected = {field: [] for field, _ in SENS_METRICS}

    for val in values:
        shaper_override = None
        scheduler_override = None

        if shaper_key is not None:
            # 深拷贝原始整形器配置，仅替换目标参数，避免污染后续试验。
            shaper_override = deepcopy(cfg["shapers"][shaper_key])
            shaper_override[param_name] = val

        if scheduler_key is not None:
            # lambda 属于 SHCS 速度调度器，因此这里修改 scheduler 配置块。
            scheduler_override = deepcopy(cfg["velocity_schedulers"][scheduler_key])
            scheduler_override[param_name] = val

        result = run_trial(
            "shcs",
            cfg,
            waypoints,
            eta0,
            nu0,
            dist_cfg,
            seed,
            shaper_override=shaper_override,
            scheduler_override=scheduler_override,
        )

        for field, _ in SENS_METRICS:
            collected[field].append(float(result["summary"].get(field, np.nan)))

    return collected


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _limits(values: Iterable[float], pad_ratio: float, min_pad: float) -> tuple[float, float]:
    """根据数据范围生成较适合投稿图的纵轴范围。

    坐标太宽会让真实趋势看起来像水平线；坐标太紧又会夸大小波动。这里采用
    “比例留白 + 最小绝对留白”的组合，让曲线变化可见，同时保留数值尺度。
    """
    vals = np.asarray(list(values), dtype=float)
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    span = max(hi - lo, 1e-12)
    pad = max(span * pad_ratio, min_pad)
    return lo - pad, hi + pad


def _annotate_default(ax: plt.Axes, x: float, text: str = "Default") -> None:
    """用竖向虚线标出论文其余实验采用的默认参数值。"""
    y0, y1 = ax.get_ylim()
    ax.axvline(x, color=GRAY, ls=":", lw=0.9, zorder=0)
    ax.text(
        x,
        y0 + 0.90 * (y1 - y0),
        text,
        ha="center",
        va="top",
        fontsize=6.4,
        color=GRAY,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.7},
    )


def _style_dual_axes(ax: plt.Axes, ax2: plt.Axes) -> None:
    """统一设置双纵轴子图样式。"""
    ax.grid(True, color=GRID, lw=0.45, alpha=0.74)
    ax.tick_params(axis="y", colors=BLUE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax.spines["left"].set_color(BLUE)
    ax2.spines["right"].set_color(ORANGE)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)


def _add_subcaption(ax: plt.Axes, text: str) -> None:
    """在子图下方添加中文子图标题。

    期刊排版中，图内标题容易占用数据区域；把子图标题放到横轴下方，可以让
    曲线区域更干净，也更接近中文期刊常见的子图说明形式。
    """
    ax.text(
        0.5,
        -0.24,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="bold",
    )


def _hide_subcaptions(fig: plt.Figure) -> None:
    """Hide panel captions before exporting standalone subfigures."""
    for ax in fig.axes:
        for text in ax.texts:
            if text.get_text().lstrip().startswith(("(", "（")):
                text.set_visible(False)


def _draw_sensitivity_panel(
    ax: plt.Axes,
    x_vals: list[float],
    cte_vals: list[float],
    energy_vals: list[float],
    default_x: float,
    xlabel: str,
    title: str,
    marker: str,
    cte_min_pad: float,
    energy_min_pad: float,
    pad_ratio: float = TIGHT_PAD_RATIO,
) -> tuple[plt.Line2D, plt.Line2D]:
    """绘制一个参数对应的双纵轴子图。"""
    ax2 = ax.twinx()

    cte_line, = ax.plot(
        x_vals,
        cte_vals,
        marker + "-",
        color=BLUE,
        ms=3.3,
        lw=1.2,
        label="CTE RMS",
        zorder=3,
    )
    energy_line, = ax2.plot(
        x_vals,
        energy_vals,
        marker + "--",
        color=ORANGE,
        ms=3.3,
        lw=1.2,
        label="Yaw Energy",
        zorder=3,
    )

    # 收紧坐标范围，让单栏小图里也能看清趋势。由于坐标刻度仍然完整保留，
    # 读者仍能判断变化幅度到底是小幅变化还是强敏感变化。
    ax.set_ylim(_limits(cte_vals, pad_ratio, cte_min_pad))
    ax2.set_ylim(_limits(energy_vals, pad_ratio, energy_min_pad))
    _annotate_default(ax, default_x)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CTE RMS / m", color=BLUE)
    ax2.set_ylabel(r"Yaw Energy / $(N^2\cdot m^2\cdot s)$", color=ORANGE)
    _style_dual_axes(ax, ax2)
    _add_subcaption(ax, title)

    return cte_line, energy_line


def make_composite(
    lambda_data: dict[str, list[float]],
    rnom_data: dict[str, list[float]],
    default_lambda: float,
    default_rnom: float,
) -> plt.Figure:
    """生成论文投稿用的单栏敏感性分析图。

    布局：
        (a) lambda_schedule 对 CTE RMS 和偏航能耗的影响；
        (b) r_nominal 对 CTE RMS 和偏航能耗的影响。

    设计原因：
        原 4 x 1 图适合屏幕检查，但放进双栏论文会过长。现在每个参数对应一个
        子图，两个评价指标通过双纵轴压缩到同一子图，读图顺序更自然。
    """
    apply_plot_style("sensitivity")

    fig, axes = plt.subplots(2, 1, figsize=FIG_SIZE, constrained_layout=False)
    # 手动控制边距：右侧需要容纳第二纵轴标签；上下子图之间需要给中文子图题
    # 留出空间。若插入论文后显得过松/过紧，可优先调 hspace。
    fig.subplots_adjust(left=0.22, right=0.78, top=0.95, bottom=0.12, hspace=0.42)

    lambda_handles = _draw_sensitivity_panel(
        axes[0],
        LAMBDA_VALS,
        lambda_data[CTE_FIELD],
        lambda_data[ENERGY_FIELD],
        default_lambda,
        r"$\lambda$",
        r"(a) 降速强度 $\lambda$ 的影响",
        "o",
        LAMBDA_MIN_CTE_PAD,
        LAMBDA_MIN_ENERGY_PAD,
        LAMBDA_PAD_RATIO,
    )
    _draw_sensitivity_panel(
        axes[1],
        R_NOMINAL_VALS,
        rnom_data[CTE_FIELD],
        rnom_data[ENERGY_FIELD],
        default_rnom,
        # r"$r_{\mathrm{nom}}$ / (rad$\cdot$s$^{-1}$)",
        r"$r_{\mathrm{nom}}$ / (rad/s)",
        r"(b) 名义偏航角速度上限 $r_{\mathrm{nom}}$ 的影响",
        "s",
        MIN_RNOM_CTE_PAD,
        MIN_RNOM_ENERGY_PAD,
    )

    # 图例放入第一个子图内部，避免占用整图上方空间。使用白底半透明边框，
    # 防止与网格线和曲线混在一起。
    axes[0].legend(
        lambda_handles,
        [h.get_label() for h in lambda_handles],
        loc="upper left",
        frameon=True,
        framealpha=0.86,
        facecolor="white",
        edgecolor="0.82",
        handlelength=1.8,
    )

    return fig


def _matching_twin_axis(fig: plt.Figure, primary_ax: plt.Axes) -> plt.Axes | None:
    """Return the twinned y-axis that occupies the same panel as primary_ax."""
    primary_bounds = primary_ax.get_position().bounds
    for candidate in fig.axes:
        if candidate is primary_ax:
            continue
        if np.allclose(candidate.get_position().bounds, primary_bounds):
            return candidate
    return None


def _save_axes_crop(
    fig: plt.Figure,
    axes: Iterable[plt.Axes],
    path: Path,
    dpi: int = 1200,
    pad_inches: float = 0.02,
    x_limits: tuple[float, float] | None = None,
) -> None:
    """Save the exact rendered area occupied by axes in an existing figure."""
    apply_cjk_text_fonts(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [ax.get_tightbbox(renderer) for ax in axes]
    bboxes = [bbox for bbox in bboxes if bbox is not None]
    if not bboxes:
        raise RuntimeError(f"No drawable axes bbox found for {path}")

    bbox_inches = Bbox.union(bboxes).transformed(fig.dpi_scale_trans.inverted())
    if x_limits is not None:
        bbox_inches = Bbox.from_extents(x_limits[0], bbox_inches.y0, x_limits[1], bbox_inches.y1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches.padded(pad_inches))
    print(f"  [保存] {path}")


def _save_composite_panel_crops(
    lambda_data: dict[str, list[float]],
    rnom_data: dict[str, list[float]],
    default_lambda: float,
    default_rnom: float,
    subfig_dir: Path,
) -> None:
    """Save the two combined panels by cropping the actual composite figure."""
    fig = make_composite(lambda_data, rnom_data, default_lambda, default_rnom)
    _hide_subcaptions(fig)
    primary_axes = [ax for ax in fig.axes if ax.get_ylabel() == "CTE RMS / m"]
    if len(primary_axes) != 2:
        plt.close(fig)
        raise RuntimeError(f"Expected 2 primary sensitivity panels, found {len(primary_axes)}")

    panel_names = ["fig5a_lambda_combined", "fig5b_rnom_combined"]
    panel_specs: list[tuple[str, list[plt.Axes]]] = []
    for filename, primary_ax in zip(panel_names, primary_axes):
        panel_axes: list[plt.Axes] = [primary_ax]
        twin_ax = _matching_twin_axis(fig, primary_ax)
        if twin_ax is not None:
            panel_axes.append(twin_ax)
        panel_specs.append((filename, panel_axes))

    apply_cjk_text_fonts(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_bboxes = [
        Bbox.union([ax.get_tightbbox(renderer) for ax in panel_axes])
        .transformed(fig.dpi_scale_trans.inverted())
        for _, panel_axes in panel_specs
    ]
    shared_x_limits = (
        min(bbox.x0 for bbox in panel_bboxes),
        max(bbox.x1 for bbox in panel_bboxes),
    )

    for filename, panel_axes in panel_specs:
        _save_axes_crop(
            fig,
            panel_axes,
            subfig_dir / f"{filename}.png",
            x_limits=shared_x_limits,
        )

    plt.close(fig)


def _draw_single_metric(
    ax: plt.Axes,
    x_vals: list[float],
    y_vals: list[float],
    default_x: float,
    xlabel: str,
    ylabel: str,
    title: str,
    color: str,
    marker: str,
    min_pad: float,
    pad_ratio: float,
    show_subcaption: bool = True,
) -> None:
    """绘制单指标子图。

    这些子图不作为正文主图的默认版本，而是作为备选材料保留：当你想单独检查
    某个指标，或在补充材料/汇报中拆开使用时，可以直接引用这些文件。
    """
    ax.plot(x_vals, y_vals, marker + "-", color=color, ms=3.6, lw=1.25, zorder=3)
    ax.set_ylim(_limits(y_vals, pad_ratio, min_pad))
    _annotate_default(ax, default_x)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, color=color)
    ax.tick_params(axis="y", colors=color)
    ax.spines["left"].set_color(color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, lw=0.45, alpha=0.74)
    if show_subcaption:
        _add_subcaption(ax, title)


def _make_single_metric_figure(
    x_vals: list[float],
    y_vals: list[float],
    default_x: float,
    xlabel: str,
    ylabel: str,
    title: str,
    color: str,
    marker: str,
    min_pad: float,
    pad_ratio: float,
    show_subcaption: bool = True,
) -> plt.Figure:
    """生成单栏宽的单指标子图 Figure。"""
    apply_plot_style("sensitivity")
    fig, ax = plt.subplots(1, 1, figsize=heu_figsize("small", 0.75), constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.96, top=0.94, bottom=0.30)
    _draw_single_metric(
        ax,
        x_vals,
        y_vals,
        default_x,
        xlabel,
        ylabel,
        title,
        color,
        marker,
        min_pad,
        pad_ratio,
        show_subcaption,
    )
    return fig


def save_subfigures(
    lambda_data: dict[str, list[float]],
    rnom_data: dict[str, list[float]],
    default_lambda: float,
    default_rnom: float,
) -> None:
    """保存拆分子图，作为合并总图之外的备选输出。

    输出目录：
        results/baseline/04_parameter_sensitivity/subfigs/

    其中最常用于论文排版的是两张“双纵轴合并子图”：
        fig5a_lambda_combined.*  lambda -> CTE RMS + yaw energy
        fig5b_rnom_combined.*    r_nom -> CTE RMS + yaw energy

    同时额外保留 4 张单指标子图，便于检查单个指标或制作补充材料：
        fig5a_lambda_cte.*      lambda -> CTE RMS
        fig5b_lambda_energy.*   lambda -> yaw energy
        fig5c_rnom_cte.*        r_nom -> CTE RMS
        fig5d_rnom_energy.*     r_nom -> yaw energy
    """
    subfig_dir = OUT_DIR / "subfigs"

    # 1) 直接从完整 2x1 组图中裁切两张双纵轴子图，保证轴区、字号和图例
    # 与组图中的对应面板一致；子图标题在导出前隐藏，方便论文中手动输入。
    _save_composite_panel_crops(
        lambda_data,
        rnom_data,
        default_lambda,
        default_rnom,
        subfig_dir,
    )

    # 2) 额外保存 4 张单指标子图，主要用于诊断和补充展示。
    subfig_specs = [
        (
            "fig5a_lambda_cte",
            LAMBDA_VALS,
            lambda_data[CTE_FIELD],
            default_lambda,
            r"$\lambda$",
            "CTE RMS / m",
            r"(a) $\lambda$ 对横向误差的影响",
            BLUE,
            "o",
            LAMBDA_MIN_CTE_PAD,
            LAMBDA_PAD_RATIO,
        ),
        (
            "fig5b_lambda_energy",
            LAMBDA_VALS,
            lambda_data[ENERGY_FIELD],
            default_lambda,
            r"$\lambda$",
            r"Yaw Energy / $(N^2\cdot m^2\cdot s)$",
            r"(b) $\lambda$ 对偏航能耗的影响",
            ORANGE,
            "o",
            LAMBDA_MIN_ENERGY_PAD,
            LAMBDA_PAD_RATIO,
        ),
        (
            "fig5c_rnom_cte",
            R_NOMINAL_VALS,
            rnom_data[CTE_FIELD],
            default_rnom,
            # r"$r_{\mathrm{nom}}$ / (rad$\cdot$s$^{-1}$)",
            r"$r_{\mathrm{nom}}$ / (rad/s)",
            "CTE RMS / m",
            r"(c) $r_{\mathrm{nom}}$ 对横向误差的影响",
            BLUE,
            "s",
            MIN_RNOM_CTE_PAD,
            TIGHT_PAD_RATIO,
        ),
        (
            "fig5d_rnom_energy",
            R_NOMINAL_VALS,
            rnom_data[ENERGY_FIELD],
            default_rnom,
            # r"$r_{\mathrm{nom}}$ / (rad$\cdot$s$^{-1}$)",
            r"$r_{\mathrm{nom}}$ / (rad/s)",
            r"Yaw Energy / $(N^2\cdot m^2\cdot s)$",
            r"(d) $r_{\mathrm{nom}}$ 对偏航能耗的影响",
            ORANGE,
            "s",
            MIN_RNOM_ENERGY_PAD,
            TIGHT_PAD_RATIO,
        ),
    ]

    for filename, x_vals, y_vals, default_x, xlabel, ylabel, title, color, marker, min_pad, pad_ratio in subfig_specs:
        fig = _make_single_metric_figure(
            x_vals,
            y_vals,
            default_x,
            xlabel,
            ylabel,
            title,
            color,
            marker,
            min_pad,
            pad_ratio,
            show_subcaption=False,
        )
        save_fig(fig, subfig_dir / f"{filename}.png")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _append_rows(all_csv_rows: list[dict], param: str, values: list[float], data: dict[str, list[float]]) -> None:
    """将一次参数扫描结果追加到 CSV 行缓存中。"""
    for i, value in enumerate(values):
        all_csv_rows.append({
            "param": param,
            "value": value,
            CTE_FIELD: round(data[CTE_FIELD][i], 5),
            ENERGY_FIELD: round(data[ENERGY_FIELD][i], 5),
        })


def _print_sweep_table(param_label: str, values: list[float], data: dict[str, list[float]]) -> None:
    """打印简洁表格，便于运行脚本后快速核对数值趋势。"""
    print(f"\n  {param_label:>8}  {'CTE_RMS':>10}  {'YawEnergy':>10}")
    for i, value in enumerate(values):
        print(f"  {value:8.3f}  {data[CTE_FIELD][i]:10.4f}  {data[ENERGY_FIELD][i]:10.2f}")


def main() -> None:
    cfg = get_config()
    waypoints = get_path(PATH_NAME, cfg)
    dist_cfg = cfg["disturbances"][DIST_NAME]
    eta0, nu0 = get_initial_state(cfg, EXPERIMENT_NAME)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Sec 4.5 sensitivity: lambda_schedule sweep ({len(LAMBDA_VALS)} values)")
    print("=" * 60)
    lambda_data = sweep(
        cfg,
        waypoints,
        eta0,
        nu0,
        dist_cfg,
        SEED,
        param_name="lambda_schedule",
        values=LAMBDA_VALS,
        scheduler_key="shcs",
    )
    _print_sweep_table("lambda", LAMBDA_VALS, lambda_data)

    print("\n" + "=" * 60)
    print(f"  Sec 4.5 sensitivity: r_nominal sweep ({len(R_NOMINAL_VALS)} values)")
    print("=" * 60)
    rnom_data = sweep(
        cfg,
        waypoints,
        eta0,
        nu0,
        dist_cfg,
        SEED,
        param_name="r_nominal",
        values=R_NOMINAL_VALS,
        shaper_key="dynamic",
    )
    _print_sweep_table("r_nom", R_NOMINAL_VALS, rnom_data)

    all_csv_rows: list[dict] = []
    _append_rows(all_csv_rows, "lambda_schedule", LAMBDA_VALS, lambda_data)
    _append_rows(all_csv_rows, "r_nominal", R_NOMINAL_VALS, rnom_data)
    save_summaries_csv(all_csv_rows, OUT_DIR / "sensitivity_summary.csv")
    print(f"\n  CSV: {OUT_DIR / 'sensitivity_summary.csv'}")

    default_lambda = cfg["velocity_schedulers"]["shcs"]["lambda_schedule"]
    default_rnom = cfg["shapers"]["dynamic"]["r_nominal"]

    fig = make_composite(lambda_data, rnom_data, default_lambda, default_rnom)
    save_fig(fig, OUT_DIR / "fig5_sensitivity.png")
    save_subfigures(lambda_data, rnom_data, default_lambda, default_rnom)
    print("\n  Experiment 4.5 completed.")


if __name__ == "__main__":
    main()
