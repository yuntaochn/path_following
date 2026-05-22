"""
实验工具层 — 所有实验脚本共享的基础设施

职责：
  1. 从 baseline_config.json 读取参数，组装完整控制器（LOS + PID + 整形器 + 调速器）
  2. 运行单次闭环仿真（RK4 积分 + 双螺旋桨推力分配 + 扰动）
  3. 保存原始时间序列（NPZ）、汇总指标（JSON）、对比表格（CSV）
  4. 统一的 matplotlib 论文风格（Times New Roman，colorblind-safe 配色）
  5. 共享绘图辅助函数（参考路径绘制、终点标记、饱和阴影等）

设计原则：
  每个实验脚本只需声明"用哪些方法、哪条路径、哪种扰动"，
  其余的控制器组装、模型初始化、仿真运行、结果保存均在本模块完成。
  若未来要新增一种方法，只需在 baseline_config.json 中添加配置，
  再调用 run_trial() 即可，无需修改各实验脚本的控制器构造逻辑。
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

# ─── matplotlib 环境配置 ─────────────────────────────────────────────────────
# 在服务器/无 GUI 环境中，需要在导入 matplotlib 前设置配置目录和后端，
# 否则可能因权限问题报错或弹出 GUI 窗口。
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent.parent / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")  # 非交互后端，直接写文件，不弹窗
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.text import Text

# ─── 源码路径配置 ─────────────────────────────────────────────────────────────
# 确保 src/ 目录在 Python 搜索路径中，使 baseline 包可正常导入。
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baseline import (
    get_config,
    get_path,
    ILOSGuidance,
    LOSGuidance,
    AdaptiveLOSGuidance,
    PID,
    make_shaper,
    make_velocity_scheduler,
    TwinThrusterAllocator,
    USVLOSController,
    USV3DOF,
    Simulator,
    summarize_tracking_log,
)
from baseline.metrics import ci95, evaluate_metric


# ─────────────────────────────────────────────────────────────────────────────
# 全局路径
# ─────────────────────────────────────────────────────────────────────────────

# REPO_ROOT：仓库根目录（scripts/ 的父目录）
REPO_ROOT    = Path(__file__).parent.parent
# RESULTS_ROOT：所有实验输出的根目录，各实验各自建子目录
RESULTS_ROOT = REPO_ROOT / "results" / "baseline"


# 《哈尔滨工程大学学报》模板要求图中说明尽量使用中文，图题为中文，且期刊
# 可能黑白印刷。这里把 CJK 字体选择集中到共享工具层，避免各实验脚本重复写
# Windows/Linux/macOS 字体分支。
#
# 字体规范：中文黑体（小五号 = 9 pt），英文 Times New Roman（小五号 = 9 pt）。
# 候选列表按优先级排列：首选 SimHei（黑体），依次回退到其他无衬线中文字体。
CJK_FONT_CANDIDATES = [
    "SimHei",              # 黑体，首选（Windows/macOS 内置）
    "Microsoft YaHei",     # 微软雅黑，黑体系，Windows 备选
    "Source Han Sans SC",  # 思源黑体，跨平台
    "Noto Sans CJK SC",    # Google Noto 黑体，Linux 备选
    "WenQuanYi Micro Hei", # 文泉驿，Linux 最后备选
    "SimSun",              # 宋体，最终回退（有 CJK 支持但非首选）
]
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_CJK_FONT_CACHE: dict[str, fm.FontProperties] = {}

# 《哈尔滨工程大学学报》写作模板中的图形尺寸要求：
# 小图宽度 6.5-7.0 cm，大图宽度 12-15 cm。脚本统一按 7 cm / 15 cm
# 生成单栏和通栏图，避免插入论文后再缩放导致字体和线宽失真。
CM_PER_INCH = 2.54
HEU_SMALL_FIG_WIDTH_CM = 7.0
HEU_LARGE_FIG_WIDTH_CM = 15.0
HEU_SMALL_FIG_WIDTH_IN = HEU_SMALL_FIG_WIDTH_CM / CM_PER_INCH
HEU_LARGE_FIG_WIDTH_IN = HEU_LARGE_FIG_WIDTH_CM / CM_PER_INCH


def heu_figsize(width: str = "large", aspect: float = 0.62) -> tuple[float, float]:
    """Return a journal-compliant figure size in inches."""
    fig_width = HEU_SMALL_FIG_WIDTH_IN if width == "small" else HEU_LARGE_FIG_WIDTH_IN
    return fig_width, fig_width * aspect


def get_cjk_font(weight: str | int = "normal") -> fm.FontProperties:
    """Return an installed Chinese-capable font for Matplotlib text."""
    key = str(weight)
    if key in _CJK_FONT_CACHE:
        return _CJK_FONT_CACHE[key]

    for family in CJK_FONT_CANDIDATES:
        prop = fm.FontProperties(family=family, weight=weight)
        try:
            fm.findfont(prop, fallback_to_default=False)
        except ValueError:
            continue
        _CJK_FONT_CACHE[key] = prop
        return prop

    prop = fm.FontProperties(weight=weight)
    _CJK_FONT_CACHE[key] = prop
    return prop


def _has_cjk(text: str | None) -> bool:
    return bool(text and _CJK_RE.search(text))


def apply_cjk_text_fonts(fig: plt.Figure) -> None:
    """将图中含中文的文字对象字体切换为黑体，同时保留原有字号和字重。

    set_fontproperties() 会用新的 FontProperties 对象完整替换旧属性，
    若新对象未显式指定 size，matplotlib 会回退到 rcParams 默认值，
    导致脚本中用 fontsize= 精细设定的小号注记文字被意外放大。
    因此必须在切换字体前先读取、切换后再写回字号。
    """
    for text in fig.findobj(match=Text):
        if _has_cjk(text.get_text()):
            size = text.get_fontsize()                      # 保存当前字号
            text.set_fontproperties(get_cjk_font(text.get_fontweight()))
            text.set_fontsize(size)                         # 恢复字号


# ─────────────────────────────────────────────────────────────────────────────
# 实验协议辅助
# ─────────────────────────────────────────────────────────────────────────────

def get_initial_state(cfg: dict, experiment_name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """读取论文实验初始状态（位置/姿态 η0，速度 ν0）。

    默认使用配置根节点的 initial_state。若某实验在 experiments.<name>.initial_state
    中显式覆盖了初始状态，则优先使用该局部设置（例如泛化实验可能需要不同起点）。

    返回：
        eta0: shape (3,) 数组 [x, y, ψ]，位置与艏向角（m, m, rad）
        nu0:  shape (3,) 数组 [u, v, r]，体坐标系速度（m/s, m/s, rad/s）
    """
    state = cfg["initial_state"]
    if experiment_name is not None:
        exp = cfg.get("experiments", {}).get(experiment_name, {})
        state = exp.get("initial_state", state)
    return np.asarray(state["eta0"], dtype=float), np.asarray(state["nu0"], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# 控制器与仿真器组装
# ─────────────────────────────────────────────────────────────────────────────

def _build_los(cfg: dict):
    """根据配置字典的 los.type 字段实例化 LOS 制导律对象。

    支持三种制导律：
      - "ilos"：积分型 LOS（Integral LOS），通过积分项补偿稳态横流
      - "alos"：自适应 LOS，自动调整前视距离
      - 其他/默认：标准 LOS，纯几何前视制导
    """
    lc = cfg["los"]
    t  = lc["type"]
    if t == "ilos":
        return ILOSGuidance(
            lookahead=lc["lookahead"],
            k_i=lc["k_i"],
            sigma_limit=lc["sigma_limit"],
            switch_radius=lc["switch_radius"],
        )
    if t == "alos":
        return AdaptiveLOSGuidance(
            lookahead_base=lc.get("lookahead", 10.0),
            switch_radius=lc.get("switch_radius", 3.0),
        )
    # 默认标准 LOS
    return LOSGuidance(
        lookahead=lc["lookahead"],
        switch_radius=lc["switch_radius"],
    )


def _build_pid(p: dict) -> PID:
    """从配置字典实例化 PID 控制器（支持抗积分卷绕和微分滤波）。

    output_limit 若存在则转为元组（下限, 上限），对应偏航力矩的物理饱和值。
    angle_wrap=True 表示误差信号需要角度归一化（适用于航向 PID）。
    aw_gain 为抗积分卷绕增益（0.0 表示不启用，>0 启用 back-calculation 策略）。
    """
    return PID(
        kp=p["kp"], ki=p["ki"], kd=p["kd"],
        integral_limit=p.get("integral_limit"),
        output_limit=tuple(p["output_limit"]) if p.get("output_limit") else None,
        derivative_filter=p.get("derivative_filter", 0.0),
        angle_wrap=p.get("angle_wrap", False),
        aw_gain=p.get("aw_gain", 0.0),
    )


def build_controller(
    method_name: str,
    cfg: dict,
    shaper_override: dict | None = None,
    scheduler_override: dict | None = None,
) -> USVLOSController:
    """从 baseline_config.json 组装完整的三层控制器。

    三层结构：
      第1层：ILOS 制导律 → 期望艏向角 ψ_d（纯几何）
      第2层：航向整形器 → 可执行参考 ψ_ref（速率限制/一阶滤波/动态整形）
      第3层：速度调度器 → 动态期望速度 u_d(t)（基于整形残差 + 推进器约束）

    shaper_override / scheduler_override：用于参数敏感性分析，注入替换后的
    配置字典，而无需修改全局 cfg（避免多次实验间的状态污染）。
    """
    method_cfg = cfg["methods"][method_name]
    ctrl_cfg   = cfg["controller"]

    # SHCS 类方法可以用更短的前视距离（整形后的参考已经"慢下来"，不需要保守前视），
    # 而基线 ILOS-PID 保持较长前视以避免路径点切换时的过渡振荡。
    los_cfg = cfg
    if method_cfg.get("los_override"):
        los_cfg = deepcopy(cfg)
        los_cfg["los"].update(method_cfg["los_override"])
    los     = _build_los(los_cfg)
    pid_u   = _build_pid(cfg["pid"]["surge"])

    # 部分方法使用带抗积分卷绕的特殊航向 PID（如 anti_windup 消融变体）
    heading_pid_key = method_cfg.get("pid_heading", "heading")
    pid_psi = _build_pid(cfg["pid"][heading_pid_key])

    # 整形器（可选）：将阶跃式艏向指令平滑为连续可跟踪参考
    shaper_name = method_cfg.get("shaper")
    shaper = None
    if shaper_name is not None:
        sc     = shaper_override if shaper_override is not None else cfg["shapers"][shaper_name]
        shaper = make_shaper(**sc)

    # 速度调度器（可选）：在转弯时动态降速，给偏航控制让出推力余量
    sched_name = method_cfg.get("velocity_scheduler")
    scheduler  = None
    if sched_name is not None:
        vc        = scheduler_override if scheduler_override is not None else cfg["velocity_schedulers"][sched_name]
        scheduler = make_velocity_scheduler(**vc)

    return USVLOSController(
        pid_u=pid_u,
        pid_psi=pid_psi,
        los=los,
        u_d=ctrl_cfg["u_d"],
        shaper=shaper,
        velocity_scheduler=scheduler,
        tau_r_filter_tau=ctrl_cfg["tau_r_filter_tau"],
        tau_r_rate_limit=ctrl_cfg["tau_r_rate_limit"],
    )


def run_trial(
    method_name: str,
    cfg: dict,
    waypoints: np.ndarray,
    eta0: np.ndarray,
    nu0: np.ndarray,
    dist_cfg: dict,
    seed: int,
    shaper_override: dict | None = None,
    scheduler_override: dict | None = None,
) -> dict:
    """组装完整闭环系统并执行一次确定性仿真。

    随机种子 seed 控制扰动序列（海流噪声、测量噪声）。
    配对对比实验中，对所有方法使用相同 seed，确保扰动条件完全一致，
    从而将方法差异从随机性中分离出来（paired design）。

    返回字典：
      result["log"]：完整时间序列（55+ 变量，含 x/y/psi/u/e_ct/e_psi/tau_r_cmd 等）
      result["summary"]：汇总指标（CTE RMS/IAE、航向 RMS、偏航能量、到达时间等）
    """
    controller = build_controller(method_name, cfg, shaper_override, scheduler_override)
    model      = USV3DOF()
    sc         = cfg["simulation"]
    act_cfg    = cfg.get("actuator", {"T_max": 30.0, "b": 0.30})
    allocator  = TwinThrusterAllocator(T_max=act_cfg["T_max"], b=act_cfg["b"])

    sim = Simulator(
        model=model,
        controller=controller,
        dt=sc["dt"],
        t_final=sc["t_final"],
        integration_method=sc["integration_method"],
        disturbance_config=dist_cfg,
        random_seed=seed,
        metrics_config=cfg["metrics"],
        actuator_allocator=allocator,
    )
    return sim.run(
        eta0=eta0, nu0=nu0, waypoints=waypoints,
        goal_tolerance=sc["goal_tolerance"],
        stop_when_reached=sc["stop_when_reached"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# 结果保存
# ─────────────────────────────────────────────────────────────────────────────

def _json_safe(obj):
    """递归将 numpy 标量/数组转为 Python 原生类型，使其可 JSON 序列化。"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_result(result: dict, out_dir: Path, prefix: str) -> None:
    """保存单次仿真结果到磁盘。

    文件：
      {prefix}_log.npz    — 压缩的 numpy 归档，含完整时间序列（事后分析用）
      {prefix}_summary.json — 汇总指标的 JSON，供论文写作引用具体数值
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{prefix}_log.npz", **result["log"])
    with open(out_dir / f"{prefix}_summary.json", "w") as f:
        json.dump(_json_safe(result["summary"]), f, indent=2)


def save_summaries_csv(rows: list[dict], out_path: Path) -> None:
    """将指标字典列表保存为 CSV（可直接用 Excel 打开或在 LaTeX 中引用）。

    自动收集所有行的全部键作为表头，确保不同方法的列数不同时也能正确输出。
    """
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 收集所有行的键，保持首次出现顺序（不用 set，避免乱序）
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# matplotlib 论文风格
# ─────────────────────────────────────────────────────────────────────────────

# 低饱和学术配色：整体比纯灰更清爽，同时配合线型保证黑白打印后仍可区分。
# SHCS 使用克制的砖红色，在各图中突出显示但不过分鲜艳。
METHOD_COLORS = {
    "los_pid":        "#4B5563",   # slate gray
    "fixed_rate":     "#3B6EA8",   # muted blue
    "first_order":    "#4C8B72",   # muted teal
    "dynamic_shaper": "#7E6AA6",   # muted violet
    "anti_windup":    "#B6783F",   # warm ochre
    "shcs_simple":    "#7A9E65",   # olive green
    "shcs":           "#B24A3B",   # brick red
}
METHOD_LS = {
    "los_pid":        "-",
    "fixed_rate":     "--",
    "first_order":    "-.",
    "dynamic_shaper": ":",
    "anti_windup":    "--",
    "shcs_simple":    "-.",
    "shcs":           "-",   # SHCS 用实线，颜色+线宽区分
}
# 未知方法的备用配色（按索引循环）
METHOD_FALLBACK_COLORS = ["#4B5563", "#3B6EA8", "#4C8B72", "#B6783F", "#B24A3B"]

PAPER_BLUE = "#3B6EA8"
PAPER_ORANGE = "#B6783F"
PAPER_GRAY = "#6B7280"
PAPER_GRID = "#D6DDE3"
PAPER_TEXT = "#3F4752"
PAPER_SUCCESS = "#2D7B4F"

METHOD_FILL_COLORS = {
    "los_pid":        "#7E8898",
    "fixed_rate":     "#8CADCF",
    "first_order":    "#8FB9A8",
    "dynamic_shaper": "#B0A4CE",
    "anti_windup":    "#D2A574",
    "shcs_simple":    "#B5C99F",
    "shcs":           "#D08573",
}

ABLATION_BAR_FACES = {
    # 冷灰梯度：基线方法由浅到深；SHCS 用暖陶色区分，避免面积色过饱和。
    "los_pid":        "#C8CDD5",   # cool light gray
    "anti_windup":    "#B0BAC5",   # steel gray
    "dynamic_shaper": "#98A5B2",   # medium steel
    "shcs_simple":    "#7E8E9E",   # dark steel
    "shcs":           "#C27B73",   # warm muted terracotta（比线条色更浅）
}
ABLATION_BAR_EDGES = {
    # 统一细描边（0.65 pt），与填充色同色系但稍深
    "los_pid":        "#8B929E",
    "anti_windup":    "#6B7888",
    "dynamic_shaper": "#546070",
    "shcs_simple":    "#3E4E5C",
    "shcs":           "#8F4A44",
}

# 复合图（多子图拼版）专用 RC：字体按小五号（9 pt）标准，刻度略小保持版面整洁
COMPOSITE_RC: dict = {
    "font.size":         9.0,    # 小五号
    "axes.titlesize":    9.0,    # 小五号
    "axes.labelsize":    9.0,    # 小五号
    "xtick.labelsize":   7.5,    # 介于小六（6.5 pt）与小五之间
    "ytick.labelsize":   7.5,
    "legend.fontsize":   7.5,
    "lines.linewidth":   1.0,
    "figure.dpi":        100,    # 屏幕预览分辨率（保存时用 600）
    "grid.alpha":        0.28,
    "grid.linewidth":    0.5,
}

PANEL_RC: dict = {
    "font.size":         9.0,    # 小五号
    "axes.titlesize":    9.0,    # 小五号
    "axes.labelsize":    9.0,    # 小五号
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   7.5,
    "lines.linewidth":   1.0,
    "grid.alpha":        0.28,
    "grid.linewidth":    0.5,
}

ENVELOPE_RC: dict = {
    "font.size":         9.0,    # 小五号
    "axes.titlesize":    9.0,    # 小五号
    "axes.labelsize":    9.0,    # 小五号
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   7.5,
    "lines.linewidth":   1.6,
    "grid.alpha":        0.28,
    "grid.linewidth":    0.5,
}

SENSITIVITY_RC: dict = {
    "font.size":         9.0,    # 小五号
    "axes.titlesize":    9.0,    # 小五号
    "axes.labelsize":    9.0,    # 小五号
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   7.5,
    "lines.linewidth":   1.2,
    "axes.linewidth":    0.65,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size":  2.3,
    "ytick.major.size":  2.3,
    "grid.alpha":        0.28,
    "grid.linewidth":    0.5,
}

BAR_RC: dict = {
    "font.size":         9.0,    # 小五号
    "axes.titlesize":    9.0,    # 小五号
    "axes.labelsize":    9.0,    # 小五号
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   7.5,
    "grid.alpha":        0.28,
    "grid.linewidth":    0.5,
}


def paper_style() -> None:
    """全局设置 matplotlib 为论文发表风格。

    字体规范（哈尔滨工程大学学报）：
      英文 Times New Roman，中文黑体（SimHei），均为小五号（9 pt）。
      apply_cjk_text_fonts() 在 save_fig() 中自动将含汉字的文字对象
      切换为黑体，同时保留各文字对象自身的字号设置。
    配色：低饱和色板 + 线型，兼顾屏幕阅读和黑白打印可区分性。
    布局：无顶/右脊线（现代期刊风格），轻灰网格辅助读图。
    分辨率：显示用 150 dpi，保存用 600 dpi（满足大多数期刊要求）。
    """
    plt.rcParams.update({
        # ── 字体 ────────────────────────────────────────────────
        # 英文/数字默认使用 Times New Roman（衬线体，学术期刊标准）；
        # 中文由 apply_cjk_text_fonts() 在保存前统一切换为 SimHei（黑体）。
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "SimSun", "DejaVu Serif"],
        "font.sans-serif":    ["Arial", "SimHei", "Microsoft YaHei", "DejaVu Sans"],
        "mathtext.fontset":   "stix",   # 数学公式用 STIX，与 Times 配套
        # ── 字号（小五号 = 9 pt）────────────────────────────────
        "font.size":          9,        # 全局基准，小五号
        "axes.titlesize":     9,        # 子图标题，小五号
        "axes.labelsize":     9,        # 坐标轴标签，小五号
        "xtick.labelsize":    7.5,      # 刻度标签，介于小六（6.5 pt）与小五之间
        "ytick.labelsize":    7.5,
        "legend.fontsize":    7.5,      # 图例，同刻度标签
        # ── 分辨率 ──────────────────────────────────────────────
        "figure.dpi":         150,
        "savefig.dpi":        900,
        "savefig.facecolor":  "white",
        "figure.facecolor":   "white",
        # ── 网格与脊线 ──────────────────────────────────────────
        "axes.grid":          True,
        "grid.alpha":         0.18,
        "grid.linewidth":     0.45,
        "grid.color":         "#b8c0c8",
        "axes.spines.top":    False,    # 去掉顶部脊线（现代风格）
        "axes.spines.right":  False,    # 去掉右侧脊线
        # ── 线条与坐标轴 ────────────────────────────────────────
        "lines.linewidth":    1.55,
        "axes.linewidth":     0.7,
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.major.size":   2.5,
        "ytick.major.size":   2.5,
        # ── 图例 ────────────────────────────────────────────────
        "legend.framealpha":  0.92,
        "legend.edgecolor":   "#d0d0d0",
        "legend.borderpad":   0.35,
        "legend.labelspacing": 0.25,
        "axes.unicode_minus": False,
        # ── 矢量图嵌字：提交 PDF 时保留可编辑 TrueType 字体 ───────────────
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "svg.fonttype":       "none",
    })


def apply_plot_style(kind: str = "standard") -> None:
    """Apply the shared paper style plus a figure-type-specific RC preset."""
    paper_style()
    presets = {
        "standard": {},
        "composite": COMPOSITE_RC,
        "panel": PANEL_RC,
        "envelope": ENVELOPE_RC,
        "sensitivity": SENSITIVITY_RC,
        "bar": BAR_RC,
    }
    plt.rcParams.update(presets.get(kind, presets["standard"]))


def get_method_style(method_name: str, idx: int = 0) -> tuple[str, str]:
    """返回 (color, linestyle) 元组，优先按方法名查表，否则按索引取备用色。"""
    c  = METHOD_COLORS.get(method_name, METHOD_FALLBACK_COLORS[idx % len(METHOD_FALLBACK_COLORS)])
    ls = METHOD_LS.get(method_name, ["-", "--", "-.", ":"][idx % 4])
    return c, ls


def get_envelope_style(method_name: str, label: str | None = None, idx: int = 0) -> dict[str, Any]:
    """Return consistent line/fill styling for Monte-Carlo envelope plots."""
    color, ls = get_method_style(method_name, idx)
    return {
        "label": label or method_name,
        "color": color,
        "fill": METHOD_FILL_COLORS.get(method_name, color),
        "linewidth": 1.9 if method_name == "shcs" else 1.7,
        "ls": ls,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 共享绘图辅助
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, *paths, dpi: int = 900, ensure_pdf: bool = False) -> None:
    """将图形对象保存到一个或多个路径，完成后关闭以释放内存。

    自动创建所需的父目录。dpi 默认 900，满足 Word 版论文插图的高清 PNG 需求。
    如需同时保存 PDF，可传入 ensure_pdf=True。
    """
    apply_cjk_text_fonts(fig)

    save_paths = [Path(p) for p in paths]
    if ensure_pdf and save_paths and not any(p.suffix.lower() == ".pdf" for p in save_paths):
        save_paths.extend(p.with_suffix(".pdf") for p in save_paths if p.suffix.lower() in {".png", ".jpg", ".jpeg"})

    seen: set[Path] = set()
    for p in save_paths:
        if p in seen:
            continue
        seen.add(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"  [保存] {p}")
    plt.close(fig)


def draw_final_position(ax: plt.Axes, log: dict, color: str) -> None:
    """在轨迹图上标记仿真结束位置（白底彩边圆圈）。

    不人为延伸轨迹到路径终止航点，诚实反映仿真的实际终点。
    白色填充使其在彩色线条上清晰可辨（zorder=7 确保在最顶层）。
    """
    ax.plot(
        log["x"][-1], log["y"][-1],
        marker="o", ms=3.4, mfc="white", mec=color, mew=0.9,
        linestyle="none", zorder=7,
    )


def plot_waypoints(
    ax: plt.Axes,
    wps: np.ndarray,
    goal_tol: float | None = None,
    ref_label: str = "参考路径",
) -> None:
    """绘制参考路径（灰色虚线）、起点三角形、终点星形及目标容差圆。

    goal_tol：目标到达容差半径（m），若为 None 则不绘制容差圆。
    ref_label：图例中参考路径的标签文字。
    """
    wps = np.asarray(wps)
    ax.plot(wps[:, 0], wps[:, 1], color="gray", ls="--", lw=1.0, alpha=0.5,
            label=ref_label)
    ax.plot(wps[0, 0],  wps[0, 1], marker="^", color="#3B6EA8", ms=5.0,
            linestyle="none", zorder=6, label="_nolegend_")
    ax.plot(wps[-1, 0], wps[-1, 1], marker="*", color="#B24A3B", ms=6.5,
            linestyle="none", zorder=6, label="_nolegend_")
    if goal_tol is not None:
        ax.add_patch(plt.Circle(
            (wps[-1, 0], wps[-1, 1]),
            goal_tol, fill=False, color="#B24A3B", ls=":", lw=0.8, alpha=0.38,
        ))


def shade_saturation_windows(
    ax: plt.Axes,
    t: np.ndarray,
    tau_raw: np.ndarray,
    tau_r_lim: float,
    color: str = "tomato",
    alpha: float = 0.18,
) -> None:
    """对偏航力矩饱和时间段添加半透明背景色，直观展示控制器饱和程度。

    判断准则：|τ_raw| ≥ 0.97 × τ_r_lim（97% 阈值容许传感器数值小幅抖动）。
    饱和表示 PID 输出超过执行器物理极限，此时实际施加的偏航力矩被截断，
    积分项继续累积导致积分卷绕，是 ILOS-PID 基线在急转弯工况的主要缺陷。
    """
    sat  = np.abs(tau_raw) >= tau_r_lim * 0.97
    if not sat.any():
        return
    diff   = np.diff(sat.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        ax.axvspan(
            t[min(s, len(t) - 1)],
            t[min(e, len(t) - 1)],
            alpha=alpha, color=color, zorder=0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 通用打印工具
# ─────────────────────────────────────────────────────────────────────────────

# 论文表格常用指标的展示配置：(数据键名, 表头文字, 格式串)
PAPER_METRICS = [
    ("cross_track_rms",           "CTE RMS [m]",           ".3f"),
    ("cross_track_max_abs",       "CTE Max [m]",            ".3f"),
    ("turn_cte_rms",              "Turn CTE RMS [m]",       ".3f"),
    ("turn_cte_peak_abs",         "Turn CTE Peak [m]",      ".3f"),
    ("cross_track_iae",           "CTE IAE [m·s]",          ".2f"),
    # 公平航向误差（psi_d - psi，对所有方法一致）
    ("heading_los_error_rms",     "Hdg LOS RMS [rad]",      ".4f"),
    ("turn_heading_los_peak_abs", "Turn Hdg LOS Peak [rad]",".4f"),
    # PID参考跟踪误差（psi_ref - psi，SHCS因整形参考平滑而人为偏小）
    ("heading_error_rms",         "Hdg Ref RMS [rad]",      ".4f"),
    ("control_energy_tau_r_cmd",  "Yaw Energy",             ".1f"),
    ("speed_reduction_max_pct",   "Max Speed Red [%]",      ".1f"),
    ("turn_sat_ratio_tau_r_cmd",  "Turn Sat Ratio",         ".3f"),
    ("reach_time",                "Reach Time [s]",         ".1f"),
]


def print_metrics_table(
    results: list[dict],
    labels: list[str],
    metrics: list[tuple] | None = None,
) -> None:
    """在控制台打印多方法指标对比表（便于快速核查仿真结果）。

    metrics：自定义指标列表，默认使用 PAPER_METRICS。
    """
    ms     = metrics if metrics is not None else PAPER_METRICS
    col_w  = 20
    name_w = 26
    sep    = "─" * (name_w + col_w * len(results))
    print(f"\n{sep}")
    print(f"{'Metric':<{name_w}}", end="")
    for lbl in labels:
        print(f"{lbl[:col_w]:>{col_w}}", end="")
    print(f"\n{sep}")
    for field, display, fmt in ms:
        print(f"{display:<{name_w}}", end="")
        for res in results:
            val = res["summary"].get(field, float("nan"))
            try:
                txt = format(float(val), fmt)
            except (ValueError, TypeError):
                txt = "N/A"
            print(f"{txt:>{col_w}}", end="")
        print()
    print(sep)


def relative_improvement(base: float, new: float) -> float:
    """计算相对改善量 (base - new) / |base| × 100%。

    正值表示 new < base（指标值降低，性能改善）；
    负值表示 new > base（指标值升高，性能变差）。
    当 base ≈ 0 时返回 NaN 避免除零。
    """
    if abs(base) < 1e-12:
        return float("nan")
    return (base - new) / abs(base) * 100.0
