"""Configuration loader for baseline experiments.

所有可调参数统一写在仓库根目录的 `configs/baseline_config.json`。
这个 Python 模块只做三件事：
    1. 读取 JSON；
    2. 把 JSON 中的列表转换成仿真代码需要的 numpy 数组；
    3. 提供 `get_path`、`get_method` 等小工具，避免脚本直接解析配置细节。

这样可以让 `src/baseline` 保持为算法代码，配置文件保持为可读、可复现的
实验记录。
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "baseline_config.json"


# ---------------------------------------------------------------------------
# 路径生成函数：只接收 JSON 传入的参数，不在代码里保留项目默认路径参数
# ---------------------------------------------------------------------------

def make_s_curve_path(
    x_start: float,
    y_offset: float,
    phase: float,
    length: float,
    amplitude: float,
    num_points: int,
    periods: float,
) -> np.ndarray:
    """生成平滑 S 形路径，用于检查方法是否只适合折线路径。"""

    xs = np.linspace(float(x_start), float(x_start) + length, int(num_points))
    progress = (xs - float(x_start)) / length
    ys = y_offset + amplitude * np.sin(2.0 * np.pi * periods * progress + phase)
    return np.column_stack([xs, ys]).astype(float)


def make_turn_path(
    start: list[float] | tuple[float, float] | np.ndarray,
    initial_heading: float,
    turn_deg: float,
    leg_length: float,
) -> np.ndarray:
    """生成指定转角的两段折线路径。"""

    p0 = _as_array(start, length=2)
    heading_0 = float(initial_heading)
    heading_1 = heading_0 + np.deg2rad(turn_deg)
    p1 = p0 + leg_length * np.array([np.cos(heading_0), np.sin(heading_0)], dtype=float)
    p2 = p1 + leg_length * np.array([np.cos(heading_1), np.sin(heading_1)], dtype=float)
    return np.vstack([p0, p1, p2]).astype(float)


# ---------------------------------------------------------------------------
# JSON 读取与规范化
# ---------------------------------------------------------------------------

def _as_array(value: Any, length: int | None = None) -> np.ndarray:
    """把 JSON 列表转为 float 数组，并可选检查长度。"""

    arr = np.asarray(value, dtype=float)
    if length is not None:
        arr = arr.reshape(length,)
    return arr


def _load_raw_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """读取 JSON 原始配置。"""

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_config(raw: dict) -> dict:
    """把 JSON 配置转换成仿真代码更容易使用的形式。"""

    cfg = deepcopy(raw)

    initial = cfg["initial_state"]
    initial["eta0"] = _as_array(initial["eta0"], length=3)
    initial["nu0"] = _as_array(initial["nu0"], length=3)

    for disturbance in cfg["disturbances"].values():
        # 省略的噪声/偏置字段默认为零向量，允许 JSON 中只写非零项
        disturbance["current_velocity"] = _as_array(
            disturbance.get("current_velocity", [0.0, 0.0]), length=2)
        for key in (
            "force_bias",
            "force_noise_std",
            "control_noise_std",
            "eta_noise_std",
            "nu_noise_std",
        ):
            disturbance[key] = _as_array(disturbance.get(key, [0.0, 0.0, 0.0]), length=3)

    for experiment in cfg["experiments"].values():
        if "initial_state" not in experiment:
            continue
        experiment_initial = experiment["initial_state"]
        experiment_initial["eta0"] = _as_array(experiment_initial["eta0"], length=3)
        experiment_initial["nu0"] = _as_array(experiment_initial["nu0"], length=3)

    return cfg


def validate_config(config: dict) -> None:
    """检查 JSON 中跨字段引用是否一致。

    这里不判断“参数好不好”，只判断“引用是否存在”。例如实验里写了
    `method: shcs`，就必须能在 `methods` 中找到 `shcs`。
    """

    required_sections = (
        "simulation",
        "initial_state",
        "model",
        "los",
        "pid",
        "controller",
        "paths",
        "disturbances",
        "shapers",
        "velocity_schedulers",
        "methods",
        "experiments",
        "plotting",
        "summary_fields",
    )
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing required config section: {section}")

    method_names = set(config["methods"])
    path_names = set(config["paths"])
    disturbance_names = set(config["disturbances"])
    shaper_names = set(config["shapers"])
    scheduler_names = set(config["velocity_schedulers"])

    for method_name, method in config["methods"].items():
        shaper = method.get("shaper")
        scheduler = method.get("velocity_scheduler")
        if shaper is not None and shaper not in shaper_names:
            raise KeyError(f"Method {method_name!r} references unknown shaper {shaper!r}")
        if scheduler is not None and scheduler not in scheduler_names:
            raise KeyError(
                f"Method {method_name!r} references unknown velocity scheduler {scheduler!r}"
            )

    for experiment_name, experiment in config["experiments"].items():
        methods = [experiment["method"]] if "method" in experiment else list(experiment["methods"])
        paths = [experiment["path"]] if "path" in experiment else list(experiment["paths"])
        seeds = [experiment["seed"]] if "seed" in experiment else list(experiment["seeds"])

        for method in methods:
            if method not in method_names:
                raise KeyError(
                    f"Experiment {experiment_name!r} references unknown method {method!r}"
                )
        for path in paths:
            if path not in path_names:
                raise KeyError(f"Experiment {experiment_name!r} references unknown path {path!r}")

        disturbances = (
            [experiment["disturbance"]]
            if "disturbance" in experiment
            else list(experiment["disturbances"])
        )
        for disturbance in disturbances:
            if disturbance not in disturbance_names:
                raise KeyError(
                    f"Experiment {experiment_name!r} references unknown disturbance {disturbance!r}"
                )
        if not seeds:
            raise ValueError(f"Experiment {experiment_name!r} must define at least one seed")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """读取并规范化配置文件。"""

    config = _normalize_config(_load_raw_config(path))
    validate_config(config)
    return config


BASELINE_CONFIG = load_config()


def get_config() -> dict:
    """返回配置深拷贝，防止仿真过程意外修改全局配置。"""

    return deepcopy(BASELINE_CONFIG)


# ---------------------------------------------------------------------------
# 配置访问工具
# ---------------------------------------------------------------------------

def _path_from_spec(spec: dict) -> np.ndarray:
    """根据 JSON 中的路径描述生成路径点数组。"""

    kind = spec.get("kind", "points")
    if kind == "points":
        return _as_array(spec["points"]).reshape(-1, 2)
    if kind == "s_curve":
        return make_s_curve_path(
            x_start=spec["x_start"],
            y_offset=spec["y_offset"],
            phase=spec["phase"],
            length=spec["length"],
            amplitude=spec["amplitude"],
            num_points=spec["num_points"],
            periods=spec["periods"],
        )
    if kind == "turn":
        return make_turn_path(
            start=spec["start"],
            initial_heading=spec["initial_heading"],
            turn_deg=spec["turn_deg"],
            leg_length=spec["leg_length"],
        )
    raise ValueError(f"Unknown path kind: {kind!r}")


def get_path(name: str, config: dict | None = None) -> np.ndarray:
    """按路径名读取路径点数组。"""

    cfg = get_config() if config is None else config
    try:
        return _path_from_spec(cfg["paths"][name])
    except KeyError as exc:
        raise KeyError(f"Unknown path {name!r}. Available: {list(cfg['paths'])}") from exc


def get_disturbance(name: str, config: dict | None = None) -> dict:
    """按名称读取扰动配置。"""

    cfg = get_config() if config is None else config
    try:
        return deepcopy(cfg["disturbances"][name])
    except KeyError as exc:
        raise KeyError(
            f"Unknown disturbance {name!r}. Available: {list(cfg['disturbances'])}"
        ) from exc


def get_method(name: str, config: dict | None = None) -> dict:
    """按名称读取方法配置。"""

    cfg = get_config() if config is None else config
    try:
        return deepcopy(cfg["methods"][name])
    except KeyError as exc:
        raise KeyError(f"Unknown method {name!r}. Available: {list(cfg['methods'])}") from exc


def get_experiment(name: str, config: dict | None = None) -> dict:
    """按名称读取实验配置。"""

    cfg = get_config() if config is None else config
    try:
        return deepcopy(cfg["experiments"][name])
    except KeyError as exc:
        raise KeyError(
            f"Unknown experiment {name!r}. Available: {list(cfg['experiments'])}"
        ) from exc


__all__ = [
    "BASELINE_CONFIG",
    "DEFAULT_CONFIG_PATH",
    "REPO_ROOT",
    "get_config",
    "get_disturbance",
    "get_experiment",
    "get_method",
    "get_path",
    "load_config",
    "make_s_curve_path",
    "make_turn_path",
    "validate_config",
]
