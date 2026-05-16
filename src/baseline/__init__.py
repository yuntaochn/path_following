"""Baseline public interfaces."""

from baseline.actuator import TwinThrusterAllocator
from baseline.config import (
    BASELINE_CONFIG,
    get_config,
    get_disturbance,
    get_experiment,
    get_method,
    get_path,
    make_s_curve_path,
    make_turn_path,
)
from baseline.controller import USVLOSController
from baseline.heading_shaper import HeadingReferenceShaper, make_shaper
from baseline.los import AdaptiveLOSGuidance, ILOSGuidance, LOSGuidance
from baseline.metrics import (
    control_energy,
    iae,
    ise,
    itae,
    max_abs,
    rms,
    summarize_tracking_log,
    to_paper_table_rows,
)
from baseline.pid import PID
from baseline.simulator import Simulator
from baseline.usv_model import CS2Params, SimpleUSV3DOF, USV3DOF
from baseline.velocity_scheduler import VelocityScheduler, make_velocity_scheduler

__all__ = [
    "AdaptiveLOSGuidance",
    "BASELINE_CONFIG",
    "TwinThrusterAllocator",
    "CS2Params",
    "HeadingReferenceShaper",
    "ILOSGuidance",
    "LOSGuidance",
    "PID",
    "SimpleUSV3DOF",
    "Simulator",
    "USV3DOF",
    "USVLOSController",
    "VelocityScheduler",
    "control_energy",
    "get_config",
    "get_disturbance",
    "get_experiment",
    "get_method",
    "get_path",
    "iae",
    "ise",
    "itae",
    "make_s_curve_path",
    "make_shaper",
    "make_turn_path",
    "make_velocity_scheduler",
    "max_abs",
    "rms",
    "summarize_tracking_log",
    "to_paper_table_rows",
]
