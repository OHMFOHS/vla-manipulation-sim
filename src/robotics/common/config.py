"""Typed configuration objects loaded from the YAML files in ``configs/``.

Business logic receives these validated objects and never parses YAML itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvConfig(_Strict):
    scene: Literal["color_sort"] = "color_sort"
    control_freq: int = 20
    perception_camera: str = "overhead"
    render_camera: str = "frontview"
    camera_height: int = 320
    camera_width: int = 320
    overhead_camera_pos: tuple[float, float, float] = (0.1, 0.015, 2.0)
    overhead_camera_fovy: float = 52.0
    initialization_noise: float = 0.02
    object_friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    object_density: float = 500.0
    hard_reset: bool = False


class ControllerConfig(_Strict):
    max_speed: float = 0.45  # m/s along a trajectory segment
    max_delta_pos: float = 0.05  # m, OSC output limit per control step (robosuite default)
    max_delta_rot: float = 0.5  # rad, OSC output limit per control step
    position_gain: float = 1.6  # proportional gain on (setpoint - eef) before clipping
    rotation_gain: float = 1.0
    action_clip: float = 1.0  # normalised action limit
    position_tolerance: float = 0.012  # m, waypoint reached
    pregrasp_height: float = 0.10  # m above the object top surface
    grasp_depth: float = 0.025  # m below the object top where finger tips close
    lift_height: float = 1.04  # absolute z after lifting
    place_release_height: float = 0.035  # m above container floor to release object bottom
    # Empty-handed clearance height after releasing / before navigating (fingertips clear the
    # 0.90 m bin walls).  Rising higher above the far compartments fully stretches the arm.
    retreat_height: float = 0.96
    gripper_close_steps: int = 14
    gripper_open_steps: int = 10
    segment_timeout_steps: int = 120
    settle_steps: int = 6
    grasp_min_width: float = 0.012  # finger gap (m) below which the gripper is "empty"
    grasp_max_width: float = 0.068  # finger gap above which the gripper is open (nothing held)


class RecoveryConfig(_Strict):
    max_recovery_attempts: int = Field(default=2, ge=0, le=5)
    episode_step_budget: int = 3000


class FaultConfig(_Strict):
    """Fault injection used to stress-test the closed loop (disabled for the main benchmark)."""

    enabled: bool = False
    slip_probability: float = 0.0  # force-open the gripper during transport
    grasp_offset_probability: float = 0.0  # corrupt the first grasp target by a few cm
    grasp_offset_m: float = 0.04


class SimulationConfig(_Strict):
    env: EnvConfig = EnvConfig()
    controller: ControllerConfig = ControllerConfig()
    recovery: RecoveryConfig = RecoveryConfig()
    faults: FaultConfig = FaultConfig()


class ColorClass(_Strict):
    name: str
    hsv_low: tuple[int, int, int]
    hsv_high: tuple[int, int, int]
    # optional second hue range (red wraps around hue=180 in OpenCV)
    hsv_low2: tuple[int, int, int] | None = None
    hsv_high2: tuple[int, int, int] | None = None


class PerceptionConfig(_Strict):
    min_pixels: int = 25
    max_valid_depth: float = 3.0
    min_valid_depth: float = 0.1
    min_height_above_floor: float = 0.015  # m: rejects flat floor markers / bin texture
    morph_kernel: int = 3
    top_surface_percentile: float = 90.0
    occlusion_visible_ratio: float = 0.35
    classes: list[ColorClass] = []


class TrainingConfig(_Strict):
    seed: int = 0
    hidden_sizes: list[int] = [256, 256, 256]
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    epochs: int = 60
    val_fraction: float = 0.15
    gripper_loss_weight: float = 1.0
    device: str = "cpu"
    demo_path: str = "outputs/data/demos.npz"
    checkpoint_dir: str = "outputs/checkpoints"
    collect_noise_std: float = 0.15  # DART-style action noise during collection
    collect_noise_fraction: float = 0.5  # fraction of demo episodes with noise injection
    # Feature columns dropped before training.  End-effector velocity is recorded but excluded
    # by default: it nearly encodes the previous action, and the closed-loop benchmark showed
    # the resulting "copycat" policy performs much worse (see docs/EXPERIMENTS.md).
    exclude_features: list[str] = ["eef_vx", "eef_vy", "eef_vz"]


T = TypeVar("T", bound=BaseModel)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return data


def load_config(model: type[T], path: str | Path | None) -> T:
    """Validate a YAML file into ``model``; ``None`` returns the defaults."""
    if path is None:
        return model()
    return model.model_validate(load_yaml(path))


def default_simulation_config() -> SimulationConfig:
    p = CONFIG_DIR / "simulation.yaml"
    return load_config(SimulationConfig, p if p.exists() else None)


def default_perception_config() -> PerceptionConfig:
    p = CONFIG_DIR / "perception.yaml"
    return load_config(PerceptionConfig, p if p.exists() else None)


def default_training_config() -> TrainingConfig:
    p = CONFIG_DIR / "training.yaml"
    return load_config(TrainingConfig, p if p.exists() else None)
