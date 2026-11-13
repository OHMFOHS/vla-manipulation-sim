"""Mapping Cartesian set-points to robosuite OSC_POSE delta actions.

robosuite's default Panda controller is ``OSC_POSE`` with ``input_type="delta"`` and
``input_ref_frame="base"``: a normalised action ``a ∈ [-1, 1]^7`` is scaled to a position delta
of at most ``0.05 m`` and an axis-angle rotation delta of at most ``0.5 rad`` per control step,
the last entry commands the gripper (-1 open, +1 close).  The robot base in the bins arena is
not rotated, so base-frame and world-frame axes coincide (verified by a unit test).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from robotics.common.config import ControllerConfig

GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0


def top_down_orientation(yaw: float = 0.0) -> Rotation:
    """End-effector orientation with the gripper pointing straight down (fingers along world y).

    The Panda ``grip_site`` points along +z of the hand, so "down" is a 180° roll about world x.
    """
    return Rotation.from_euler("z", yaw) * Rotation.from_euler("x", np.pi)


def quat_xyzw_to_rotation(q: np.ndarray) -> Rotation:
    return Rotation.from_quat(np.asarray(q, dtype=np.float64))  # scipy uses (x, y, z, w)


class OSCCommander:
    """Proportional Cartesian controller producing *bounded* normalised OSC actions."""

    def __init__(self, config: ControllerConfig, action_dim: int = 7):
        self.cfg = config
        self.action_dim = action_dim

    def position_action(self, eef_pos: np.ndarray, setpoint: np.ndarray) -> np.ndarray:
        err = np.asarray(setpoint, dtype=np.float64) - np.asarray(eef_pos, dtype=np.float64)
        a = self.cfg.position_gain * err / self.cfg.max_delta_pos
        return np.clip(a, -self.cfg.action_clip, self.cfg.action_clip)

    def rotation_action(self, eef_quat_xyzw: np.ndarray, target: Rotation) -> np.ndarray:
        """Axis-angle error ``R_target · R_current⁻¹`` (world frame), normalised and clipped."""
        current = quat_xyzw_to_rotation(eef_quat_xyzw)
        err = (target * current.inv()).as_rotvec()
        a = self.cfg.rotation_gain * err / self.cfg.max_delta_rot
        return np.clip(a, -self.cfg.action_clip, self.cfg.action_clip)

    def action(
        self,
        eef_pos: np.ndarray,
        eef_quat_xyzw: np.ndarray,
        setpoint: np.ndarray,
        target_rot: Rotation,
        gripper: float,
    ) -> np.ndarray:
        a = np.zeros(self.action_dim)
        a[0:3] = self.position_action(eef_pos, setpoint)
        a[3:6] = self.rotation_action(eef_quat_xyzw, target_rot)
        a[-1] = float(np.clip(gripper, -1.0, 1.0))
        return a
