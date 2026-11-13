"""Observation features shared by demo collection and policy deployment.

The policy is conditioned on the *same* inputs the scripted controller receives - the grasp
target and place target produced by perception + planning - plus proprioception.  It never
sees live simulator object state, so the learned skill can run in Vision mode.
"""

from __future__ import annotations

import numpy as np

from robotics.simulation.scene import OBJECT_NAMES

FEATURE_VERSION = 1
FEATURE_NAMES = (
    ["eef_x", "eef_y", "eef_z"]
    + ["grasp_dx", "grasp_dy", "grasp_dz", "grasp_top_dz"]
    + ["place_dx", "place_dy", "place_dz"]
    + ["gripper_width", "prev_gripper_cmd", "held_latch"]
    + [f"is_{n}" for n in OBJECT_NAMES]
)
OBS_DIM = len(FEATURE_NAMES)
ACTION_NAMES = ["dx", "dy", "dz", "gripper"]
ACTION_DIM = len(ACTION_NAMES)


class HeldLatch:
    """One bit of memory derived from proprioception: "the object has been grasped".

    It turns on after the gripper has been commanded closed and the finger gap has stayed in
    the holding range for a few consecutive steps.  This removes the ambiguity between
    "about to pick" and "just released" states, which look alike in a single snapshot.
    """

    def __init__(self, min_width: float, max_width: float, steps: int = 5):
        self.min_width, self.max_width, self.steps = min_width, max_width, steps
        self.count = 0
        self.value = 0.0

    def update(self, width: float, gripper_cmd: float) -> float:
        if gripper_cmd > 0 and self.min_width < width < self.max_width:
            self.count += 1
        else:
            self.count = 0
        if self.count >= self.steps:
            self.value = 1.0
        return self.value


def build_features(
    eef_pos: np.ndarray,
    gripper_width: float,
    grasp_pos: np.ndarray,
    grasp_top_z: float,
    place_pos: np.ndarray,
    prev_gripper_cmd: float,
    held_latch: float,
    object_name: str,
) -> np.ndarray:
    eef = np.asarray(eef_pos, dtype=np.float64)
    onehot = np.array([1.0 if object_name == n else 0.0 for n in OBJECT_NAMES])
    f = np.concatenate(
        [
            eef,
            np.asarray(grasp_pos) - eef,
            [grasp_top_z - eef[2]],
            np.asarray(place_pos) - eef,
            [gripper_width, prev_gripper_cmd, held_latch],
            onehot,
        ]
    )
    return f.astype(np.float32)
