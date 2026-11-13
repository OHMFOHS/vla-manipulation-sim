"""Runtime monitors attached to the controller as :class:`ControlHook` objects."""

from __future__ import annotations

import numpy as np

from robotics.common.config import ControllerConfig
from robotics.control.skills import ControlHook

# Phases during which the gripper must be holding the object.
HOLD_PHASES = frozenset({"transport", "lower", "bc_transport"})


class HoldMonitor(ControlHook):
    """Detects an unexpected drop from proprioception only (finger gap).

    A Panda gripper closed on one of our objects keeps a 3.5-5 cm gap.  A gap near zero means
    the fingers closed on nothing; a gap near 8 cm means the gripper opened.  Either way the
    object is no longer held.  A short debounce avoids false alarms from contact transients.
    """

    def __init__(self, config: ControllerConfig, debounce: int = 2):
        self.cfg = config
        self.debounce = debounce
        self._bad = 0

    def after_step(self, env, phase: str) -> str | None:
        if phase not in HOLD_PHASES:
            self._bad = 0
            return None
        w = env.gripper_width
        holding = self.cfg.grasp_min_width < w < self.cfg.grasp_max_width
        self._bad = 0 if holding else self._bad + 1
        if self._bad >= self.debounce:
            self._bad = 0
            return "object_dropped"
        return None


class StepBudgetMonitor(ControlHook):
    """Aborts any running skill once the episode-level step budget is exhausted."""

    def __init__(self, budget: int):
        self.budget = budget

    def after_step(self, env, phase: str) -> str | None:
        if env.step_count >= self.budget:
            return "episode_timeout"
        return None


class CollisionCounter(ControlHook):
    """Evaluation-only: counts contact events between the robot and the static scene.

    Uses simulator contacts, so it is a *metric*, never an input to the controller.
    """

    def __init__(self):
        self.events = 0
        self.contact_steps = 0
        self._in_contact = False
        self.pairs: set[tuple[str, str]] = set()

    def after_step(self, env, phase: str) -> str | None:
        hits = env.robot_environment_contacts()
        if hits:
            self.contact_steps += 1
            if not self._in_contact:
                self.events += 1
            self.pairs.update(hits)
        self._in_contact = bool(hits)
        return None


class FrameRecorder(ControlHook):
    """Captures rendered frames every ``every`` steps (for GIFs / videos)."""

    def __init__(
        self, camera: str = "frontview", every: int = 4, height: int = 360, width: int = 480
    ):
        self.camera = camera
        self.every = every
        self.size = (height, width)
        self.frames: list[np.ndarray] = []
        self.captions: list[str] = []
        self.caption = ""

    def after_step(self, env, phase: str) -> str | None:
        if env.step_count % self.every == 0:
            self.frames.append(env.render_rgb(self.camera, *self.size))
            self.captions.append(phase)
        return None
