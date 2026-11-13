"""Physical fault injection used to stress-test failure detection and recovery.

Faults act through the same interfaces a real failure would: the gripper actuator command
(slip / actuator glitch) and the perceived grasp target (calibration error).  Objects are never
teleported; the physics engine decides what happens next.
"""

from __future__ import annotations

import logging

import numpy as np

from robotics.common.config import FaultConfig
from robotics.control.skills import ControlHook

logger = logging.getLogger(__name__)


class FaultInjector(ControlHook):
    def __init__(self, config: FaultConfig, seed: int):
        self.cfg = config
        self.rng = np.random.default_rng(seed + 7919)
        self.slip_at: int | None = None
        self._transport_steps = 0
        self._slip_remaining = 0
        self.injected: list[str] = []

    def arm_attempt(self, attempt: int) -> None:
        """Called by the executor before each manipulation attempt.

        Faults are only injected on the first attempt of a step, so that the system has a fair
        chance to recover (a real intermittent fault, not a permanently broken gripper).
        """
        self.slip_at = None
        self._transport_steps = 0
        self._slip_remaining = 0
        if not self.cfg.enabled or attempt > 0:
            return
        if self.rng.random() < self.cfg.slip_probability:
            self.slip_at = int(self.rng.integers(3, 15))

    def grasp_offset(self, attempt: int) -> np.ndarray:
        if (
            not self.cfg.enabled
            or attempt > 0
            or self.rng.random() >= self.cfg.grasp_offset_probability
        ):
            return np.zeros(3)
        angle = self.rng.uniform(0, 2 * np.pi)
        self.injected.append("grasp_offset")
        return np.array([np.cos(angle), np.sin(angle), 0.0]) * self.cfg.grasp_offset_m

    def before_step(
        self, env, phase: str, clean_action: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        if phase in ("transport", "bc_transport") and self.slip_at is not None:
            self._transport_steps += 1
            if self._transport_steps == self.slip_at:
                self._slip_remaining = 8
                self.injected.append("gripper_slip")
                logger.info("fault injection: gripper slip during transport")
        if self._slip_remaining > 0:
            self._slip_remaining -= 1
            action = action.copy()
            action[-1] = -1.0  # actuator glitch: gripper opens
            if self._slip_remaining == 0:
                self.slip_at = None
        return action
