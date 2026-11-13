"""Module E building blocks: monitors and fault injection (fake environment)."""

from __future__ import annotations

import numpy as np

from robotics.common.config import ControllerConfig, FaultConfig
from robotics.recovery.faults import FaultInjector
from robotics.recovery.monitors import HoldMonitor, StepBudgetMonitor


class FakeEnv:
    def __init__(self, width=0.042, step_count=0):
        self.gripper_width = width
        self.step_count = step_count


def test_hold_monitor_detects_drop_after_debounce():
    m = HoldMonitor(ControllerConfig(), debounce=2)
    env = FakeEnv(0.042)
    assert m.after_step(env, "transport") is None
    env.gripper_width = 0.002  # fingers closed on nothing
    assert m.after_step(env, "transport") is None
    assert m.after_step(env, "transport") == "object_dropped"


def test_hold_monitor_detects_opened_gripper():
    m = HoldMonitor(ControllerConfig(), debounce=1)
    assert m.after_step(FakeEnv(0.079), "lower") == "object_dropped"


def test_hold_monitor_ignores_non_hold_phases():
    m = HoldMonitor(ControllerConfig(), debounce=1)
    assert m.after_step(FakeEnv(0.0), "approach") is None
    assert m.after_step(FakeEnv(0.08), "release") is None


def test_step_budget_monitor():
    m = StepBudgetMonitor(100)
    assert m.after_step(FakeEnv(step_count=99), "x") is None
    assert m.after_step(FakeEnv(step_count=100), "x") == "episode_timeout"


def test_fault_injector_disabled_is_identity():
    f = FaultInjector(FaultConfig(enabled=False), seed=0)
    f.arm_attempt(0)
    a = np.array([0.1, 0, 0, 0, 0, 0, 1.0])
    for _ in range(30):
        assert np.array_equal(f.before_step(FakeEnv(), "transport", a, a), a)
    assert not np.any(f.grasp_offset(0))


def test_fault_injector_slip_opens_gripper_only_on_first_attempt():
    f = FaultInjector(FaultConfig(enabled=True, slip_probability=1.0), seed=0)
    f.arm_attempt(0)
    a = np.array([0.1, 0, 0, 0, 0, 0, 1.0])
    opened = sum(f.before_step(FakeEnv(), "transport", a, a)[-1] < 0 for _ in range(40))
    assert opened == 8
    assert "gripper_slip" in f.injected
    f.arm_attempt(1)  # retries are fault free
    assert all(f.before_step(FakeEnv(), "transport", a, a)[-1] > 0 for _ in range(40))


def test_grasp_offset_magnitude():
    f = FaultInjector(
        FaultConfig(enabled=True, grasp_offset_probability=1.0, grasp_offset_m=0.04), seed=1
    )
    off = f.grasp_offset(0)
    assert np.linalg.norm(off) == np.float64(0.04) or abs(np.linalg.norm(off) - 0.04) < 1e-9
    assert off[2] == 0.0
    assert not np.any(f.grasp_offset(1))
