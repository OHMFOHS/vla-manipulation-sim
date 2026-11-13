"""Module D: trajectory generation and OSC action mapping (no simulator)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotics.common.config import ControllerConfig
from robotics.control.osc import OSCCommander, top_down_orientation
from robotics.control.trajectory import max_step_speed, min_jerk_profile, min_jerk_segment


def test_min_jerk_profile_boundary_conditions():
    tau = np.linspace(0, 1, 101)
    s = min_jerk_profile(tau)
    assert s[0] == 0 and s[-1] == pytest.approx(1.0)
    assert np.all(np.diff(s) >= -1e-12)  # monotonic
    v = np.gradient(s, tau)
    assert abs(v[0]) < 0.01 and abs(v[-1]) < 0.01  # zero velocity at both ends


@pytest.mark.parametrize("dist", [0.01, 0.2, 0.6])
def test_segment_respects_speed_limit(dist):
    start, goal = np.zeros(3), np.array([dist, 0.0, 0.0])
    pts = min_jerk_segment(start, goal, max_speed=0.45, dt=0.05)
    assert np.allclose(pts[-1], goal)
    full = np.vstack([start, pts])
    assert max_step_speed(full, 0.05) <= 0.45 * 1.02


def test_segment_invalid_arguments():
    with pytest.raises(ValueError):
        min_jerk_segment(np.zeros(3), np.ones(3), max_speed=0.0, dt=0.05)


def test_position_action_is_bounded_and_directional():
    osc = OSCCommander(ControllerConfig())
    a = osc.position_action(np.zeros(3), np.array([1.0, -1.0, 0.0]))
    assert np.all(np.abs(a) <= 1.0)
    assert a[0] == 1.0 and a[1] == -1.0 and a[2] == 0.0
    small = osc.position_action(np.zeros(3), np.array([0.005, 0, 0]))
    assert 0 < small[0] < 1.0


def test_rotation_action_zero_at_target_and_bounded():
    osc = OSCCommander(ControllerConfig())
    target = top_down_orientation(0.3)
    q = target.as_quat()
    assert np.allclose(osc.rotation_action(q, target), 0.0, atol=1e-9)
    off = (Rotation.from_euler("z", 1.0) * target).as_quat()
    a = osc.rotation_action(off, target)
    assert np.all(np.abs(a) <= 1.0) and np.linalg.norm(a) > 0.5


def test_full_action_layout():
    osc = OSCCommander(ControllerConfig(), action_dim=7)
    a = osc.action(
        np.zeros(3),
        top_down_orientation().as_quat(),
        np.array([0.01, 0, 0]),
        top_down_orientation(),
        1.0,
    )
    assert a.shape == (7,) and a[-1] == 1.0
