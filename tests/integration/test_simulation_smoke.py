"""Headless simulation smoke tests (real robosuite + MuJoCo)."""

from __future__ import annotations

import numpy as np

from robotics.simulation.scene import BIN_FLOOR_Z, OBJECT_NAMES


def test_reset_is_deterministic(sim_env):
    sim_env.reset(123)
    a = sim_env.oracle_object_positions()
    q1 = sim_env.eef_pos.copy()
    sim_env.reset(456)
    sim_env.reset(123)
    b = sim_env.oracle_object_positions()
    for n in OBJECT_NAMES:
        assert np.allclose(a[n], b[n], atol=1e-6)
    assert np.allclose(q1, sim_env.eef_pos, atol=1e-6)


def test_objects_start_in_source_bin(sim_env):
    sim_env.reset(0)
    for n, p in sim_env.oracle_object_positions().items():
        assert sim_env.oracle_is_in_container(n, "source_bin"), (n, p)
        assert p[2] > BIN_FLOOR_Z


def test_step_and_action_frame(sim_env):
    """+x action moves the end effector along world +x (base frame == world frame)."""
    sim_env.reset(0)
    p0 = sim_env.eef_pos.copy()
    a = sim_env.hold_action()
    a[0] = 0.5
    for _ in range(10):
        sim_env.step(a)
    d = sim_env.eef_pos - p0
    assert d[0] > 0.03 and abs(d[1]) < 0.01
    assert sim_env.step_count == 10


def test_rgbd_capture(sim_env):
    sim_env.reset(0)
    f = sim_env.capture_rgbd()
    h, w = f.rgb.shape[:2]
    assert f.rgb.dtype == np.uint8 and f.depth.shape == (h, w)
    assert np.isfinite(f.depth).all()
    # the bin floor is ~1.18 m below the overhead camera
    centre = f.depth[h // 2 - 5 : h // 2 + 5, w // 2 - 5 : w // 2 + 5]
    assert 1.0 < float(np.median(centre)) < 1.25
    assert f.intrinsics.shape == (3, 3) and f.cam_to_world.shape == (4, 4)


def test_invalid_action_shape(sim_env):
    import pytest

    with pytest.raises(ValueError):
        sim_env.step(np.zeros(3))
