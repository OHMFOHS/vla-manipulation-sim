"""Module B on rendered MuJoCo images: vision agrees with the oracle without reading it."""

from __future__ import annotations

import numpy as np

from robotics.control.skills import ScriptedPickPlaceController
from robotics.perception.oracle import OraclePerception
from robotics.perception.vision import VisionPerception, VisionPerceptionBackend
from robotics.simulation.scene import OBJECT_NAMES, WORKSPACE


def test_vision_matches_oracle_on_rendered_frames(sim_env):
    errors = []
    for seed in (0, 5, 9):
        sim_env.reset(seed)
        ScriptedPickPlaceController(sim_env).go_to_pose(WORKSPACE.perception_pose, "perception")
        vision = VisionPerceptionBackend(sim_env).perceive()
        oracle = OraclePerception(sim_env).perceive()
        assert vision.missing == []
        for n in OBJECT_NAMES:
            errors.append(np.linalg.norm(vision.get(n).position - oracle.get(n).position))
            assert abs(vision.get(n).top_z - oracle.get(n).top_z) < 0.01
    assert max(errors) < 0.01  # < 1 cm everywhere
    assert np.mean(errors) < 0.004


def test_vision_backend_only_uses_sensor_frames(sim_env):
    """The detector works on a *copy* of the frame; the backend never stores the env."""
    sim_env.reset(0)
    backend = VisionPerceptionBackend(sim_env)
    assert not hasattr(backend, "env")
    frame = sim_env.capture_rgbd()
    # Detection is a pure function of the frame
    r1 = VisionPerception().detect(frame)
    sim_env.reset(3)  # change the world - must not influence detections on the old frame
    r2 = VisionPerception().detect(frame)
    for n in r1.detections:
        assert np.allclose(r1.get(n).position, r2.get(n).position)
