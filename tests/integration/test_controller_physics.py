"""Module D on real physics: the scripted skill moves objects only through contact."""

from __future__ import annotations

import numpy as np

from robotics.control.skills import GraspTarget, PlaceTarget, ScriptedPickPlaceController
from robotics.perception.oracle import OraclePerception
from robotics.planning.placement import select_place_point
from robotics.simulation.scene import CONTAINERS, WORKSPACE


def test_pick_and_place_milk_through_physics(sim_env):
    sim_env.reset(1)
    ctrl = ScriptedPickPlaceController(sim_env)
    det = OraclePerception(sim_env).perceive().get("milk")
    start = sim_env.oracle_object_position("milk").copy()

    r = ctrl.pick(GraspTarget("milk", det.position, det.top_z, det.yaw))
    assert r.success, r
    lifted = sim_env.oracle_object_position("milk")
    assert lifted[2] > start[2] + 0.1  # really lifted by the gripper
    assert np.linalg.norm(lifted - sim_env.eef_pos) < 0.08  # and moving with the hand
    assert ctrl.is_holding()

    place = select_place_point(CONTAINERS["milk_bin"])
    r = ctrl.place(PlaceTarget("milk_bin", place))
    assert r.success, r
    assert sim_env.oracle_is_in_container("milk", "milk_bin")
    assert sim_env.robot_environment_contacts() == []


def test_empty_grasp_is_detected(sim_env):
    """Grasping 6 cm next to the object closes on air and must be reported."""
    sim_env.reset(2)
    ctrl = ScriptedPickPlaceController(sim_env)
    det = OraclePerception(sim_env).perceive().get("bread")
    # choose an offset direction that points away from the other objects
    others = [p for n, p in sim_env.oracle_object_positions().items() if n != "bread"]
    best, best_d = None, -1.0
    for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        cand = det.position + 0.07 * np.array([np.cos(ang), np.sin(ang), 0.0])
        d = min(np.linalg.norm(cand[:2] - o[:2]) for o in others)
        if d > best_d and CONTAINERS["source_bin"].contains(cand, margin=0.03):
            best, best_d = cand, d
    r = ctrl.pick(GraspTarget("bread", best, det.top_z, 0.0))
    assert not r.success and r.reason == "grasp_empty"


def test_motion_respects_speed_limit_and_timeout(sim_env):
    sim_env.reset(0)
    ctrl = ScriptedPickPlaceController(sim_env)
    positions = []

    class Spy:
        def before_step(self, env, phase, clean, action):
            assert np.all(np.abs(action) <= 1.0)
            return action

        def after_step(self, env, phase):
            positions.append(env.eef_pos.copy())

    ctrl.hooks = [Spy()]
    r = ctrl.move_to(np.array([0.1, -0.25, 1.0]), "test")
    assert r.success
    speeds = np.linalg.norm(np.diff(np.array(positions), axis=0), axis=1) / sim_env.control_dt
    assert speeds.max() < 0.75  # min-jerk segments with a 0.45 m/s set-point limit
    r = ctrl.move_to(
        np.array([0.1, -0.25, 1.0]) + [0, 0, 5.0], "unreachable", timeout=15, clip=False
    )
    assert not r.success and r.reason == "timeout:unreachable"


def test_perception_pose_does_not_occlude_bins(sim_env):
    sim_env.reset(0)
    ctrl = ScriptedPickPlaceController(sim_env)
    assert ctrl.go_to_pose(WORKSPACE.perception_pose, "perception").success
