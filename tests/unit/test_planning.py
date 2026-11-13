"""Module C: task planner and placement selection (synthetic perception)."""

from __future__ import annotations

import numpy as np

from robotics.language import RuleBasedTaskParser
from robotics.perception.types import Detection, PerceptionResult
from robotics.planning.placement import select_place_point
from robotics.planning.task_planner import StepStatus, TaskPlanner
from robotics.simulation.scene import CONTAINERS, OBJECT_NAMES, WORKSPACE, Workspace


def perception(conf: dict | None = None, **positions) -> PerceptionResult:
    conf = conf or {}
    dets = {
        n: Detection(n, np.array(p, float), p[2] + 0.04, confidence=conf.get(n, 1.0))
        for n, p in positions.items()
    }
    return PerceptionResult("vision", dets, [n for n in OBJECT_NAMES if n not in dets])


def plan_for(text):
    return RuleBasedTaskParser().parse(text)


def test_ready_task_has_full_action_sequence():
    p = plan_for("Pick up the milk")
    task = TaskPlanner().plan_step(p.steps[0], perception(milk=[0.1, -0.25, 0.865]))
    assert task.status == StepStatus.READY
    assert [a.kind for a in task.actions] == [
        "locate",
        "estimate_pose",
        "select_container",
        "pick",
        "place",
        "verify",
    ]
    assert CONTAINERS["milk_bin"].contains(task.place.position)
    assert np.allclose(task.grasp.position, [0.1, -0.25, 0.865])


def test_missing_object():
    task = TaskPlanner().plan_step(
        plan_for("move the can").steps[0], perception(milk=[0.1, -0.25, 0.865])
    )
    assert task.status == StepStatus.OBJECT_NOT_FOUND


def test_object_already_in_target_is_not_repeated():
    c = CONTAINERS["bread_bin"]
    task = TaskPlanner().plan_step(
        plan_for("move the bread").steps[0], perception(bread=[c.center[0], c.center[1], 0.85])
    )
    assert task.status == StepStatus.ALREADY_SATISFIED


def test_unreachable_object():
    task = TaskPlanner().plan_step(
        plan_for("move the milk").steps[0], perception(milk=[0.6, 0.6, 0.86])
    )
    assert task.status == StepStatus.OBJECT_UNREACHABLE


def test_unreachable_container():
    tiny_reach = Workspace(comfortable_reach=0.3)
    task = TaskPlanner(tiny_reach).plan_step(
        plan_for("move the can").steps[0], perception(can=[0.0, -0.2, 0.86])
    )
    assert task.status == StepStatus.TARGET_UNREACHABLE


def test_low_confidence_detection():
    task = TaskPlanner().plan_step(
        plan_for("move the milk").steps[0], perception(conf={"milk": 0.1}, milk=[0.1, -0.25, 0.865])
    )
    assert task.status == StepStatus.LOW_CONFIDENCE


def test_plan_multiple_objects_in_order():
    pr = perception(milk=[0.1, -0.25, 0.865], cereal=[0.0, -0.3, 0.87])
    tasks = TaskPlanner().plan(plan_for("Move the milk first, then the cereal."), pr)
    assert [t.step.object_name for t in tasks] == ["milk", "cereal"]
    assert all(t.executable for t in tasks)


def test_place_points_are_reachable_and_inside():
    for name, c in CONTAINERS.items():
        p = select_place_point(c)
        assert p is not None, name
        assert c.contains(p)
        assert WORKSPACE.horizontal_reach(p) <= WORKSPACE.comfortable_reach + 1e-9


def test_place_point_avoids_occupied_spots():
    c = CONTAINERS["milk_bin"]
    first = select_place_point(c)
    second = select_place_point(c, occupied=[first])
    assert np.linalg.norm(second[:2] - first[:2]) >= 0.06 - 1e-9
