"""Configuration loading and benchmark metric aggregation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from robotics.common.config import (
    SimulationConfig,
    default_perception_config,
    default_simulation_config,
    default_training_config,
)
from robotics.evaluation.metrics import episode_row, summarize
from robotics.recovery.executor import EpisodeReport, StepOutcome


def test_yaml_configs_validate():
    sim = default_simulation_config()
    assert sim.recovery.max_recovery_attempts == 2
    assert sim.faults.enabled is False
    assert len(default_perception_config().classes) == 4
    assert default_training_config().val_fraction > 0


def test_config_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        SimulationConfig.model_validate({"env": {"not_a_field": 1}})


def _report(seed, placed: dict, recoveries=0, collisions=0, success=None):
    outcomes = [
        StepOutcome(o, f"{o}_bin", success=ok, attempts=1 + recoveries, recovery_attempts=recoveries,
                    manipulation_actions=1 + recoveries, pick_successes=1 if ok else 0,
                    failures=[] if ok else ["grasp_empty"])
        for o, ok in placed.items()
    ]  # fmt: skip
    return EpisodeReport(
        instruction="x", mode="vision", controller="scripted", seed=seed, plan=[], outcomes=outcomes,
        success=all(placed.values()) if success is None else success, aborted=False, abort_reason=None,
        control_steps=1000, sim_time=50.0, wall_time=5.0, events=[],
        ground_truth={"per_object_in_target": placed, "all_in_target": all(placed.values()),
                      "robosuite_success": all(placed.values()), "collision_events": collisions},
    )  # fmt: skip


def test_summary_metrics():
    reports = [
        _report(0, {"milk": True, "can": True}),
        _report(1, {"milk": True, "can": False}, recoveries=2, collisions=1),
        _report(
            2, {"milk": True, "can": True}, success=False
        ),  # belief disagrees with ground truth
    ]
    s = summarize("cfg", reports)
    assert s["episodes"] == 3
    assert s["task_success_rate"] == pytest.approx(2 / 3)
    assert s["placement_success_rate"] == pytest.approx(5 / 6)
    assert s["avg_recovery_attempts"] == pytest.approx(4 / 3)
    assert s["collision_rate"] == pytest.approx(1 / 3)
    assert s["failure_rate"] == pytest.approx(1 / 3)
    assert s["belief_ground_truth_agreement"] == pytest.approx(2 / 3)
    assert s["per_object"]["can"]["placement_success_rate"] == pytest.approx(2 / 3)
    assert s["failure_reasons"]["grasp_empty"] >= 1
    row = episode_row("cfg", reports[1])
    assert row["objects_placed"] == 1 and row["recovery_attempts"] == 4


def test_summary_requires_episodes():
    with pytest.raises(ValueError):
        summarize("cfg", [])
