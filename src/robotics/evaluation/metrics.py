"""Benchmark metrics computed from :class:`EpisodeReport` objects.

All success numbers use simulator ground truth (``report.ground_truth``), not the robot's own
belief, so a perception error that makes the robot *think* it succeeded is still counted as a
failure.  The agreement between belief and ground truth is reported separately.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from robotics.recovery.executor import EpisodeReport
from robotics.simulation.scene import OBJECT_NAMES


def episode_row(config: str, r: EpisodeReport) -> dict:
    gt = r.ground_truth
    per = gt.get("per_object_in_target", {})
    return {
        "config": config,
        "seed": r.seed,
        "mode": r.mode,
        "controller": r.controller,
        "task_success": bool(gt.get("all_in_target", False)),
        "robosuite_success": bool(gt.get("robosuite_success", False)),
        "self_reported_success": bool(r.success),
        "objects_requested": len(per),
        "objects_placed": int(sum(per.values())),
        "recovery_attempts": r.recovery_attempts,
        "manipulation_actions": r.manipulation_actions,
        "control_steps": r.control_steps,
        "sim_time_s": round(r.sim_time, 2),
        "wall_time_s": round(r.wall_time, 2),
        "collision_events": gt.get("collision_events", 0),
        "aborted": r.aborted,
        "faults_injected": len(r.faults_injected),
        "failure_reasons": ";".join(f for o in r.outcomes for f in o.failures),
    }


def summarize(config: str, reports: list[EpisodeReport]) -> dict:
    if not reports:
        raise ValueError("no episodes to summarise")
    rows = [episode_row(config, r) for r in reports]
    n = len(rows)

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    pick_attempts: dict[str, int] = defaultdict(int)
    pick_success: dict[str, int] = defaultdict(int)
    placed: dict[str, int] = defaultdict(int)
    requested: dict[str, int] = defaultdict(int)
    first_try: dict[str, int] = defaultdict(int)
    failure_counts: dict[str, int] = defaultdict(int)
    for r in reports:
        per = r.ground_truth.get("per_object_in_target", {})
        for o in r.outcomes:
            pick_attempts[o.object_name] += o.manipulation_actions
            pick_success[o.object_name] += o.pick_successes
            requested[o.object_name] += 1
            placed[o.object_name] += int(per.get(o.object_name, False))
            first_try[o.object_name] += int(
                per.get(o.object_name, False) and o.recovery_attempts == 0
            )
            for f in o.failures:
                failure_counts[f.split(":")[0]] += 1

    per_object = {
        name: {
            "pick_attempts": pick_attempts[name],
            "pick_success_rate": (pick_success[name] / pick_attempts[name])
            if pick_attempts[name]
            else None,
            "placement_success_rate": (placed[name] / requested[name]) if requested[name] else None,
            "first_attempt_success_rate": (first_try[name] / requested[name])
            if requested[name]
            else None,
        }
        for name in OBJECT_NAMES
        if requested[name]
    }
    total_attempts = sum(pick_attempts.values())
    total_requested = sum(requested.values())
    agree = np.mean([row["self_reported_success"] == row["task_success"] for row in rows])
    return {
        "config": config,
        "episodes": n,
        "task_success_rate": mean("task_success"),
        "robosuite_success_rate": mean("robosuite_success"),
        "pick_success_rate": (sum(pick_success.values()) / total_attempts)
        if total_attempts
        else 0.0,
        "placement_success_rate": (sum(placed.values()) / total_requested)
        if total_requested
        else 0.0,
        "avg_episode_sim_time_s": mean("sim_time_s"),
        "avg_episode_wall_time_s": mean("wall_time_s"),
        "avg_recovery_attempts": mean("recovery_attempts"),
        "collision_rate": float(np.mean([row["collision_events"] > 0 for row in rows])),
        "avg_collision_events": mean("collision_events"),
        "failure_rate": 1.0 - mean("task_success"),
        "avg_manipulation_actions": mean("manipulation_actions"),
        "avg_control_steps": mean("control_steps"),
        "abort_rate": mean("aborted"),
        "belief_ground_truth_agreement": float(agree),
        "per_object": per_object,
        "failure_reasons": dict(sorted(failure_counts.items(), key=lambda kv: -kv[1])),
    }
