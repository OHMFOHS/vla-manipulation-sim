"""Vision-mode localisation accuracy against simulator ground truth."""

from __future__ import annotations

import numpy as np

from robotics.control.skills import ScriptedPickPlaceController
from robotics.perception.oracle import OraclePerception
from robotics.perception.vision import VisionPerceptionBackend
from robotics.simulation.scene import OBJECT_NAMES, WORKSPACE


def evaluate_perception(env, seeds: list[int]) -> dict:
    """Detect all objects at the start of each seed and compare with ground truth."""
    errors: dict[str, list[float]] = {n: [] for n in OBJECT_NAMES}
    top_errors: list[float] = []
    detected = total = 0
    for seed in seeds:
        env.reset(seed)
        ScriptedPickPlaceController(env).go_to_pose(WORKSPACE.perception_pose, "perception")
        vision = VisionPerceptionBackend(env).perceive()
        oracle = OraclePerception(env).perceive()  # ground truth for scoring only
        for name in OBJECT_NAMES:
            total += 1
            det = vision.get(name)
            if det is None:
                continue
            detected += 1
            errors[name].append(float(np.linalg.norm(det.position - oracle.get(name).position)))
            top_errors.append(abs(det.top_z - oracle.get(name).top_z))
    all_err = np.concatenate([np.asarray(v) for v in errors.values() if v])
    return {
        "frames": len(seeds),
        "detection_rate": detected / total,
        "position_error_mm": {
            "mean": 1000 * float(all_err.mean()),
            "median": 1000 * float(np.median(all_err)),
            "p95": 1000 * float(np.percentile(all_err, 95)),
            "max": 1000 * float(all_err.max()),
        },
        "top_height_error_mm_mean": 1000 * float(np.mean(top_errors)),
        "per_object_mean_error_mm": {k: 1000 * float(np.mean(v)) for k, v in errors.items() if v},
    }
