"""Save an annotated RGB-D perception snapshot and print vision-vs-oracle errors.

uv run python scripts/perception_snapshot.py --seed 10000 --out assets/demo/perception.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--out", default="assets/demo/perception.png")
    p.add_argument(
        "--accuracy-episodes",
        type=int,
        default=0,
        help="also measure localisation error over N evaluation seeds (writes JSON)",
    )
    args = p.parse_args()

    from robotics.common.logging_utils import setup_logging

    setup_logging("WARNING")
    import cv2
    import numpy as np

    from robotics.control.skills import ScriptedPickPlaceController
    from robotics.perception.oracle import OraclePerception
    from robotics.perception.vision import VisionPerceptionBackend
    from robotics.perception.visualize import depth_colormap, draw_detections
    from robotics.simulation.env import SimEnv
    from robotics.simulation.scene import WORKSPACE

    env = SimEnv()
    env.reset(args.seed)
    ScriptedPickPlaceController(env).go_to_pose(WORKSPACE.perception_pose, "perception")
    backend = VisionPerceptionBackend(env)
    vision = backend.perceive()
    oracle = OraclePerception(env).perceive()  # only to report the error, never fed to vision
    frame = backend.last_frame
    left = draw_detections(frame.rgb, vision, scale=2)
    right = cv2.resize(
        depth_colormap(frame.depth), (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    for img, text in (
        (left, "RGB + detections (vision mode)"),
        (right, "metric depth (0.9-1.3 m)"),
    ):
        cv2.putText(img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            img, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(np.concatenate([left, right], axis=1), cv2.COLOR_RGB2BGR))
    print(f"{'object':8s} {'vision xyz (m)':28s} {'error vs oracle (mm)':>20s} {'conf':>5s}")
    for name, det in vision.detections.items():
        err = 1000 * np.linalg.norm(det.position - oracle.get(name).position)
        print(
            f"{name:8s} {np.array2string(det.position, precision=3):28s} {err:20.1f} {det.confidence:5.2f}"
        )
    print(f"missing: {vision.missing}\nwrote {out}")
    if args.accuracy_episodes:
        import json

        from robotics.evaluation.perception_eval import evaluate_perception
        from robotics.evaluation.runner import eval_seeds

        acc = evaluate_perception(env, eval_seeds(args.accuracy_episodes))
        path = ROOT / "outputs" / "perception_accuracy.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(acc, indent=2))
        print(json.dumps(acc, indent=2))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
