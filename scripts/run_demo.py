"""Run the robot on a natural-language instruction.

Examples:
    uv run python scripts/run_demo.py
    uv run python scripts/run_demo.py --instruction "Move the milk first, then the cereal." --seed 3
    uv run python scripts/run_demo.py --perception oracle --record outputs/demo/run.gif
    uv run mjpython scripts/run_demo.py --gui          # interactive MuJoCo viewer (macOS)
    uv run python scripts/run_demo.py --gui            # OpenCV window fallback
    uv run python scripts/run_demo.py --controller bc  # learned policy (after train_bc.py)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--instruction", default="Sort all objects into their matching bins.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perception", choices=["vision", "oracle"], default="vision")
    p.add_argument("--controller", choices=["scripted", "bc"], default="scripted")
    p.add_argument("--checkpoint", default="outputs/checkpoints/bc_policy.pt")
    p.add_argument("--planner", choices=["rule", "llm", "auto"], default="rule")
    p.add_argument("--gui", action="store_true", help="show a live viewer")
    p.add_argument(
        "--record", default=None, help="write a GIF/MP4 of the run (e.g. outputs/demo/run.gif)"
    )
    p.add_argument("--record-camera", default="frontview")
    p.add_argument(
        "--faults", action="store_true", help="enable fault injection to exercise recovery"
    )
    p.add_argument("--out-dir", default="outputs/demo")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    from robotics.common.logging_utils import setup_logging

    setup_logging(args.log_level)
    import imageio.v2 as imageio

    from robotics.agent import AgentOptions, ManipulationAgent
    from robotics.common.config import default_simulation_config
    from robotics.language import PlanningError, parse_instruction
    from robotics.perception.vision import VisionPerceptionBackend
    from robotics.perception.visualize import draw_detections

    try:
        plan = parse_instruction(args.instruction, args.planner)
    except PlanningError as exc:
        print(f"Could not understand the instruction: {exc.message}")
        if exc.suggestions:
            print(f"Suggestions: {exc.suggestions}")
        return 2

    cfg = default_simulation_config()
    if args.faults:
        cfg.faults.enabled = True
        cfg.faults.slip_probability = 0.5
        cfg.faults.grasp_offset_probability = 0.3
    opts = AgentOptions(
        perception=args.perception,
        controller=args.controller,
        checkpoint=args.checkpoint if args.controller == "bc" else None,
        planner=args.planner,
        gui=args.gui,
        record_camera=args.record_camera if args.record else None,
        record_every=3,
    )
    try:
        agent = ManipulationAgent(opts, sim_config=cfg)
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    try:
        report = agent.run(plan, args.seed)
        # Snapshot of what the vision system sees at the end of the episode.
        if agent.env.offscreen_available:
            vis = VisionPerceptionBackend(agent.env)
            res = vis.perceive()
            imageio.imwrite(out / "perception_final.png", draw_detections(vis.last_frame.rgb, res))
        if args.record and agent.last_frames:
            rec = Path(args.record)
            rec = rec if rec.is_absolute() else ROOT / rec
            rec.parent.mkdir(parents=True, exist_ok=True)
            frames = agent.last_frames
            if rec.suffix == ".gif":
                imageio.mimsave(rec, frames[::2], duration=0.1, loop=0)
            else:
                imageio.mimsave(rec, frames, fps=20)
            print(f"recording: {rec}")
    finally:
        agent.close()

    with open(out / "last_report.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    print("\n=== Episode report ===")
    print(f"instruction : {plan.instruction}")
    print(f"plan        : {plan.summary()}")
    print(f"perception  : {report.mode} | controller: {report.controller} | seed: {args.seed}")
    for o in report.outcomes:
        gt = report.ground_truth["per_object_in_target"][o.object_name]
        print(
            f"  {o.object_name:7s} -> {o.target:11s} status={o.final_status:24s} attempts={o.attempts} "
            f"recoveries={o.recovery_attempts} ground_truth={'OK' if gt else 'FAIL'} failures={o.failures}"
        )
    print(f"success (ground truth): {report.ground_truth['all_in_target']}")
    print(
        f"sim time {report.sim_time:.1f}s, {report.control_steps} control steps, wall {report.wall_time:.1f}s"
    )
    print(f"report: {out / 'last_report.json'}")
    return 0 if report.ground_truth["all_in_target"] else 1


if __name__ == "__main__":
    sys.exit(main())
