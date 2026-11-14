"""Top-level facade wiring all modules together.

instruction ──► language (Module A) ──► TaskPlan
                                          │
camera ──► perception (Module B) ──► task planner (Module C) ──► skill / controller (Module D)
                ▲                                                     │
                └────────── closed-loop executor & recovery (Module E) ◄┘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from robotics.common.config import (
    PerceptionConfig,
    SimulationConfig,
    default_perception_config,
    default_simulation_config,
)
from robotics.control.skills import ScriptedPickPlaceController
from robotics.language import TaskPlan, parse_instruction
from robotics.perception.oracle import OraclePerception
from robotics.perception.vision import VisionPerceptionBackend
from robotics.planning.task_planner import TaskPlanner
from robotics.recovery.executor import ClosedLoopExecutor, EpisodeReport
from robotics.recovery.faults import FaultInjector
from robotics.recovery.monitors import CollisionCounter, FrameRecorder
from robotics.simulation.env import SimEnv

logger = logging.getLogger(__name__)


@dataclass
class AgentOptions:
    perception: str = "vision"  # vision | oracle
    controller: str = "scripted"  # scripted | bc
    checkpoint: str | None = None  # required for controller="bc"
    planner: str = "rule"  # rule | llm | auto
    gui: bool = False
    record_camera: str | None = None
    record_every: int = 4


class ManipulationAgent:
    def __init__(
        self,
        options: AgentOptions | None = None,
        sim_config: SimulationConfig | None = None,
        perception_config: PerceptionConfig | None = None,
        env: SimEnv | None = None,
    ):
        self.opts = options or AgentOptions()
        if self.opts.perception not in ("vision", "oracle"):
            raise ValueError("perception must be 'vision' or 'oracle'")
        if self.opts.controller not in ("scripted", "bc"):
            raise ValueError("controller must be 'scripted' or 'bc'")
        self.cfg = sim_config or default_simulation_config()
        self.pcfg = perception_config or default_perception_config()
        self.env = env or SimEnv(self.cfg.env, gui=self.opts.gui)
        if self.opts.perception == "vision" and not self.env.offscreen_available:
            raise ValueError(
                "Vision mode needs camera rendering, which the native macOS viewer (mjpython) "
                "disables. Use --perception oracle, or drop mjpython to get the OpenCV viewer."
            )
        self._bc_policy = None
        if self.opts.controller == "bc":
            from robotics.learning.policy import load_policy

            if not self.opts.checkpoint:
                raise ValueError("controller='bc' requires a checkpoint path")
            self._bc_policy = load_policy(self.opts.checkpoint)
        self.last_frames: list = []
        self.last_captions: list[str] = []
        self.last_collisions: CollisionCounter | None = None

    def make_perception(self):
        if self.opts.perception == "oracle":
            return OraclePerception(self.env)
        return VisionPerceptionBackend(self.env, self.pcfg)

    def run(self, instruction: str | TaskPlan, seed: int) -> EpisodeReport:
        plan = (
            instruction
            if isinstance(instruction, TaskPlan)
            else parse_instruction(instruction, self.opts.planner)
        )
        logger.info("Plan (%s): %s", plan.planner, plan.summary())
        self.env.reset(seed)
        motion = ScriptedPickPlaceController(self.env, self.cfg.controller)
        if self._bc_policy is not None:
            from robotics.learning.bc_skill import BCPickPlaceSkill

            skill = BCPickPlaceSkill(self.env, self._bc_policy, self.cfg.controller)
        else:
            skill = motion
        collisions = CollisionCounter()
        hooks = [collisions]
        recorder = None
        if self.opts.record_camera:
            recorder = FrameRecorder(self.opts.record_camera, every=self.opts.record_every)
            recorder.caption = plan.instruction
            hooks.append(recorder)
        faults = FaultInjector(self.cfg.faults, seed) if self.cfg.faults.enabled else None
        executor = ClosedLoopExecutor(
            self.env,
            self.make_perception(),
            skill,
            motion,
            TaskPlanner(grasp_depth=self.cfg.controller.grasp_depth),
            self.cfg.recovery,
            faults,
            extra_hooks=hooks,
        )
        report = executor.run(plan, seed=seed)
        # Ground truth (evaluation only): where did the objects really end up?
        per_object = {
            s.object_name: bool(self.env.oracle_is_in_container(s.object_name, s.target_location))
            for s in plan.steps
        }
        report.ground_truth = {
            "per_object_in_target": per_object,
            "all_in_target": all(per_object.values()),
            "robosuite_success": bool(self.env.oracle_task_success()),
            "collision_events": collisions.events,
            "collision_steps": collisions.contact_steps,
            "collision_pairs": sorted({f"{a}|{b}" for a, b in collisions.pairs})[:20],
        }
        self.last_frames = recorder.frames if recorder else []
        self.last_captions = recorder.captions if recorder else []
        self.last_collisions = collisions
        return report

    def close(self) -> None:
        self.env.close()
