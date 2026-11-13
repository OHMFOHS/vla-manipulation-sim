"""Demonstration collection with the scripted expert.

The full closed-loop system (perception → planner → scripted skill → recovery) runs as usual.
A recording wrapper around the skill logs ``(observation, action, timestamp, task_id)`` for
every control step of every pick-and-place attempt.  Only attempts that ended with the object
in its target container (checked with simulator ground truth - acceptable for *offline* data
curation) are kept.

DART-style noise injection: in a fraction of episodes the *executed* translation action is
perturbed with Gaussian noise while the *recorded* label stays the expert's clean action.
The expert then demonstrates how to recover from small deviations, which reduces
compounding error at deployment time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robotics.agent import AgentOptions, ManipulationAgent
from robotics.common.config import (
    PROJECT_ROOT,
    SimulationConfig,
    TrainingConfig,
    default_simulation_config,
)
from robotics.control.skills import ControlHook, GraspTarget, PlaceTarget, SkillResult
from robotics.learning.features import HeldLatch, VelocityEstimator, build_features

logger = logging.getLogger(__name__)


class DemoRecorder(ControlHook):
    """Records features + clean expert action for each step while ``active``."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.active = False
        self.grasp: GraspTarget | None = None
        self.place: PlaceTarget | None = None
        self.prev_cmd = -1.0
        self.latch: HeldLatch | None = None
        self.vel: VelocityEstimator | None = None
        self.obs: list[np.ndarray] = []
        self.act: list[np.ndarray] = []
        self.time: list[float] = []

    def start(self, grasp: GraspTarget, place: PlaceTarget) -> None:
        self.active = True
        self.grasp, self.place = grasp, place
        self.prev_cmd = -1.0
        self.latch = HeldLatch(self.cfg.grasp_min_width, self.cfg.grasp_max_width)
        self.vel = None  # created lazily with the env's control period
        self.obs, self.act, self.time = [], [], []

    def stop(self) -> None:
        self.active = False

    def before_step(self, env, phase, clean_action, action):
        if self.active:
            latch = self.latch.update(env.gripper_width, self.prev_cmd)
            if self.vel is None:
                self.vel = VelocityEstimator(env.control_dt)
            self.obs.append(
                build_features(
                    env.eef_pos,
                    self.vel.update(env.eef_pos),
                    env.gripper_width,
                    self.grasp.position,
                    self.grasp.top_z,
                    self.place.position,
                    self.prev_cmd,
                    latch,
                    self.grasp.object_name,
                )
            )
            label = np.array(
                [*clean_action[:3], 1.0 if clean_action[-1] > 0 else -1.0], dtype=np.float32
            )
            self.act.append(label)
            self.time.append(env.sim_time)
            self.prev_cmd = float(clean_action[-1])
        return action


class ActionNoise(ControlHook):
    """Perturbs executed translation actions (labels stay clean)."""

    def __init__(self, std: float, seed: int):
        self.std = std
        self.rng = np.random.default_rng(seed + 104729)
        self.enabled = False

    def before_step(self, env, phase, clean_action, action):
        if not self.enabled or self.std <= 0:
            return action
        a = action.copy()
        a[:3] = np.clip(a[:3] + self.rng.normal(0.0, self.std, 3), -1.0, 1.0)
        return a


@dataclass
class _Buffer:
    obs: list = field(default_factory=list)
    act: list = field(default_factory=list)
    timestamp: list = field(default_factory=list)
    task_id: list = field(default_factory=list)
    episode: list = field(default_factory=list)
    step_index: list = field(default_factory=list)
    object_id: list = field(default_factory=list)
    noisy: list = field(default_factory=list)


class RecordingSkill:
    """Wraps the scripted skill: records each attempt and keeps the successful ones."""

    name = "scripted_recording"

    def __init__(
        self, inner, recorder: DemoRecorder, env, buffer: _Buffer, episode: int, noisy: bool
    ):
        self.inner = inner
        self.recorder = recorder
        self.env = env
        self.buffer = buffer
        self.episode = episode
        self.noisy = noisy
        self.attempt_counter = 0
        self.kept = 0
        self.discarded = 0

    @property
    def hooks(self):
        return self.inner.hooks

    @hooks.setter
    def hooks(self, value):
        self.inner.hooks = value

    @property
    def cfg(self):
        return self.inner.cfg

    def pick_and_place(self, grasp: GraspTarget, place: PlaceTarget) -> SkillResult:
        from robotics.simulation.scene import OBJECT_NAMES

        self.recorder.start(grasp, place)
        result = self.inner.pick_and_place(grasp, place)
        self.recorder.stop()
        task_id = self.episode * 100 + self.attempt_counter
        self.attempt_counter += 1
        ok = result.success and self.env.oracle_is_in_container(
            grasp.object_name, place.container_name
        )
        if ok and self.recorder.obs:
            n = len(self.recorder.obs)
            b = self.buffer
            b.obs += self.recorder.obs
            b.act += self.recorder.act
            t0 = self.recorder.time[0]
            b.timestamp += [t - t0 for t in self.recorder.time]
            b.task_id += [task_id] * n
            b.episode += [self.episode] * n
            b.step_index += list(range(n))
            b.object_id += [OBJECT_NAMES.index(grasp.object_name)] * n
            b.noisy += [self.noisy] * n
            self.kept += 1
        else:
            self.discarded += 1
        return result


def collect_demos(
    n_episodes: int,
    out_path: str | Path,
    seed_start: int = 0,
    perception: str = "vision",
    training_cfg: TrainingConfig | None = None,
    sim_cfg: SimulationConfig | None = None,
    instruction: str = "Sort all objects into their matching bins.",
) -> dict:
    tcfg = training_cfg or TrainingConfig()
    scfg = sim_cfg or default_simulation_config()
    agent = ManipulationAgent(AgentOptions(perception=perception), sim_config=scfg)
    buffer = _Buffer()
    kept = discarded = 0
    rng = np.random.default_rng(tcfg.seed)
    try:
        for ep in range(n_episodes):
            seed = seed_start + ep
            noisy = bool(rng.random() < tcfg.collect_noise_fraction)
            recorder = DemoRecorder(scfg.controller)
            noise = ActionNoise(tcfg.collect_noise_std, seed)
            noise.enabled = noisy
            holder: dict = {}

            # Wrap the executor's skill so this episode's attempts are recorded.
            def make_skill(
                env, inner, _ep=ep, _noisy=noisy, _rec=recorder, _noise=noise, _holder=holder
            ):
                inner.hooks = [*inner.hooks, _noise, _rec]
                s = RecordingSkill(inner, _rec, env, buffer, _ep, _noisy)
                _holder["skill"] = s
                return s

            _run_with_skill(agent, instruction, seed, make_skill)
            s = holder["skill"]
            kept += s.kept
            discarded += s.discarded
            if (ep + 1) % 10 == 0 or ep == n_episodes - 1:
                logger.warning(
                    "collected %d/%d episodes, kept %d trajectories (%d discarded), %d samples",
                    ep + 1,
                    n_episodes,
                    kept,
                    discarded,
                    len(buffer.obs),
                )
    finally:
        agent.close()

    out = Path(out_path)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        observations=np.asarray(buffer.obs, dtype=np.float32),
        actions=np.asarray(buffer.act, dtype=np.float32),
        timestamps=np.asarray(buffer.timestamp, dtype=np.float32),
        task_ids=np.asarray(buffer.task_id, dtype=np.int64),
        episodes=np.asarray(buffer.episode, dtype=np.int64),
        step_index=np.asarray(buffer.step_index, dtype=np.int64),
        object_ids=np.asarray(buffer.object_id, dtype=np.int64),
        noisy=np.asarray(buffer.noisy, dtype=bool),
        seeds=np.arange(seed_start, seed_start + n_episodes),
    )
    stats = {
        "episodes": n_episodes,
        "trajectories_kept": kept,
        "trajectories_discarded": discarded,
        "samples": len(buffer.obs),
        "path": str(out),
    }
    logger.warning("saved demos: %s", stats)
    return stats


def _run_with_skill(agent: ManipulationAgent, instruction: str, seed: int, make_skill) -> object:
    """Run one agent episode with a custom skill wrapper (mirrors ManipulationAgent.run)."""
    from robotics.control.skills import ScriptedPickPlaceController
    from robotics.language import parse_instruction
    from robotics.planning.task_planner import TaskPlanner
    from robotics.recovery.executor import ClosedLoopExecutor

    plan = parse_instruction(instruction)
    agent.env.reset(seed)
    motion = ScriptedPickPlaceController(agent.env, agent.cfg.controller)
    skill_inner = ScriptedPickPlaceController(agent.env, agent.cfg.controller)
    executor = ClosedLoopExecutor(
        agent.env,
        agent.make_perception(),
        skill_inner,
        motion,
        TaskPlanner(),
        agent.cfg.recovery,
        None,
    )
    # The executor installed its monitors on skill_inner; now add recording hooks + wrapper.
    executor.skill = make_skill(agent.env, skill_inner)
    return executor.run(plan, seed=seed)
