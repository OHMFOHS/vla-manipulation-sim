"""Module E - closed-loop execution with failure detection and recovery.

For every step of a :class:`TaskPlan` the executor runs

    perceive → plan → pick & place (monitored) → perceive again → verify placement

and on any failure it records the reason, puts the robot in a safe state, re-perceives,
re-plans and retries - at most ``max_recovery_attempts`` times per step.  Steps that cannot
be recovered are terminated safely and the episode continues with the remaining steps
(partial failure), unless the global step budget is exhausted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from robotics.common.config import RecoveryConfig
from robotics.control.skills import ControlHook, SkillResult
from robotics.language.schema import TaskPlan, TaskStep
from robotics.perception.types import PerceptionResult
from robotics.planning.task_planner import StepStatus, TaskPlanner
from robotics.recovery.faults import FaultInjector
from robotics.recovery.monitors import HoldMonitor, StepBudgetMonitor
from robotics.simulation.scene import CONTAINERS, WORKSPACE

logger = logging.getLogger(__name__)

# Planner statuses that a fresh look at the scene might fix.
_RETRYABLE_STATUSES = {StepStatus.OBJECT_NOT_FOUND, StepStatus.LOW_CONFIDENCE}


@dataclass
class StepOutcome:
    object_name: str
    target: str
    success: bool = False
    attempts: int = 0
    recovery_attempts: int = 0
    manipulation_actions: int = 0  # pick-and-place executions
    pick_successes: int = 0
    failures: list[str] = field(default_factory=list)
    final_status: str = "pending"
    steps: int = 0


@dataclass
class EpisodeReport:
    instruction: str
    mode: str
    controller: str
    seed: int | None
    plan: list[dict]
    outcomes: list[StepOutcome]
    success: bool
    aborted: bool
    abort_reason: str | None
    control_steps: int
    sim_time: float
    wall_time: float
    events: list[dict]
    faults_injected: list[str] = field(default_factory=list)
    ground_truth: dict = field(default_factory=dict)

    @property
    def recovery_attempts(self) -> int:
        return sum(o.recovery_attempts for o in self.outcomes)

    @property
    def manipulation_actions(self) -> int:
        return sum(o.manipulation_actions for o in self.outcomes)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["recovery_attempts"] = self.recovery_attempts
        d["manipulation_actions"] = self.manipulation_actions
        return d


class ClosedLoopExecutor:
    def __init__(
        self,
        env,
        perception,
        skill,
        motion,
        planner: TaskPlanner | None = None,
        recovery: RecoveryConfig | None = None,
        faults: FaultInjector | None = None,
        extra_hooks: list[ControlHook] | None = None,
    ):
        """
        Args:
            env: :class:`SimEnv`.
            perception: oracle or vision backend (``perceive() -> PerceptionResult``).
            skill: object with ``pick_and_place(grasp, place)`` (scripted or learned).
            motion: scripted controller used for utility motions (perception pose, safe stop).
        """
        self.env = env
        self.perception = perception
        self.skill = skill
        self.motion = motion
        self.planner = planner or TaskPlanner()
        self.cfg = recovery or RecoveryConfig()
        self.faults = faults
        self.events: list[dict] = []
        hooks: list[ControlHook] = []
        if faults is not None:
            hooks.append(faults)
        hooks += [HoldMonitor(motion.cfg), StepBudgetMonitor(self.cfg.episode_step_budget)]
        hooks += list(extra_hooks or [])
        for ctrl in {id(skill): skill, id(motion): motion}.values():
            ctrl.hooks = list(hooks)
        self._occupied: dict[str, list[np.ndarray]] = {}

    # ------------------------------------------------------------------ logging
    def _log(
        self, event: str, step: TaskStep | None = None, attempt: int | None = None, **detail
    ) -> None:
        rec = {
            "t": round(self.env.sim_time, 3),
            "control_step": self.env.step_count,
            "event": event,
            "object": step.object_name if step else None,
            "attempt": attempt,
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in detail.items()},
        }
        self.events.append(rec)
        logger.info(
            "%-24s %-7s %s",
            event,
            step.object_name if step else "-",
            " ".join(f"{k}={v}" for k, v in detail.items() if k not in ("perception",)),
        )

    # ------------------------------------------------------------------ helpers
    def _perceive(self) -> PerceptionResult:
        r = self.motion.go_to_pose(WORKSPACE.perception_pose, "to_perception_pose")
        if not r.success and r.reason == "episode_timeout":
            raise _EpisodeTimeout
        result = self.perception.perceive()
        self._log(
            "perceive",
            detected=sorted(result.detections),
            missing=result.missing,
        )
        return result

    def _verify_placement(self, step: TaskStep, perception: PerceptionResult) -> tuple[bool, str]:
        container = CONTAINERS[step.target_location]
        det = perception.get(step.object_name)
        if det is None:
            return False, "object_not_visible_after_place"
        if container.contains(det.position, margin=0.0):
            return True, "ok"
        return False, "not_in_target_container"

    def _safe_stop(self, step: TaskStep, attempt: int) -> None:
        r = self.motion.release_and_retreat()
        self._log("safe_stop", step, attempt, ok=r.success)
        if not r.success and r.reason == "episode_timeout":
            raise _EpisodeTimeout

    # ------------------------------------------------------------------ main loop
    def run(self, plan: TaskPlan, seed: int | None = None) -> EpisodeReport:
        wall0 = time.perf_counter()
        self.events = []
        self._occupied = {}
        outcomes = [StepOutcome(s.object_name, s.target_location) for s in plan.steps]
        aborted, abort_reason = False, None
        self._log(
            "plan", plan=[s.model_dump(mode="json") for s in plan.steps], planner=plan.planner
        )
        try:
            perception = self._perceive()
            for step, outcome in zip(plan.steps, outcomes, strict=True):
                perception = self._run_step(step, outcome, perception)
        except _EpisodeTimeout:
            aborted, abort_reason = True, "episode_timeout"
            self._log("abort", reason=abort_reason)
            try:
                self.motion.hooks = [
                    h for h in self.motion.hooks if not isinstance(h, StepBudgetMonitor)
                ]
                self.motion.release_and_retreat()
            except Exception as exc:  # pragma: no cover - best effort safe stop
                logger.warning("safe stop after timeout failed: %s", exc)
            for o in outcomes:
                if o.final_status == "pending":
                    o.final_status = "aborted"
        success = all(o.success for o in outcomes) and not aborted
        report = EpisodeReport(
            instruction=plan.instruction,
            mode=self.perception.mode,
            controller=getattr(self.skill, "name", type(self.skill).__name__),
            seed=seed,
            plan=[s.model_dump(mode="json") for s in plan.steps],
            outcomes=outcomes,
            success=success,
            aborted=aborted,
            abort_reason=abort_reason,
            control_steps=self.env.step_count,
            sim_time=self.env.sim_time,
            wall_time=time.perf_counter() - wall0,
            events=self.events,
            faults_injected=list(self.faults.injected) if self.faults else [],
        )
        self._log("episode_end", success=success, recoveries=report.recovery_attempts)
        return report

    def _run_step(
        self, step: TaskStep, outcome: StepOutcome, perception: PerceptionResult
    ) -> PerceptionResult:
        start = self.env.step_count
        attempt = 0
        while True:
            det = perception.get(step.object_name)
            if det is not None and np.linalg.norm(det.position - self.env.eef_pos) < 0.12:
                # The object moved with the (open) hand: it is wedged in the gripper.  Picking it
                # again is impossible and dangerous, so terminate this step safely.
                outcome.failures.append("object_stuck_in_gripper")
                outcome.final_status = "object_stuck_in_gripper"
                self._log(
                    "step_failed",
                    step,
                    attempt,
                    reason="object_stuck_in_gripper",
                    recoverable=False,
                )
                break
            task = self.planner.plan_step(step, perception, self._occupied)
            self._log("plan_step", step, attempt, status=task.status.value, reason=task.reason)
            if task.status == StepStatus.ALREADY_SATISFIED:
                outcome.success, outcome.final_status = True, "already_satisfied"
                break
            failure: str | None = None
            if not task.executable:
                failure = task.status.value
                if task.status not in _RETRYABLE_STATUSES:
                    # Re-trying cannot make an unreachable object or container reachable.
                    outcome.failures.append(failure)
                    outcome.final_status = failure
                    self._log("step_failed", step, attempt, reason=failure, recoverable=False)
                    break
            else:
                offset = self.faults.grasp_offset(attempt) if self.faults else np.zeros(3)
                if np.any(offset):
                    task.grasp.position = task.grasp.position + offset
                if self.faults:
                    self.faults.arm_attempt(attempt)
                outcome.attempts += 1
                outcome.manipulation_actions += 1
                self._log(
                    "execute",
                    step,
                    attempt,
                    grasp=task.grasp.position.round(3),
                    place=task.place.position.round(3),
                )
                result: SkillResult = self.skill.pick_and_place(task.grasp, task.place)
                if result.reason == "episode_timeout":
                    raise _EpisodeTimeout
                if result.info.get("stage") == "place" or result.success:
                    outcome.pick_successes += 1
                if not result.success:
                    failure = result.reason or f"failed_{result.phase}"
                    self._log("skill_failed", step, attempt, phase=result.phase, reason=failure)
                    self._safe_stop(step, attempt)
                perception = self._perceive()
                if failure is None:
                    ok, why = self._verify_placement(step, perception)
                    self._log("verify", step, attempt, ok=ok, reason=why)
                    if ok:
                        outcome.success, outcome.final_status = True, "placed"
                        det = perception.get(step.object_name)
                        self._occupied.setdefault(step.target_location, []).append(
                            det.position.copy()
                        )
                        break
                    failure = why
            outcome.failures.append(failure)
            if attempt >= self.cfg.max_recovery_attempts:
                outcome.final_status = f"gave_up:{failure}"
                self._log("step_failed", step, attempt, reason=failure, recoverable=False)
                break
            attempt += 1
            outcome.recovery_attempts += 1
            self._log("recovery", step, attempt, after=failure)
            if task.executable is False:
                perception = self._perceive()
        outcome.steps = self.env.step_count - start
        return perception


class _EpisodeTimeout(Exception):
    pass
