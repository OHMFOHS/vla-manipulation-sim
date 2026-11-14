"""Module C - high-level task planner.

Turns one symbolic :class:`TaskStep` plus the *current* perception result into a concrete,
ordered list of primitive actions with geometric targets:

    locate → estimate_pose → select_container → pick → place → verify

The planner is stateless and cheap, so the executor re-plans every step from fresh
perception.  That is how moved objects, failed attempts and partially completed plans are
handled: the world is re-observed and the plan is rebuilt instead of trusting stale state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from robotics.control.skills import GraspTarget, PlaceTarget
from robotics.language.schema import TaskPlan, TaskStep
from robotics.perception.types import PerceptionResult
from robotics.planning.grasp_selection import choose_grasp_yaw
from robotics.planning.placement import select_place_point
from robotics.simulation.scene import CONTAINERS, WORKSPACE, Workspace

logger = logging.getLogger(__name__)


class StepStatus(StrEnum):
    READY = "ready"
    ALREADY_SATISFIED = "already_satisfied"
    OBJECT_NOT_FOUND = "object_not_found"
    OBJECT_UNREACHABLE = "object_unreachable"
    TARGET_UNREACHABLE = "target_unreachable"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class PrimitiveAction:
    kind: str  # locate | estimate_pose | select_container | pick | place | verify
    object_name: str
    description: str
    params: dict = field(default_factory=dict)


@dataclass
class ManipulationTask:
    step: TaskStep
    status: StepStatus
    actions: list[PrimitiveAction] = field(default_factory=list)
    grasp: GraspTarget | None = None
    place: PlaceTarget | None = None
    reason: str | None = None

    @property
    def executable(self) -> bool:
        return self.status == StepStatus.READY


class TaskPlanner:
    def __init__(
        self,
        workspace: Workspace = WORKSPACE,
        min_confidence: float = 0.3,
        grasp_depth: float = 0.025,
    ):
        self.ws = workspace
        self.min_confidence = min_confidence
        self.grasp_depth = grasp_depth  # must match ControllerConfig.grasp_depth

    def plan(self, plan: TaskPlan, perception: PerceptionResult) -> list[ManipulationTask]:
        """Plan all steps against one perception snapshot (used for previews / dry runs)."""
        seen: set[str] = set()
        out: list[ManipulationTask] = []
        for step in plan.steps:
            if step.object_name in seen:  # defensive: TaskPlan already enforces uniqueness
                continue
            seen.add(step.object_name)
            out.append(self.plan_step(step, perception))
        return out

    def plan_step(
        self,
        step: TaskStep,
        perception: PerceptionResult,
        occupied: dict[str, list[np.ndarray]] | None = None,
    ) -> ManipulationTask:
        name = step.object_name
        container = CONTAINERS[step.target_location]
        actions = [
            PrimitiveAction(
                "locate", name, f"Locate {name} in the {perception.mode} perception result"
            )
        ]
        det = perception.get(name)
        if det is None:
            return ManipulationTask(
                step, StepStatus.OBJECT_NOT_FOUND, actions, reason=f"{name} not detected"
            )
        actions.append(
            PrimitiveAction(
                "estimate_pose",
                name,
                f"Estimate {name} position ({det.source})",
                {
                    "position": det.position.round(4).tolist(),
                    "top_z": round(det.top_z, 4),
                    "confidence": det.confidence,
                },
            )
        )
        if det.confidence < self.min_confidence:
            return ManipulationTask(
                step,
                StepStatus.LOW_CONFIDENCE,
                actions,
                reason=f"{name} detection confidence {det.confidence:.2f}",
            )
        # Task repetition: if the object already sits in its target, there is nothing to do.
        if container.contains(det.position, margin=0.0):
            actions.append(PrimitiveAction("verify", name, f"{name} already in {container.name}"))
            return ManipulationTask(
                step, StepStatus.ALREADY_SATISFIED, actions, reason="already in target"
            )

        if not self.ws.is_reachable(np.array([*det.position[:2], max(det.top_z, self.ws.low[2])])):
            return ManipulationTask(
                step,
                StepStatus.OBJECT_UNREACHABLE,
                actions,
                reason=f"{name} at {det.position.round(3).tolist()} is outside the reachable workspace",
            )

        occ = (occupied or {}).get(container.name, [])
        place_xyz = select_place_point(container, self.ws, occupied=occ)
        actions.append(
            PrimitiveAction(
                "select_container",
                name,
                f"Select {container.name} for {name}",
                {"place_point": None if place_xyz is None else place_xyz.round(4).tolist()},
            )
        )
        if place_xyz is None:
            return ManipulationTask(
                step,
                StepStatus.TARGET_UNREACHABLE,
                actions,
                reason=f"no reachable free spot in {container.name}",
            )
        others = [d.position for n, d in perception.detections.items() if n != name]
        elongated = "elongated_footprint" in det.flags or "toppled" in det.flags
        grasp_z = float(det.top_z) - self.grasp_depth
        yaw, clearance = choose_grasp_yaw(
            det.position[:2], float(det.yaw), elongated, others, grasp_z
        )
        grasp = GraspTarget(name, det.position.copy(), float(det.top_z), yaw)
        place = PlaceTarget(container.name, place_xyz)
        actions += [
            PrimitiveAction(
                "pick",
                name,
                f"Pick {name}",
                {
                    "grasp": det.position.round(4).tolist(),
                    "yaw": round(yaw, 3),
                    "finger_clearance": round(clearance, 3),
                },
            ),
            PrimitiveAction(
                "place",
                name,
                f"Place {name} into {container.name}",
                {"place": place_xyz.round(4).tolist()},
            ),
            PrimitiveAction("verify", name, f"Verify {name} is inside {container.name}"),
        ]
        return ManipulationTask(step, StepStatus.READY, actions, grasp=grasp, place=place)
