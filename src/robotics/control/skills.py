"""Scripted pick-and-place skill built on robosuite's OSC_POSE controller.

The skill only consumes *targets* (where to grasp, where to place).  It does not know where the
targets came from (oracle or vision) and it does not decide what to do next - that is the job
of the task planner and the closed-loop executor.  Every motion is produced by the physics
engine through controller actions; nothing is teleported.

Pick sequence: pre-grasp → approach → descend → close → verify grasp → lift → verify hold.
Place sequence: transport at safe height → descend → release → retreat → (home).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from robotics.common.config import ControllerConfig
from robotics.control.osc import GRIPPER_CLOSE, GRIPPER_OPEN, OSCCommander, top_down_orientation
from robotics.control.trajectory import min_jerk_segment
from robotics.simulation.scene import BIN_FLOOR_Z, WORKSPACE, Workspace

logger = logging.getLogger(__name__)


@dataclass
class GraspTarget:
    object_name: str
    position: np.ndarray  # estimated object centre (world)
    top_z: float  # estimated top surface height
    yaw: float = 0.0  # gripper yaw to align fingers with the object faces


@dataclass
class PlaceTarget:
    container_name: str
    position: np.ndarray  # container centre at floor height (world)


@dataclass
class SkillResult:
    success: bool
    phase: str
    reason: str | None = None
    steps: int = 0
    info: dict = field(default_factory=dict)


class ControlHook:
    """Extension point called around every environment step.

    ``before_step`` may modify the executed action (fault injection, exploration noise) and
    ``after_step`` may abort the running skill by returning a failure reason (closed-loop
    monitors).  ``clean_action`` is the controller's intended action, used as the BC label.
    """

    def before_step(
        self, env, phase: str, clean_action: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        return action

    def after_step(self, env, phase: str) -> str | None:
        return None


class ScriptedPickPlaceController:
    """Expert controller: min-jerk set-points tracked by a bounded P-controller on OSC_POSE."""

    name = "scripted"

    def __init__(
        self,
        env,
        config: ControllerConfig | None = None,
        workspace: Workspace = WORKSPACE,
        hooks: list[ControlHook] | None = None,
    ):
        self.env = env
        self.cfg = config or ControllerConfig()
        self.ws = workspace
        self.hooks: list[ControlHook] = list(hooks or [])
        self.osc = OSCCommander(self.cfg, env.action_dim)
        self.target_rot = top_down_orientation(0.0)
        self.gripper_cmd = GRIPPER_OPEN
        self.steps = 0
        self.held_offset_z: float | None = None  # eef_z - object_bottom_z while holding

    # ------------------------------------------------------------------ low level
    def _step(self, action: np.ndarray, phase: str) -> str | None:
        executed = action.copy()
        for h in self.hooks:
            executed = h.before_step(self.env, phase, action, executed)
        self.env.step(executed)
        self.steps += 1
        for h in self.hooks:
            reason = h.after_step(self.env, phase)
            if reason:
                return reason
        return None

    def _action_towards(self, setpoint: np.ndarray) -> np.ndarray:
        return self.osc.action(
            self.env.eef_pos, self.env.eef_quat, setpoint, self.target_rot, self.gripper_cmd
        )

    def move_to(
        self,
        target: np.ndarray,
        phase: str,
        tolerance: float | None = None,
        timeout: int | None = None,
        max_speed: float | None = None,
        allow_stall: bool = False,
        clip: bool = True,
    ) -> SkillResult:
        """Follow a min-jerk segment to ``target`` then converge within ``tolerance``.

        ``allow_stall`` accepts a final error of up to 2.5×tolerance when the arm has stopped
        moving (e.g. fingertips touching the object or the arm at the edge of its workspace),
        instead of pushing until the timeout.
        """
        target = np.asarray(target, dtype=np.float64)
        if clip:
            target = self.ws.clip(target)
        tol = tolerance if tolerance is not None else self.cfg.position_tolerance
        budget = timeout if timeout is not None else self.cfg.segment_timeout_steps
        start_steps = self.steps
        setpoints = min_jerk_segment(
            self.env.eef_pos, target, max_speed or self.cfg.max_speed, self.env.control_dt
        )
        for sp in setpoints:
            reason = self._step(self._action_towards(sp), phase)
            if reason:
                return SkillResult(False, phase, reason, self.steps - start_steps)
            if self.steps - start_steps >= budget:
                break
        history: list[np.ndarray] = []
        while self.steps - start_steps < budget:
            err = float(np.linalg.norm(target - self.env.eef_pos))
            if err < tol:
                return SkillResult(True, phase, None, self.steps - start_steps, {"error": err})
            history.append(self.env.eef_pos.copy())
            if allow_stall and len(history) > 8 and err < 2.5 * tol:
                moved = float(np.linalg.norm(history[-1] - history[-8]))
                if moved < 0.002:
                    return SkillResult(
                        True, phase, None, self.steps - start_steps, {"error": err, "stalled": True}
                    )
            reason = self._step(self._action_towards(target), phase)
            if reason:
                return SkillResult(False, phase, reason, self.steps - start_steps)
        err = float(np.linalg.norm(target - self.env.eef_pos))
        if err < tol:
            return SkillResult(True, phase, None, self.steps - start_steps, {"error": err})
        return SkillResult(
            False, phase, f"timeout:{phase}", self.steps - start_steps, {"error": err}
        )

    def actuate_gripper(self, command: float, n_steps: int, phase: str) -> SkillResult:
        self.gripper_cmd = command
        hold = self.env.eef_pos.copy()
        start = self.steps
        for _ in range(n_steps):
            reason = self._step(self._action_towards(hold), phase)
            if reason:
                return SkillResult(False, phase, reason, self.steps - start)
        return SkillResult(True, phase, None, self.steps - start, {"width": self.env.gripper_width})

    def is_holding(self) -> bool:
        """Proprioceptive grasp check: closed fingers that stop on an object keep a gap."""
        w = self.env.gripper_width
        return (
            self.gripper_cmd == GRIPPER_CLOSE
            and self.cfg.grasp_min_width < w < self.cfg.grasp_max_width
        )

    # ------------------------------------------------------------------ skills
    def pick(self, target: GraspTarget) -> SkillResult:
        c = self.cfg
        total = 0
        self.target_rot = top_down_orientation(target.yaw)
        xy = np.asarray(target.position[:2], dtype=np.float64)
        top = float(target.top_z)
        grasp_z = max(top - c.grasp_depth, BIN_FLOOR_Z + 0.012)
        pre_z = max(top + c.pregrasp_height, self.env.eef_pos[2])

        if self.gripper_cmd != GRIPPER_OPEN or self.env.gripper_width < 0.07:
            r = self.actuate_gripper(GRIPPER_OPEN, c.gripper_open_steps, "open")
            total += r.steps
            if not r.success:
                return SkillResult(False, "open", r.reason, total)

        plan = [
            ("pre_grasp", np.array([*xy, pre_z]), None),
            ("approach", np.array([*xy, top + 0.03]), None),
            ("descend", np.array([*xy, grasp_z]), 0.015),
        ]
        for phase, wp, tol in plan:
            r = self.move_to(
                wp,
                phase,
                tolerance=tol,
                max_speed=c.max_speed if phase == "pre_grasp" else 0.2,
                allow_stall=phase != "pre_grasp",
            )
            total += r.steps
            if not r.success:
                return SkillResult(False, phase, r.reason, total, r.info)

        r = self.actuate_gripper(GRIPPER_CLOSE, c.gripper_close_steps, "close")
        total += r.steps
        if not r.success:
            return SkillResult(False, "close", r.reason, total)
        if not self.is_holding():
            return SkillResult(
                False, "close", "grasp_empty", total, {"width": self.env.gripper_width}
            )
        # Offset between the end-effector and the object bottom, used to compute place height.
        object_bottom = top - 2.0 * (top - float(target.position[2]))
        self.held_offset_z = float(self.env.eef_pos[2] - object_bottom)

        r = self.move_to(
            np.array([*self.env.eef_pos[:2], c.lift_height]),
            "lift",
            max_speed=0.3,
            allow_stall=True,
        )
        total += r.steps
        if not r.success:
            return SkillResult(False, "lift", r.reason, total, r.info)
        for _ in range(c.settle_steps):
            reason = self._step(self._action_towards(self.env.eef_pos), "lift")
            total += 1
            if reason:
                return SkillResult(False, "lift", reason, total)
        if not self.is_holding():
            return SkillResult(
                False, "lift", "dropped_during_lift", total, {"width": self.env.gripper_width}
            )
        return SkillResult(True, "lift", None, total, {"width": self.env.gripper_width})

    def place(self, target: PlaceTarget) -> SkillResult:
        c = self.cfg
        total = 0
        xy = np.asarray(target.position[:2], dtype=np.float64)
        offset = self.held_offset_z if self.held_offset_z is not None else 0.05
        release_z = float(target.position[2]) + c.place_release_height + offset
        transport_z = max(self.ws.safe_height, self.env.eef_pos[2])
        plan = [
            ("transport", np.array([*xy, transport_z]), 0.02, c.max_speed),
            ("lower", np.array([*xy, release_z]), 0.015, 0.2),
        ]
        for phase, wp, tol, speed in plan:
            r = self.move_to(wp, phase, tolerance=tol, max_speed=speed, allow_stall=True)
            total += r.steps
            if not r.success:
                return SkillResult(False, phase, r.reason, total, r.info)
        r = self.actuate_gripper(GRIPPER_OPEN, c.gripper_open_steps, "release")
        total += r.steps
        if not r.success:
            return SkillResult(False, "release", r.reason, total)
        self.held_offset_z = None
        r = self.move_to(
            np.array([*xy, transport_z]), "retreat", tolerance=0.02, max_speed=0.3, allow_stall=True
        )
        total += r.steps
        if not r.success:
            return SkillResult(False, "retreat", r.reason, total, r.info)
        return SkillResult(True, "retreat", None, total)

    def pick_and_place(self, grasp: GraspTarget, place: PlaceTarget) -> SkillResult:
        """Full manipulation skill used by the executor (same interface as the learned policy)."""
        r = self.pick(grasp)
        if not r.success:
            return SkillResult(False, r.phase, r.reason, r.steps, {**r.info, "stage": "pick"})
        r2 = self.place(place)
        return SkillResult(
            r2.success, r2.phase, r2.reason, r.steps + r2.steps, {**r2.info, "stage": "place"}
        )

    def go_to_pose(self, xyz, phase: str = "home") -> SkillResult:
        """Move to a named pose, rising to the safe height first to avoid sweeping through bins."""
        total = 0
        xyz = np.asarray(xyz, dtype=np.float64)
        if self.env.eef_pos[2] < self.ws.safe_height - 0.02:
            r = self.move_to(
                np.array([*self.env.eef_pos[:2], self.ws.safe_height]),
                f"{phase}_rise",
                tolerance=0.02,
                allow_stall=True,
            )
            total += r.steps
            if not r.success:
                return SkillResult(False, phase, r.reason, total)
        self.target_rot = top_down_orientation(0.0)
        # Named poses (home / perception) are trusted and may lie outside the manipulation box.
        r = self.move_to(xyz, phase, tolerance=0.02, allow_stall=True, clip=False)
        return SkillResult(r.success, phase, r.reason, total + r.steps)

    def release_and_retreat(self) -> SkillResult:
        """Safe-stop routine: open the gripper and back off vertically."""
        r = self.actuate_gripper(GRIPPER_OPEN, self.cfg.gripper_open_steps, "safe_release")
        self.held_offset_z = None
        r2 = self.move_to(
            np.array([*self.env.eef_pos[:2], self.ws.safe_height]),
            "safe_retreat",
            tolerance=0.03,
            allow_stall=True,
        )
        return SkillResult(
            r.success and r2.success, "safe_stop", r.reason or r2.reason, r.steps + r2.steps
        )
