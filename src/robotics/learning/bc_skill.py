"""Deploying the learned policy as a drop-in replacement for the scripted skill."""

from __future__ import annotations

import numpy as np

from robotics.common.config import ControllerConfig
from robotics.control.osc import OSCCommander, top_down_orientation
from robotics.control.skills import ControlHook, GraspTarget, PlaceTarget, SkillResult
from robotics.learning.features import HeldLatch, build_features
from robotics.learning.policy import LoadedPolicy


class BCPickPlaceSkill:
    """Runs the BC policy in closed loop for one pick-and-place.

    The policy outputs the translation action and the gripper command.  Orientation is held
    top-down at the grasp yaw by the same stabiliser the expert uses (the policy does not need
    to learn it).  The skill keeps the executor's monitors (drop detection, step budget) and
    reports where it failed, exactly like the scripted skill.
    """

    name = "bc"

    def __init__(
        self, env, policy: LoadedPolicy, config: ControllerConfig | None = None, timeout: int = 450
    ):
        self.env = env
        self.policy = policy
        self.cfg = config or ControllerConfig()
        self.osc = OSCCommander(self.cfg, env.action_dim)
        self.hooks: list[ControlHook] = []
        self.timeout = timeout

    def _step(self, action: np.ndarray, phase: str) -> str | None:
        executed = action.copy()
        for h in self.hooks:
            executed = h.before_step(self.env, phase, action, executed)
        self.env.step(executed)
        for h in self.hooks:
            reason = h.after_step(self.env, phase)
            if reason:
                return reason
        return None

    def pick_and_place(self, grasp: GraspTarget, place: PlaceTarget) -> SkillResult:
        env = self.env
        target_rot = top_down_orientation(grasp.yaw)
        latch = HeldLatch(self.cfg.grasp_min_width, self.cfg.grasp_max_width)
        prev_cmd = -1.0
        held = False
        released = False
        steps = 0
        phase = "bc_reach"
        while steps < self.timeout:
            latch_v = latch.update(env.gripper_width, prev_cmd)
            f = build_features(
                env.eef_pos, env.gripper_width, grasp.position, grasp.top_z, place.position, prev_cmd, latch_v,
                grasp.object_name,
            )  # fmt: skip
            xyz, p_close = self.policy.act(f)
            cmd = 1.0 if p_close > 0.5 else -1.0
            a = np.zeros(env.action_dim)
            a[:3] = np.clip(xyz, -1.0, 1.0)
            a[3:6] = self.osc.rotation_action(env.eef_quat, target_rot)
            a[-1] = cmd
            # Phase bookkeeping (for monitors / reporting only - it does not drive the policy).
            if latch_v > 0 and not held:
                held = True
            near_goal = np.linalg.norm(env.eef_pos[:2] - place.position[:2]) < 0.06
            if held and cmd < 0 and not released:
                released = True
                if not near_goal:
                    return SkillResult(
                        False,
                        "bc_transport",
                        "released_away_from_target",
                        steps,
                        {"stage": "place"},
                    )
            if released:
                phase = "bc_release"
            elif held and env.eef_pos[2] > grasp.top_z + 0.05:
                phase = "bc_transport"
            reason = self._step(a, phase)
            steps += 1
            prev_cmd = cmd
            if reason:
                stage = "place" if held else "pick"
                return SkillResult(False, phase, reason, steps, {"stage": stage})
            if (
                released
                and env.eef_pos[2] > place.position[2] + self.cfg.place_release_height + 0.12
            ):
                return SkillResult(True, "bc_release", None, steps, {"stage": "place"})
        stage = "place" if held else "pick"
        if not held:
            return SkillResult(False, phase, "bc_timeout_before_grasp", steps, {"stage": stage})
        return SkillResult(False, phase, "bc_timeout", steps, {"stage": stage})
