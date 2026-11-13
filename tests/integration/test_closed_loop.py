"""Modules A-E end to end on real physics, including recovery from injected faults."""

from __future__ import annotations

from robotics.agent import AgentOptions, ManipulationAgent
from robotics.common.config import default_simulation_config


def _agent(sim_env, perception="vision", **faults):
    cfg = default_simulation_config()
    if faults:
        cfg.faults.enabled = True
        for k, v in faults.items():
            setattr(cfg.faults, k, v)
    return ManipulationAgent(AgentOptions(perception=perception), sim_config=cfg, env=sim_env)


def test_language_to_placement_vision(sim_env):
    report = _agent(sim_env).run("Move the milk first, then the cereal.", seed=10_000)
    assert [o.object_name for o in report.outcomes] == ["milk", "cereal"]
    assert report.success
    assert report.ground_truth["per_object_in_target"] == {"milk": True, "cereal": True}
    events = [e["event"] for e in report.events]
    assert events.count("verify") >= 2 and "perceive" in events


def test_full_sort_oracle_matches_robosuite_success(sim_env):
    report = _agent(sim_env, "oracle").run(
        "Sort all objects into their matching bins.", seed=10_001
    )
    assert report.ground_truth["all_in_target"]
    assert report.ground_truth["robosuite_success"]  # official PickPlace success criterion


def test_recovery_after_injected_slip(sim_env):
    report = _agent(sim_env, slip_probability=1.0).run("Put the bread into its bin.", seed=10_002)
    outcome = report.outcomes[0]
    assert "gripper_slip" in report.faults_injected
    assert "object_dropped" in outcome.failures  # the drop was detected ...
    assert outcome.recovery_attempts >= 1  # ... a recovery was attempted ...
    assert report.ground_truth["per_object_in_target"]["bread"]  # ... and it succeeded
    assert any(e["event"] == "recovery" for e in report.events)


def test_recovery_disabled_fails_safely(sim_env):
    agent = _agent(sim_env, slip_probability=1.0)
    agent.cfg.recovery.max_recovery_attempts = 0
    report = agent.run("Put the bread into its bin.", seed=10_002)
    assert not report.success
    assert report.outcomes[0].final_status.startswith("gave_up")
    assert report.outcomes[0].recovery_attempts == 0
    assert any(e["event"] == "safe_stop" for e in report.events)


def test_episode_timeout_aborts_safely(sim_env):
    agent = _agent(sim_env)
    agent.cfg.recovery.episode_step_budget = 150
    report = agent.run("Sort all objects into their matching bins.", seed=10_003)
    assert report.aborted and report.abort_reason == "episode_timeout"
    assert not report.success


def test_task_repetition_is_idempotent(sim_env):
    agent = _agent(sim_env, "oracle")
    agent.run("Put the can into its bin.", seed=10_004)
    # Second instruction in the same world (no reset): the can is already there.
    from robotics.control.skills import ScriptedPickPlaceController
    from robotics.language import parse_instruction
    from robotics.perception.oracle import OraclePerception
    from robotics.recovery.executor import ClosedLoopExecutor

    motion = ScriptedPickPlaceController(sim_env)
    ex = ClosedLoopExecutor(sim_env, OraclePerception(sim_env), motion, motion)
    report = ex.run(parse_instruction("Put the can into its bin."))
    assert report.outcomes[0].final_status == "already_satisfied"
    assert report.outcomes[0].manipulation_actions == 0
