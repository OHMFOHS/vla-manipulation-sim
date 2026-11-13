"""Optional LLM planner: validated with a fake client (no network, no API key)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from robotics.language import PlanningError, parse_instruction
from robotics.language.llm_parser import LLMTaskParser, _LLMOutput, _LLMStep


class FakeMessages:
    def __init__(self, output, stop_reason="end_turn"):
        self.output = output
        self.stop_reason = stop_reason
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(parsed_output=self.output, stop_reason=self.stop_reason)


def fake_client(output, stop_reason="end_turn"):
    msgs = FakeMessages(output, stop_reason)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def step(obj, tgt, order, conf=0.9):
    return _LLMStep(
        object_name=obj,
        target_location=tgt,
        execution_order=order,
        action_type="pick_and_place",
        confidence=conf,
    )


def test_llm_plan_is_validated():
    client, msgs = fake_client(
        _LLMOutput(
            steps=[step("milk", "milk_bin", 1), step("can", "can_bin", 2)],
            unknown_objects=[],
            invalid_targets=[],
        )
    )
    plan = LLMTaskParser(client=client).parse("milk then can")
    assert plan.ordered_objects == ["milk", "can"]
    assert plan.planner.startswith("llm:")
    assert msgs.kwargs["output_format"] is _LLMOutput


def test_llm_hallucinated_object_is_rejected():
    client, _ = fake_client(
        _LLMOutput(steps=[step("banana", "milk_bin", 1)], unknown_objects=[], invalid_targets=[])
    )
    with pytest.raises(PlanningError) as exc:
        LLMTaskParser(client=client).parse("banana")
    assert exc.value.code == "llm_invalid_plan"


def test_llm_reported_unknown_object():
    client, _ = fake_client(_LLMOutput(steps=[], unknown_objects=["banana"], invalid_targets=[]))
    with pytest.raises(PlanningError) as exc:
        LLMTaskParser(client=client).parse("banana")
    assert exc.value.code == "unknown_object"


def test_llm_refusal():
    client, _ = fake_client(None, stop_reason="refusal")
    with pytest.raises(PlanningError) as exc:
        LLMTaskParser(client=client).parse("x")
    assert exc.value.code == "llm_refusal"


def test_auto_falls_back_to_rule_based(monkeypatch):
    def boom(self, instruction):
        raise PlanningError("llm_unavailable", "no credentials")

    monkeypatch.setattr(LLMTaskParser, "parse", boom)
    plan = parse_instruction("Sort all objects into their matching bins.", planner="auto")
    assert plan.planner == "rule_based"
    assert len(plan.steps) == 4
