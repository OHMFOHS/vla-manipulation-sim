"""Module A: natural-language task understanding."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from robotics.language import (
    PlanningError,
    RuleBasedTaskParser,
    TaskPlan,
    TaskStep,
    parse_instruction,
)


@pytest.fixture
def parser():
    return RuleBasedTaskParser()


def objs(plan: TaskPlan) -> list[str]:
    return plan.ordered_objects


def targets(plan: TaskPlan) -> dict[str, str]:
    return {s.object_name: s.target_location for s in plan.steps}


def test_sort_all_objects(parser):
    plan = parser.parse("Sort all objects into their matching bins.")
    assert objs(plan) == ["milk", "bread", "cereal", "can"]
    assert targets(plan) == {o: f"{o}_bin" for o in objs(plan)}
    assert plan.confidence == 1.0
    assert all(s.action_type.value == "pick_and_place" for s in plan.steps)


def test_order_first_then(parser):
    plan = parser.parse("Move the milk first, then the cereal.")
    assert objs(plan) == ["milk", "cereal"]
    assert [s.execution_order for s in plan.steps] == [1, 2]


def test_designated_container(parser):
    plan = parser.parse("Pick up the can and place it into its designated container.")
    assert targets(plan) == {"can": "can_bin"}
    assert plan.confidence == 1.0


def test_explicit_targets(parser):
    plan = parser.parse("Put the milk into the cereal bin and the bread into the can bin.")
    assert targets(plan) == {"milk": "cereal_bin", "bread": "can_bin"}


def test_spatial_container_alias(parser):
    plan = parser.parse("Put the bread in the far left bin, then the milk in its bin.")
    assert targets(plan) == {"bread": "can_bin", "milk": "milk_bin"}
    assert objs(plan) == ["bread", "milk"]


def test_after_modifier(parser):
    assert objs(parser.parse("Move the milk after the cereal.")) == ["cereal", "milk"]


def test_last_modifier_and_aliases(parser):
    plan = parser.parse("move the soda can and the loaf to their bins, cereal last")
    assert objs(plan) == ["can", "bread", "cereal"]


def test_exclusion(parser):
    plan = parser.parse("Sort everything except the can.")
    assert objs(plan) == ["milk", "bread", "cereal"]


def test_duplicates_are_removed(parser):
    plan = parser.parse("Move the milk, then the milk again")
    assert objs(plan) == ["milk"]
    assert any("duplicate" in n for n in plan.notes)


def test_typo_lowers_confidence(parser):
    plan = parser.parse("Grab the cerael")
    assert objs(plan) == ["cereal"]
    assert plan.confidence < 1.0


def test_missing_target_lowers_confidence(parser):
    plan = parser.parse("Move the bread")
    assert targets(plan) == {"bread": "bread_bin"}
    assert plan.confidence < 1.0


@pytest.mark.parametrize(
    "text,code",
    [
        ("Pick up the banana and put it in its bin.", "unknown_object"),
        ("Put the milk in the fridge.", "invalid_target"),
        ("Do a backflip", "no_objects"),
        ("", "empty_instruction"),
        ("Sort everything except milk, bread, cereal and can", "no_objects"),
    ],
)
def test_rejections_are_explained(parser, text, code):
    with pytest.raises(PlanningError) as exc:
        parser.parse(text)
    assert exc.value.code == code
    assert exc.value.message


def test_schema_rejects_unknown_object_and_target():
    with pytest.raises(ValidationError):
        TaskStep(
            object_name="banana", target_location="milk_bin", execution_order=1, confidence=1.0
        )
    with pytest.raises(ValidationError):
        TaskStep(object_name="milk", target_location="fridge", execution_order=1, confidence=1.0)
    with pytest.raises(ValidationError):
        TaskStep(object_name="milk", target_location="milk_bin", execution_order=1, confidence=1.5)


def test_schema_rejects_duplicate_orders():
    s1 = TaskStep(object_name="milk", target_location="milk_bin", execution_order=1, confidence=1.0)
    s2 = TaskStep(object_name="can", target_location="can_bin", execution_order=1, confidence=1.0)
    with pytest.raises(ValidationError):
        TaskPlan(instruction="x", steps=[s1, s2], confidence=1.0)


def test_unknown_planner_name():
    with pytest.raises(ValueError):
        parse_instruction("Sort all objects", planner="magic")
