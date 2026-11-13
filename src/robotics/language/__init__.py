"""Module A - natural-language task understanding."""

from __future__ import annotations

import logging

from robotics.language.llm_parser import LLMTaskParser
from robotics.language.rule_parser import RuleBasedTaskParser
from robotics.language.schema import ActionType, PlanningError, TaskPlan, TaskStep

logger = logging.getLogger(__name__)

__all__ = [
    "ActionType",
    "LLMTaskParser",
    "PlanningError",
    "RuleBasedTaskParser",
    "TaskPlan",
    "TaskStep",
    "parse_instruction",
]


def parse_instruction(instruction: str, planner: str = "rule") -> TaskPlan:
    """Parse ``instruction`` with the requested planner.

    ``planner``: ``rule`` (default, offline), ``llm`` (Claude, raises if unavailable) or
    ``auto`` (LLM when configured, otherwise / on service failure the rule-based parser).
    Semantic errors (unknown object, invalid target) are never silently retried.
    """
    if planner == "rule":
        return RuleBasedTaskParser().parse(instruction)
    if planner == "llm":
        return LLMTaskParser().parse(instruction)
    if planner == "auto":
        try:
            return LLMTaskParser().parse(instruction)
        except PlanningError as exc:
            if exc.code.startswith("llm_"):
                logger.warning("LLM planner unavailable (%s); using rule-based parser", exc.message)
                return RuleBasedTaskParser().parse(instruction)
            raise
    raise ValueError(f"unknown planner '{planner}' (use rule | llm | auto)")
