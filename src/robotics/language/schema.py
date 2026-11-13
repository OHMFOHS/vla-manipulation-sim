"""Structured task representation produced by the language layer.

A :class:`TaskPlan` is the *only* thing that crosses the boundary between language
understanding and robotics.  It names objects and containers symbolically; it never contains
coordinates, joint angles or controller commands.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from robotics.simulation.scene import CONTAINERS, OBJECT_NAMES


class ActionType(StrEnum):
    PICK_AND_PLACE = "pick_and_place"


class PlanningError(ValueError):
    """Raised when an instruction cannot be turned into a valid plan.

    ``code`` is machine readable (``unknown_object``, ``invalid_target``, ``no_objects``,
    ``llm_unavailable`` ...) and ``message`` explains the problem to a human operator.
    """

    def __init__(self, code: str, message: str, suggestions: list[str] | None = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.suggestions = suggestions or []


class TaskStep(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    object_name: str = Field(description="Canonical object name, e.g. 'milk'")
    target_location: str = Field(description="Container name, e.g. 'milk_bin'")
    execution_order: int = Field(ge=1)
    action_type: ActionType = ActionType.PICK_AND_PLACE
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("object_name")
    @classmethod
    def _known_object(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in OBJECT_NAMES:
            raise ValueError(f"unknown object '{v}'; known objects are {list(OBJECT_NAMES)}")
        return v

    @field_validator("target_location")
    @classmethod
    def _known_target(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in CONTAINERS:
            raise ValueError(f"invalid target '{v}'; valid containers are {sorted(CONTAINERS)}")
        return v


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str
    steps: list[TaskStep] = Field(min_length=1)
    planner: str = "rule_based"
    confidence: float = Field(ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> TaskPlan:
        orders = [s.execution_order for s in self.steps]
        if len(set(orders)) != len(orders):
            raise ValueError(f"execution_order values must be unique, got {orders}")
        names = [s.object_name for s in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"each object may appear once per plan, got {names}")
        self.steps.sort(key=lambda s: s.execution_order)
        return self

    @property
    def ordered_objects(self) -> list[str]:
        return [s.object_name for s in self.steps]

    def summary(self) -> str:
        parts = [
            f"{s.execution_order}. {s.object_name} -> {s.target_location} ({s.confidence:.2f})"
            for s in self.steps
        ]
        return "; ".join(parts)
