"""Optional LLM task parser (Claude via the official Anthropic SDK).

The LLM only maps language to the symbolic :class:`TaskPlan` schema.  It never produces
coordinates or joint angles.  Its output is validated by the same Pydantic models as the
rule-based parser, so a hallucinated object or container is rejected exactly like a user typo.

Requires ``uv sync --extra llm`` and Anthropic credentials (``ANTHROPIC_API_KEY`` or an
``ant auth login`` profile).  When anything is missing the caller falls back to the
rule-based parser (see :func:`robotics.language.parse_instruction`).
"""

from __future__ import annotations

import logging
import os

from pydantic import BaseModel, Field, ValidationError

from robotics.language.schema import PlanningError, TaskPlan, TaskStep
from robotics.simulation.scene import CONTAINERS, OBJECT_CATALOG

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("VLA_LLM_MODEL", "claude-opus-5-5")


class _LLMStep(BaseModel):
    object_name: str
    target_location: str
    execution_order: int
    action_type: str = Field(description="Always 'pick_and_place'")
    confidence: float


class _LLMOutput(BaseModel):
    steps: list[_LLMStep]
    unknown_objects: list[str] = Field(
        description="Objects the user mentioned that are not in the scene"
    )
    invalid_targets: list[str] = Field(
        description="Target locations the user mentioned that do not exist"
    )


def _system_prompt() -> str:
    objects = "\n".join(f"- {o.name} (aliases: {', '.join(o.aliases)})" for o in OBJECT_CATALOG)
    containers = "\n".join(
        f"- {name}: {', '.join(c.aliases)}"
        + (f" (designated container for {c.designated_for})" if c.designated_for else "")
        for name, c in CONTAINERS.items()
    )
    return (
        "You translate instructions for a robot arm that sorts objects between bins into a structured plan.\n"
        f"Objects in the scene:\n{objects}\n\nContainers:\n{containers}\n\n"
        "Rules: use only the canonical object and container names above. When the user says 'its bin', "
        "'matching bins' or gives no target, use the object's designated container. 'All objects' means every "
        "object above. Number steps from 1 in the order the user wants. If the user mentions an object or a "
        "target that does not exist, list it in unknown_objects / invalid_targets instead of guessing. "
        "Confidence is your certainty (0-1) that the step reflects the user's intent."
    )


class LLMTaskParser:
    name = "llm"

    def __init__(self, model: str = DEFAULT_MODEL, client=None):
        self.model = model
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # optional dependency
            raise PlanningError(
                "llm_unavailable", "Install the optional extra: uv sync --extra llm"
            ) from exc
        self._client = anthropic.Anthropic()
        return self._client

    def parse(self, instruction: str) -> TaskPlan:
        client = self._get_client()
        try:
            import anthropic

            api_errors: tuple[type[BaseException], ...] = (anthropic.APIError,)
        except ImportError:  # a fake client injected in tests
            api_errors = ()
        try:
            response = client.beta.messages.parse(
                model=self.model,
                max_tokens=4000,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=_system_prompt(),
                messages=[{"role": "user", "content": instruction}],
                output_format=_LLMOutput,
            )
        except api_errors as exc:
            raise PlanningError("llm_unavailable", f"LLM request failed: {exc}") from exc
        except TypeError as exc:  # credentials could not be resolved by the SDK
            raise PlanningError("llm_unavailable", f"LLM client not configured: {exc}") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            raise PlanningError("llm_refusal", "The LLM declined to plan this instruction.")
        out: _LLMOutput | None = getattr(response, "parsed_output", None)
        if out is None:
            raise PlanningError("llm_bad_output", "The LLM did not return a structured plan.")
        if out.unknown_objects:
            raise PlanningError("unknown_object", f"Unknown object(s): {out.unknown_objects}")
        if out.invalid_targets:
            raise PlanningError(
                "invalid_target", f"Unknown target location(s): {out.invalid_targets}"
            )
        try:
            steps = [TaskStep(**s.model_dump()) for s in out.steps]
            return TaskPlan(
                instruction=instruction,
                steps=steps,
                planner=f"llm:{self.model}",
                confidence=min((s.confidence for s in steps), default=0.0),
            )
        except ValidationError as exc:
            raise PlanningError(
                "llm_invalid_plan", f"LLM plan failed validation: {exc.errors()[0]['msg']}"
            ) from exc
