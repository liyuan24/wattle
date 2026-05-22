from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from .base import Tool

PlanStepStatus = Literal["pending", "in_progress", "completed"]

VALID_PLAN_STATUSES: frozenset[str] = frozenset(
    {"pending", "in_progress", "completed"}
)


@dataclass(frozen=True, slots=True)
class PlanStep:
    step: str
    status: PlanStepStatus


@dataclass(frozen=True, slots=True)
class PlanUpdate:
    explanation: str | None
    plan: list[PlanStep]


class PlanValidationError(ValueError):
    """Raised when update_plan input is malformed."""


def parse_plan_update_input(raw: Mapping[str, object]) -> PlanUpdate:
    if "plan" not in raw:
        raise PlanValidationError("plan is required")

    explanation_value = raw.get("explanation")
    if explanation_value is None:
        explanation = None
    elif isinstance(explanation_value, str):
        explanation = explanation_value.strip() or None
    else:
        raise PlanValidationError("explanation must be a string")

    plan_value = raw["plan"]
    if not isinstance(plan_value, list):
        raise PlanValidationError("plan must be a list")

    steps: list[PlanStep] = []
    in_progress_count = 0
    for index, item in enumerate(plan_value):
        if not isinstance(item, Mapping):
            raise PlanValidationError(f"plan[{index}] must be an object")

        step_value = item.get("step")
        if not isinstance(step_value, str):
            raise PlanValidationError(f"plan[{index}].step must be a string")
        step = step_value.strip()
        if not step:
            raise PlanValidationError(f"plan[{index}].step must not be empty")

        status_value = item.get("status")
        if not isinstance(status_value, str) or status_value not in VALID_PLAN_STATUSES:
            allowed = ", ".join(sorted(VALID_PLAN_STATUSES))
            raise PlanValidationError(
                f"plan[{index}].status must be one of: {allowed}"
            )
        if status_value == "in_progress":
            in_progress_count += 1

        steps.append(PlanStep(step=step, status=status_value))

    if in_progress_count > 1:
        raise PlanValidationError("at most one plan item can be in_progress")

    return PlanUpdate(explanation=explanation, plan=steps)


class UpdatePlanTool(Tool):
    name = "update_plan"
    description = (
        "Updates the task plan. Provide an optional explanation and a list of "
        "plan items, each with a step and status. At most one step can be "
        "in_progress at a time."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "explanation": {"type": "string"},
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                    },
                    "required": ["step", "status"],
                },
            },
        },
        "required": ["plan"],
    }

    def run(self, **kwargs: Any) -> str:
        unknown = sorted(set(kwargs) - {"explanation", "plan"})
        if unknown:
            fields = ", ".join(unknown)
            raise PlanValidationError(f"unknown update_plan field(s): {fields}")
        parse_plan_update_input(kwargs)
        return "Plan updated"
