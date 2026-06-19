"""Model-facing tools for Wattle goals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from wattle.goal import GoalState, GoalStatus, goal_summary, set_goal_status
from wattle.tools.base import Tool


class UpdateGoalTool(Tool):
    name = "update_goal"
    description = "\n".join(
        [
            "Update the active Wattle goal.",
            "Use this tool only to mark the goal achieved or genuinely blocked.",
            "Set status to `complete` only when the objective has actually been ",
            "achieved and no required work remains.",
            "Set status to `blocked` only when the same blocking condition has ",
            "repeated for at least three consecutive goal turns and the agent cannot ",
            "make meaningful progress without user input or an external-state change.",
            "Do not use `blocked` merely because the work is hard, slow, uncertain, ",
            "incomplete, or would benefit from clarification.",
            "You cannot use this tool to pause, resume, or clear a goal; those status ",
            "changes are controlled by the user.",
        ]
    )
    input_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["complete", "blocked"],
                "description": (
                    "Set to complete only when the goal is achieved; set to blocked only "
                    "after the strict repeated-blocker audit is satisfied."
                ),
            }
        },
        "required": ["status"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        get_goal: Callable[[], GoalState | None],
        set_goal: Callable[[GoalState], None],
    ) -> None:
        self._get_goal = get_goal
        self._set_goal = set_goal

    def run(self, **kwargs: Any) -> str:
        status = cast(Literal["complete", "blocked"] | None, kwargs.get("status"))
        if status not in ("complete", "blocked"):
            return "update_goal status must be either 'complete' or 'blocked'."
        goal = self._get_goal()
        if goal is None:
            return "No goal is currently set."
        result = set_goal_status(goal, cast(GoalStatus, status))
        self._set_goal(result.goal)
        return f"Goal {result.goal.status}.\n{goal_summary(result.goal)}"
