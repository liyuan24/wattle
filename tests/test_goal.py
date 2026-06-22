from __future__ import annotations

from wattle.goal import (
    GoalState,
    GoalTurnStopHook,
    build_goal_continuation_prompt,
    create_goal,
    edit_goal,
    set_goal_status,
)
from wattle.hooks import TurnStopContext
from wattle.providers import TextBlock
from wattle.tools.goal import UpdateGoalTool


def test_goal_continuation_prompt_includes_objective_without_budget_language() -> None:
    goal = GoalState(objective="Implement <feature> & verify it", status="active")

    prompt = build_goal_continuation_prompt(goal)

    assert "Continue working toward the active Wattle goal." in prompt
    assert "Implement &lt;feature&gt; &amp; verify it" in prompt
    assert "derive an independent oracle" in prompt
    assert "Do not treat file existence" in prompt
    assert "verify the representation" in prompt
    assert "exact types, names, ordering" in prompt
    assert "\"complete\" and include concise evidence" in prompt
    assert "include concise evidence explaining the authoritative" in prompt
    assert "call update_goal with status \"blocked\"" in prompt
    assert "Token budget" not in prompt


def test_goal_turn_stop_hook_continues_only_active_goal_without_pending_user_input() -> None:
    goal = create_goal("Finish the migration")
    hook = GoalTurnStopHook(lambda: goal)

    continuation = hook.on_turn_stop(
        TurnStopContext(messages=(), last_response=None, has_pending_user_input=False)
    )

    assert continuation is not None
    assert continuation.reason == "goal_continuation"
    assert len(continuation.content) == 1
    assert isinstance(continuation.content[0], TextBlock)
    assert "Finish the migration" in continuation.content[0].text
    assert hook.on_turn_stop(
        TurnStopContext(messages=(), last_response=None, has_pending_user_input=True)
    ) is None
    paused = set_goal_status(goal, "paused").goal
    assert GoalTurnStopHook(lambda: paused).on_turn_stop(
        TurnStopContext(messages=(), last_response=None, has_pending_user_input=False)
    ) is None


def test_edit_complete_goal_reactivates_goal() -> None:
    complete = set_goal_status(create_goal("old"), "complete").goal

    edited = edit_goal(complete, "new")

    assert edited.objective == "new"
    assert edited.status == "active"


def test_update_goal_tool_only_marks_existing_goal_complete_or_blocked() -> None:
    state = {"goal": create_goal("Ship it")}
    tool = UpdateGoalTool(
        get_goal=lambda: state["goal"],
        set_goal=lambda goal: state.__setitem__("goal", goal),
    )

    output = tool.run(status="complete", evidence="Focused tests passed and artifact exists.")

    assert state["goal"].status == "complete"
    assert "Goal complete." in output
    assert "Evidence: Focused tests passed and artifact exists." in output
    assert "Objective: Ship it" in output
    assert "Every call must include `evidence`" in tool.description
    assert "independent evidence from authoritative sources" in tool.description
    assert "schema shape" in tool.description
    assert "downstream representation contract" in tool.description
    assert "exact parsed types" in tool.description
    assert "either 'complete' or 'blocked'" in tool.run(status="paused")


def test_update_goal_tool_rejects_missing_evidence_without_closing_goal() -> None:
    state = {"goal": create_goal("Verify it")}
    tool = UpdateGoalTool(
        get_goal=lambda: state["goal"],
        set_goal=lambda goal: state.__setitem__("goal", goal),
    )

    output = tool.run(status="complete")

    assert state["goal"].status == "active"
    assert "requires non-empty evidence" in output
