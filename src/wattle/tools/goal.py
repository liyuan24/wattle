"""Model-facing tools for Wattle goals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from wattle.goal import GoalState, GoalStatus, goal_summary, set_goal_status
from wattle.tools.base import Tool

_INCOMPLETE_VALIDATION_PHRASES = (
    "could not run",
    "couldn't run",
    "unable to run",
    "could not execute",
    "couldn't execute",
    "unable to execute",
    "not possible",
    "not installed",
    "not available",
    "missing dependency",
    "runtime execution was not possible",
    "full runtime execution was not possible",
)

_SURFACE_ONLY_EVIDENCE_PHRASES = (
    "schema/type",
    "schema and type",
    "schema assertions",
    "type assertions",
    "syntax validation",
    "syntax check",
    "ast parsing",
    "py_compile",
    "parses as json",
    "parses as valid json",
    "parses as toml",
    "parsed as toml",
    "parsed with json.loads",
    "file exists",
    "script exists",
    "script runs",
    "runs on",
    "ran on",
    "writes ",
    "wrote ",
    "required fields",
    "integer fields",
    "argparse",
    "stub",
    "stubbed",
    "inspected after writing",
)

_SEMANTIC_EVIDENCE_MARKERS = (
    "compared",
    "matches",
    "matched",
    "equals",
    "reconstructs",
    "consumer",
    "source data",
    "oracle",
    "ground truth",
    "expected",
    "test passed",
    "tests passed",
    "oligotm",
    "cosine",
)

_AUTHORITATIVE_EVIDENCE_MARKERS = (
    "authoritative",
    "independent",
    "oracle",
    "source data",
    "raw data",
    "provided fixture",
    "fixture",
    "consumer",
    "downstream",
    "verifier",
    "public test",
    "integration test",
    "end-to-end",
    "e2e",
    "expected",
    "ground truth",
    "task contract",
    "specification",
    "spec",
    "cross-check",
    "separate",
    "recomputed",
)

_GENERIC_TEST_EVIDENCE_PHRASES = (
    "focused tests passed",
    "test passed",
    "tests passed",
)

_TEST_SCOPE_MARKERS = (
    "pytest",
    "test_",
    "unit test",
    "integration test",
    "end-to-end",
    "e2e",
    "verifier",
    "public test",
    "provided test",
    "fixture",
    "consumer",
    "source data",
    "task contract",
    "requirements",
)

_SELF_REFERENTIAL_EVIDENCE_PHRASES = (
    "artifact exists",
    "values already written",
    "what was already written",
    "current output",
    "current result",
    "stored output",
    "stored values",
    "output file",
    "result file",
    "matches /app/",
    "matched /app/",
    "matches parsed /app/",
    "matched parsed /app/",
    "matches the parsed /app/",
    "matched the parsed /app/",
    "matches result.txt",
    "matched result.txt",
    "matches results.json",
    "matched results.json",
    "matches recovered.json",
    "matched recovered.json",
    "exactly that line",
)


class UpdateGoalTool(Tool):
    name = "update_goal"
    description = "\n".join(
        [
            "Update the active Wattle goal.",
            "Use this tool only to mark the goal achieved or genuinely blocked.",
            (
                "Every call must include `evidence`: a concise current-state "
                "summary explaining why the complete or blocked status is justified."
            ),
            "Set status to `complete` only when the objective has actually been ",
            "achieved and no required work remains.",
            (
                "For data, artifact, or domain-answer goals, complete requires "
                "independent evidence from authoritative sources; do not rely only "
                "on file existence, schema shape, finite values, or values matching "
                "what was already written."
            ),
            (
                "When the deliverable is serialized or generated, complete also "
                "requires evidence that the downstream representation contract is "
                "satisfied, including exact parsed types and literal details that "
                "affect consumers."
            ),
            (
                "Do not mark complete when the evidence admits required validation "
                "could not run, dependencies were unavailable, or only surface checks "
                "such as syntax, schema, type, finite-value, or file-existence checks "
                "were performed."
            ),
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
            },
            "evidence": {
                "type": "string",
                "description": (
                    "Concise current-state evidence for the status change. For complete, "
                    "summarize the authoritative verification that proves every required "
                    "piece is done. For blocked, summarize the repeated blocking condition "
                    "and why no meaningful progress is possible."
                ),
            },
        },
        "required": ["status", "evidence"],
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
        evidence = kwargs.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            return (
                "update_goal requires non-empty evidence explaining why the goal is "
                f"{status}. Keep the goal active and gather stronger evidence if needed."
            )
        if status == "complete":
            problem = _completion_evidence_problem(evidence)
            if problem is not None:
                return (
                    "update_goal complete needs stronger current-state evidence: "
                    f"{problem} Keep the goal active and validate the result with the "
                    "actual consumer, source data, domain oracle, or closest faithful "
                    "runtime check before marking it complete."
                )
        goal = self._get_goal()
        if goal is None:
            return "No goal is currently set."
        result = set_goal_status(goal, cast(GoalStatus, status))
        self._set_goal(result.goal)
        return (
            f"Goal {result.goal.status}.\n"
            f"Evidence: {evidence.strip()}\n"
            f"{goal_summary(result.goal)}"
        )


def _completion_evidence_problem(evidence: str) -> str | None:
    normalized = " ".join(evidence.casefold().split())
    if any(phrase in normalized for phrase in _INCOMPLETE_VALIDATION_PHRASES):
        return (
            "the evidence says required validation could not run or a dependency "
            "was unavailable."
        )
    if any(phrase in normalized for phrase in _GENERIC_TEST_EVIDENCE_PHRASES) and not any(
        marker in normalized for marker in _TEST_SCOPE_MARKERS
    ):
        return (
            "the evidence cites tests too generically; name the exercised test, "
            "fixture, consumer, source data, or contract that makes the tests "
            "authoritative."
        )
    if any(phrase in normalized for phrase in _SURFACE_ONLY_EVIDENCE_PHRASES) and not any(
        marker in normalized for marker in _SEMANTIC_EVIDENCE_MARKERS
    ):
        return (
            "the evidence cites only surface validation such as syntax, schema, "
            "type, parse, or file-existence checks."
        )
    if any(phrase in normalized for phrase in _SELF_REFERENTIAL_EVIDENCE_PHRASES) and not any(
        marker in normalized for marker in _AUTHORITATIVE_EVIDENCE_MARKERS
    ):
        return (
            "the evidence is self-referential, such as checking that the produced "
            "artifact exists or matches values already written, without tying it "
            "to an authoritative source, consumer, fixture, verifier, or oracle."
        )
    return None
