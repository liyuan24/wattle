from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

RUN_DEADLINE_EPOCH_MS_ENV = "WATTLE_RUN_DEADLINE_EPOCH_MS"


@dataclass(frozen=True, slots=True)
class RunDeadline:
    """Wall-clock deadline for a Wattle run, expressed as epoch milliseconds."""

    epoch_ms: int
    clock: Callable[[], float] = field(default=time.time, repr=False, compare=False)

    def remaining_seconds(self) -> float:
        return max(0.0, (self.epoch_ms / 1000.0) - self.clock())

    def system_notice(self) -> str:
        remaining = self.remaining_seconds()
        if remaining <= 0:
            return (
                "Runtime context:\n"
                "- The wall-clock deadline for this run has passed. Finish with the "
                "current state and avoid starting new long-running work."
            )
        guidance = (
            "Use this to choose a feasible plan and validation scope."
            if remaining > 300
            else (
                "Prioritize completing the required work, avoid extra exploration, "
                "and only start commands that fit in the remaining time."
            )
        )
        return (
            "Runtime context:\n"
            f"- Wall-clock budget remaining for this run: {_format_remaining(remaining)}.\n"
            f"- {guidance}"
        )


def run_deadline_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    clock: Callable[[], float] = time.time,
) -> RunDeadline | None:
    raw_value = (os.environ if environ is None else environ).get(RUN_DEADLINE_EPOCH_MS_ENV)
    if raw_value is None or raw_value.strip() == "":
        return None
    try:
        epoch_ms = int(raw_value)
    except ValueError:
        return None
    if epoch_ms <= 0:
        return None
    return RunDeadline(epoch_ms=epoch_ms, clock=clock)


def append_runtime_deadline_notice(system: str | None, deadline: RunDeadline | None) -> str | None:
    if deadline is None:
        return system
    notice = deadline.system_notice()
    if not system:
        return notice
    return f"{system}\n\n{notice}"


def _format_remaining(seconds: float) -> str:
    rounded = max(0, int(seconds + 0.5))
    if rounded < 90:
        return f"about {rounded} seconds"
    if rounded < 3600:
        minutes = max(1, int((rounded + 59) // 60))
        return f"about {minutes} minutes"
    hours = rounded // 3600
    minutes = (rounded % 3600) // 60
    if minutes == 0:
        return f"about {hours} hours"
    return f"about {hours} hours {minutes} minutes"
