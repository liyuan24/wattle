from __future__ import annotations

from typing import Any

from wattle.runtime import WattleRuntime

from .base import Tool


class SpawnAgentTool(Tool):
    name = "spawn_agent"
    description = (
        "Start a managed Wattle subagent in the current runtime. The subagent "
        "runs in-process with its own message history and can be waited on with "
        "wait_agent."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The delegated task for the subagent.",
            },
            "agent_type": {
                "type": "string",
                "description": (
                    "Optional type name for the new agent. If omitted, "
                    "`default` is used. Common values are `explorer` for "
                    "read-only investigation and `worker` for implementation."
                ),
            },
            "instructions": {
                "type": "string",
                "description": "Optional additional system instructions for this subagent.",
            },
            "context": {
                "type": "string",
                "description": "Optional explicit context to include with the task.",
            },
            "model": {
                "type": "string",
                "description": "Optional model override. Defaults to the parent model.",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Optional per-turn output token cap for the subagent.",
            },
            "tool_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional tool allowlist for the subagent. Defaults to normal "
                    "non-collaboration tools."
                ),
            },
        },
        "required": ["task"],
    }

    def __init__(self, runtime: WattleRuntime | None = None) -> None:
        self.runtime = runtime if runtime is not None else WattleRuntime()

    def run(
        self,
        task: str,
        agent_type: str | None = None,
        instructions: str | None = None,
        context: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        tool_names: list[str] | None = None,
    ) -> str:
        from wattle.subagents import subagent_summary

        record = self.runtime.subagents.spawn(
            task=task,
            agent_type=agent_type,
            instructions=instructions,
            context=context,
            model=model,
            max_tokens=max_tokens,
            tool_names=tool_names,
        )
        return subagent_summary(record)

    async def arun(self, **kwargs: Any) -> str:
        return self.run(**kwargs)


class SendInputTool(Tool):
    name = "send_input"
    description = (
        "Send a follow-up message to an idle managed subagent, continuing "
        "that subagent's existing message history."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "subagent_id": {
                "type": "string",
                "description": "The subagent id returned by spawn_agent.",
            },
            "message": {
                "type": "string",
                "description": "The follow-up input to send.",
            },
        },
        "required": ["subagent_id", "message"],
    }

    def __init__(self, runtime: WattleRuntime | None = None) -> None:
        self.runtime = runtime if runtime is not None else WattleRuntime()

    def run(self, subagent_id: str, message: str) -> str:
        from wattle.subagents import subagent_summary

        record = self.runtime.subagents.send_input(subagent_id, message)
        return subagent_summary(record)

    async def arun(self, subagent_id: str, message: str) -> str:
        return self.run(subagent_id, message)


class WaitAgentTool(Tool):
    name = "wait_agent"
    description = (
        "Wait for a managed Wattle subagent update, completion, or current "
        "status after a timeout. The subagent keeps running if the wait times out."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "subagent_id": {
                "type": "string",
                "description": "The subagent id returned by spawn_agent.",
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Maximum seconds to wait. Defaults to 30.",
            },
        },
        "required": ["subagent_id"],
    }

    def __init__(self, runtime: WattleRuntime | None = None) -> None:
        self.runtime = runtime if runtime is not None else WattleRuntime()

    def run(self, subagent_id: str, timeout_seconds: float = 30.0) -> str:
        from wattle.subagents import subagent_summary

        snapshot = self.runtime.subagents.wait(
            subagent_id,
            timeout_seconds=timeout_seconds,
        )
        return subagent_summary(snapshot)

    async def arun(self, subagent_id: str, timeout_seconds: float = 30.0) -> str:
        from wattle.subagents import subagent_summary

        snapshot = await self.runtime.subagents.await_wait(
            subagent_id,
            timeout_seconds=timeout_seconds,
        )
        return subagent_summary(snapshot)


class CloseAgentTool(Tool):
    name = "close_agent"
    description = (
        "Request closure of a managed Wattle subagent. Active provider calls are "
        "allowed to reach the next safe checkpoint before the session closes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "subagent_id": {
                "type": "string",
                "description": "The subagent id returned by spawn_agent.",
            },
        },
        "required": ["subagent_id"],
    }

    def __init__(self, runtime: WattleRuntime | None = None) -> None:
        self.runtime = runtime if runtime is not None else WattleRuntime()

    def run(self, subagent_id: str) -> str:
        from wattle.subagents import subagent_summary

        record = self.runtime.subagents.close(subagent_id)
        return subagent_summary(record)

    async def arun(self, subagent_id: str) -> str:
        from wattle.subagents import subagent_summary

        record = await self.runtime.subagents.await_close(subagent_id)
        return subagent_summary(record)
