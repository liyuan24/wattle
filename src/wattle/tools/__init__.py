from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from functools import cache

from wattle.runtime import WattleRuntime

from .base import Tool, ToolSpec
from .bash import BashTool
from .edit import EditTool
from .image import ViewImageTool
from .monitor import MonitorTool
from .plan import UpdatePlanTool
from .read import ReadTool
from .subagent import CloseAgentTool, SendInputTool, SpawnAgentTool, WaitAgentTool
from .write import WriteTool


@cache
def default_runtime() -> WattleRuntime:
    return WattleRuntime()


class _DefaultRuntimeProxy:
    def __getattr__(self, name: str):
        return getattr(default_runtime(), name)

    def cleanup(self) -> None:
        default_runtime().cleanup()


DEFAULT_RUNTIME = _DefaultRuntimeProxy()


def build_tools(runtime: WattleRuntime | None = None) -> dict[str, Tool]:
    shared_runtime = runtime if runtime is not None else default_runtime()
    tools: list[Tool] = [
        BashTool(runtime=shared_runtime),
        MonitorTool(runtime=shared_runtime),
        SpawnAgentTool(runtime=shared_runtime),
        SendInputTool(runtime=shared_runtime),
        WaitAgentTool(runtime=shared_runtime),
        CloseAgentTool(runtime=shared_runtime),
        ViewImageTool(),
        ReadTool(),
        WriteTool(),
        EditTool(),
        UpdatePlanTool(),
    ]
    return {tool.name: tool for tool in tools}


@cache
def default_tools_by_name() -> dict[str, Tool]:
    return build_tools(default_runtime())


class _ToolsByNameProxy(MutableMapping[str, Tool]):
    def _tools(self) -> dict[str, Tool]:
        return default_tools_by_name()

    def __getitem__(self, key: str) -> Tool:
        return self._tools()[key]

    def __setitem__(self, key: str, value: Tool) -> None:
        self._tools()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._tools()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools())

    def __len__(self) -> int:
        return len(self._tools())

    def __contains__(self, key: object) -> bool:
        return key in self._tools()

    def __repr__(self) -> str:
        return repr(self._tools())


TOOLS_BY_NAME = _ToolsByNameProxy()


class _AllToolsProxy:
    def _tools(self) -> tuple[Tool, ...]:
        return tuple(TOOLS_BY_NAME.values())

    def __iter__(self):
        return iter(self._tools())

    def __len__(self) -> int:
        return len(self._tools())

    def __getitem__(self, index: int) -> Tool:
        return self._tools()[index]

    def __repr__(self) -> str:
        return repr(self._tools())


ALL_TOOLS = _AllToolsProxy()

__all__ = [
    "ALL_TOOLS",
    "TOOLS_BY_NAME",
    "BashTool",
    "CloseAgentTool",
    "DEFAULT_RUNTIME",
    "EditTool",
    "MonitorTool",
    "ReadTool",
    "SendInputTool",
    "SpawnAgentTool",
    "Tool",
    "ToolSpec",
    "UpdatePlanTool",
    "ViewImageTool",
    "WattleRuntime",
    "WaitAgentTool",
    "WriteTool",
    "build_tools",
    "default_runtime",
    "default_tools_by_name",
]
