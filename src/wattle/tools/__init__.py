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

DEFAULT_RUNTIME = WattleRuntime()


def build_tools(runtime: WattleRuntime | None = None) -> dict[str, Tool]:
    shared_runtime = runtime if runtime is not None else DEFAULT_RUNTIME
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


TOOLS_BY_NAME: dict[str, Tool] = build_tools()
ALL_TOOLS: list[Tool] = list(TOOLS_BY_NAME.values())

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
]
