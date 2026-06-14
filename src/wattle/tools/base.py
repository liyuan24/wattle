import asyncio
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, TypedDict

from wattle.tool_events import ToolRunEvent


class ToolSpec(TypedDict):
    """The shape of `Tool.spec()` — name, description, and JSON Schema input.

    `input_schema` is itself a JSON Schema document; its keys and value types
    are dictated by the JSON Schema spec, not by Wattle, so it stays a
    `dict[str, Any]`. The outer envelope is fixed and typed.
    """

    name: str
    description: str
    # Any: a JSON Schema document. Keys/values are defined by the JSON Schema
    # spec (mixing strings, numbers, lists, nested schemas), not by Wattle.
    input_schema: dict[str, Any]


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    # Any: same JSON-Schema rationale as ToolSpec.input_schema above.
    input_schema: ClassVar[dict[str, Any]]

    @abstractmethod
    # Any: kwargs are deserialized from the model's tool-call arguments
    # (`json.loads(...)`). The shape is dictated by `input_schema`, which the
    # tool author writes in JSON Schema; there is no static type for it.
    def run(self, **kwargs: Any) -> str: ...

    # Any: kwargs are deserialized from model tool-call arguments, same as run().
    async def arun(self, **kwargs: Any) -> str:
        """Async tool hook.

        Tools with async-native implementations should override this. Sync
        tools remain valid and are executed in a worker thread by default.
        """

        return await asyncio.to_thread(self.run, **kwargs)

    async def arun_with_events(
        self,
        *,
        emit: Callable[[ToolRunEvent], None],
        tool_use_id: str,
        cancel_event: threading.Event | None = None,
        **kwargs: Any,
    ) -> str:
        """Async tool hook with optional runtime UI events.

        The default implementation preserves existing tool behavior. Tools that
        can report runtime progress may override this method.
        """

        del emit, tool_use_id, cancel_event
        return await self.arun(**kwargs)

    @classmethod
    def spec(cls) -> ToolSpec:
        return {
            "name": cls.name,
            "description": cls.description,
            "input_schema": cls.input_schema,
        }
