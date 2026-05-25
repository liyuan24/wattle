"""A scripted async provider for tests."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator, Iterable

from .base import (
    CompletionRequest,
    CompletionResponse,
    Provider,
    StreamComplete,
    StreamEvent,
)


class StubProvider(Provider):
    """Replay a scripted sequence of `CompletionResponse`s."""

    def __init__(self, responses: Iterable[CompletionResponse]) -> None:
        self._responses: deque[CompletionResponse] = deque(responses)
        self.requests: list[CompletionRequest] = []

    async def acomplete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError(
                "StubProvider exhausted: loop made more calls than the test scripted."
            )
        return self._responses.popleft()

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError(
                "StubProvider exhausted: loop made more calls than the test scripted."
            )
        response = self._responses.popleft()
        yield StreamComplete(response=response)
