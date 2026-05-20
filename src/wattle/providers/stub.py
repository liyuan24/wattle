"""A scripted provider for tests.

Constructed with a list of `CompletionResponse`s; each call to `complete()`
pops the next one. Anything beyond the script raises — tests should script
exactly the turns they expect.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

from .base import (
    CompletionRequest,
    CompletionResponse,
    Provider,
    StreamComplete,
    StreamEvent,
)


class StubProvider(Provider):
    """Replay a scripted sequence of `CompletionResponse`s.

    Captures each `CompletionRequest` it receives in `self.requests` so tests
    can assert on what the loop sent (messages, tools, system prompt).

    `stream()` is provided for parity with the streaming contract: it pops
    the next scripted response and yields it inside a single
    ``StreamComplete`` event with no preceding deltas. Tests that need to
    verify per-token forwarding should use a richer scripted stream
    provider; this stub exists for terseness in tests that only care about
    loop semantics.
    """

    def __init__(self, responses: Iterable[CompletionResponse]) -> None:
        self._responses: deque[CompletionResponse] = deque(responses)
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError(
                "StubProvider exhausted: loop made more calls than the test scripted."
            )
        return self._responses.popleft()

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError(
                "StubProvider exhausted: loop made more calls than the test scripted."
            )
        response = self._responses.popleft()
        yield StreamComplete(response=response)
