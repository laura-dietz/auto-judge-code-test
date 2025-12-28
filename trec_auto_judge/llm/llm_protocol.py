# backend_protocol.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


Json = Dict[str, Any]


@dataclass(frozen=True)
class MinimaLlmRequest:
    """
    Minimal request shape for an auto-judge call.

    Keep this stdlib-only so participant tool choices do not conflict with
    the track harness.
    """
    request_id: str
    messages: List[Dict[str, str]]                 # OpenAI chat format
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra: Optional[Json] = None                   # any OpenAI-compatible fields


@dataclass(frozen=True)
class MinimaLlmResponse:
    request_id: str
    text: str
    raw: Optional[Json] = None                     # optional raw provider response


@runtime_checkable
class AsyncMinimaLlmBackend(Protocol):
    """
    One-method interface: the harness calls judge_one() for each request.

    Backends decide how to do concurrency/rate-limits internally.
    """
    async def prompt_one(self, req: MinimaLlmRequest) -> MinimaLlmResponse:
        ...

    async def aclose(self) -> None:
        ...
