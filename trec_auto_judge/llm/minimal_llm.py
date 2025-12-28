from __future__ import annotations

"""Minimal LLM endpoint. OpenAI-compatible async adapter (stdlib-only).

This adapter is intended as a *beginner-friendly default* for the TREC Auto-Judge
starter kit. Advanced users can ignore it and use the same environment variables
to configure their own LiteLLM/LangChain/DSPy stack.

Features
- OpenAI-compatible HTTP: POST /v1/chat/completions
- Backpressure / onslaught avoidance:
  - Semaphore caps max outstanding requests
  - Simple RPM pacing gate (minimum inter-arrival spacing)
- Retries with exponential backoff + jitter
- Honors Retry-After when present
- Built in batching and progress, failure tracking
- Optional request-body gzip compression
- stdlib only (urllib in a background thread via asyncio.to_thread)

Parsing of LLM config from environment via `MinimaLlmConfig`
"""

import asyncio
import gzip
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .llm_protocol import AsyncMinimaLlmBackend, MinimaLlmRequest, MinimaLlmResponse, Json
from .llm_config import MinimaLlmConfig


import os
from dataclasses import dataclass
from typing import Optional


# ----------------------------
# Backpressure primitives
# ----------------------------

class _RPMGate:
    """Dependency-free RPM pacing using minimum inter-arrival spacing."""

    def __init__(self, rpm: int):
        self._rpm = rpm
        self._lock = asyncio.Lock()
        self._next_ok = 0.0

    async def acquire(self) -> None:
        if self._rpm <= 0:
            return

        min_interval = 60.0 / float(self._rpm)
        async with self._lock:
            now = time.monotonic()
            wait = self._next_ok - now
            self._next_ok = (self._next_ok + min_interval) if wait > 0 else (now + min_interval)

        if wait > 0:
            await asyncio.sleep(wait)


class _Cooldown:
    """Global adaptive delay (decays exponentially over time)."""

    def __init__(self, floor_s: float, cap_s: float, halflife_s: float):
        self._floor = floor_s
        self._cap = cap_s
        self._halflife = max(1e-6, halflife_s)
        self._delay = 0.0
        self._t_last = time.monotonic()
        self._lock = asyncio.Lock()

    def _decay_locked(self) -> float:
        now = time.monotonic()
        dt = now - self._t_last
        if dt > 0:
            self._delay *= 0.5 ** (dt / self._halflife)
            self._t_last = now
        return self._delay

    async def pre_request_sleep(self) -> None:
        async with self._lock:
            d = self._decay_locked()
        if d > 0:
            await asyncio.sleep(d)

    async def bump(self, suggested_s: float) -> None:
        async with self._lock:
            self._decay_locked()
            target = max(self._floor, suggested_s)
            self._delay = min(self._cap, max(self._delay, target))


def _join_url(base: str, path: str) -> str:
    return base.rstrip("/") + path



# ----------------------------
#  Batcher
# ----------------------------

@dataclass(frozen=True)
class MinimaLlmFailure:
    request_id: str
    error_type: str
    message: str


Result = Union[MinimaLlmResponse, MinimaLlmFailure]



class _Heartbeat:
    def __init__(self, *, n_total: int, heartbeat_s: float, stall_s: float):
        self._n_total = n_total
        self._heartbeat_s = heartbeat_s
        self._stall_s = stall_s
        self._t0 = time.monotonic()
        self._last_completion_t = self._t0
        self._done = 0
        self._ok = 0
        self._fail = 0

    @staticmethod
    def _fmt_s(seconds: float) -> str:
        seconds = max(0.0, seconds)
        if seconds < 90:
            return f"{int(seconds)}s"
        minutes = int(seconds // 60)
        if minutes < 90:
            return f"{minutes}m{int(seconds - minutes*60)}s"
        hours = minutes // 60
        return f"{hours}h{minutes % 60}m"

    def mark_completed(self, *, is_failure: bool) -> None:
        self._done += 1
        self._last_completion_t = time.monotonic()
        if is_failure:
            self._fail += 1
        else:
            self._ok += 1

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_s)
            now = time.monotonic()
            elapsed = now - self._t0
            since_last = now - self._last_completion_t
            rate = (self._done / elapsed) if elapsed > 0 else 0.0
            remaining = self._n_total - self._done
            eta = (remaining / rate) if rate > 0 else float("inf")

            stall_note = ""
            if since_last > self._stall_s:
                stall_note = f"  STALL WARNING: no completions for {self._fmt_s(since_last)}"

            print(
                f"[{self._fmt_s(elapsed)}] done {self._done}/{self._n_total} "
                f"(ok {self._ok}, fail {self._fail}) "
                f"rate {rate:.2f}/s "
                f"ETA {self._fmt_s(eta) if eta != float('inf') else '??'} "
                f"since_last {self._fmt_s(since_last)}"
                f"{stall_note}"
            )


class _FailureTracker:
    def __init__(self, *, max_failures: Optional[int], print_first: int, keep: int):
        self._max_failures = max_failures
        self._print_first = print_first
        self._keep = keep
        self._count = 0
        self._ring: List[MinimaLlmFailure] = []

    def record(self, fail: "MinimaLlmFailure") -> None:
        self._count += 1

        if self._count <= self._print_first:
            print(f"Failure {self._count} on request_id={fail.request_id}")
            print(f"{fail.error_type}: {fail.message[:1000]}")

        self._ring.append(fail)
        if len(self._ring) > self._keep:
            self._ring.pop(0)

    def should_abort(self) -> bool:
        return self._max_failures is not None and self._count > self._max_failures

    def abort_exception(self) -> RuntimeError:
        assert self._max_failures is not None
        summary_lines = [
            f"{f.request_id}: {f.error_type}: {f.message[:300]}"
            for f in self._ring
        ]
        summary = "\n  " + "\n  ".join(summary_lines) if summary_lines else ""
        return RuntimeError(
            f"Aborting batch: {self._count} failures (limit {self._max_failures})."
            f"{summary}"
        )


# ----------------------------
# The backend
# ----------------------------



class OpenAIMinimaLlm(AsyncMinimaLlmBackend):
    """Minimal OpenAI-compatible end point with retry/rate limiter/batching. Implements AsyncMinimaLlmBackend."""

    def __init__(self, cfg: MinimaLlmConfig):
        self.cfg = cfg
        self._sem = asyncio.Semaphore(cfg.max_outstanding)
        self._rpm = _RPMGate(cfg.rpm)
        self._cooldown = _Cooldown(cfg.cooldown_floor_s, cfg.cooldown_cap_s, cfg.cooldown_halflife_s)
        self._closed = False

        b = cfg._normalize_base_url(cfg.base_url)
        self._has_v1 = b.endswith("/v1")
        self._base = b

    @classmethod
    def from_env(cls) -> "OpenAIMinimaLlm":
        return cls.read_config_from_env()

    async def aclose(self) -> None:
        # urllib has no session to close; keep for symmetry.
        self._closed = True

    def _endpoint(self, path: str) -> str:
        # If base_url already ends in /v1, avoid duplicating /v1.
        if self._has_v1 and path.startswith("/v1/"):
            path = path[len("/v1") :]
        return _join_url(self._base, path)

    @staticmethod
    def _parse_retry_after(headers: Dict[str, str]) -> Optional[float]:
        ra = headers.get("Retry-After") or headers.get("retry-after")
        if not ra:
            return None
        try:
            return float(ra)
        except ValueError:
            return None

    @staticmethod
    def _should_retry(status: int) -> bool:
        return status in (408, 409, 425, 429, 500, 502, 503, 504)

    def _backoff(self, attempt: int) -> float:
        base = min(self.cfg.max_backoff_s, self.cfg.base_backoff_s * (2 ** (attempt - 1)))
        j = self.cfg.jitter
        return max(0.0, base * (1.0 + random.uniform(-j, j)))

    def _post_sync(self, url: str, payload: Json) -> Tuple[int, Dict[str, str], bytes]:
        raw = json.dumps(payload).encode("utf-8")
        body = gzip.compress(raw) if self.cfg.compress_gzip else raw

        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key is not None:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        if self.cfg.compress_gzip:
            headers["Content-Encoding"] = "gzip"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                status = getattr(resp, "status", 200)
                hdrs = dict(resp.headers.items())
                data = resp.read()
                return status, hdrs, data

        except urllib.error.HTTPError as e:
            status = e.code
            hdrs = dict(e.headers.items()) if e.headers is not None else {}
            data = e.read() if hasattr(e, "read") else b""
            return status, hdrs, data

    async def _post_json(self, path: str, payload: Json) -> Json:
        if self._closed:
            raise RuntimeError("Adapter is closed")

        url = self._endpoint(path)
        attempt = 0
        last_status: Optional[int] = None
        last_body: bytes = b""  # for error reporting

        while attempt < self.cfg.max_attempts:
            attempt += 1

            await self._cooldown.pre_request_sleep()
            await self._rpm.acquire()
            await self._sem.acquire()
            try:
                status, headers, body = await asyncio.to_thread(self._post_sync, url, payload)
            finally:
                self._sem.release()

            last_status = status
            last_body = body

            if 200 <= status < 300:
                return json.loads(body.decode("utf-8"))

            retry_after = self._parse_retry_after(headers)

            # Bump cooldown on overload-ish signals.
            if status in (429, 503, 504):
                await self._cooldown.bump(suggested_s=(retry_after or self.cfg.cooldown_floor_s))

            if self._should_retry(status) and attempt < self.cfg.max_attempts:
                sleep_s = retry_after if retry_after is not None else self._backoff(attempt)
                await asyncio.sleep(sleep_s)
                continue

            body_txt = body.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compat POST {path} failed: status={status}, attempts={attempt}, body={body_txt[:1000]}"
            )

        body_txt = last_body.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI-compat POST {path} failed after {self.cfg.max_attempts} attempts: "
            f"status={last_status}, body={body_txt[:1000]}"
        )

    async def prompt_one(self, req: MinimaLlmRequest) -> MinimaLlmResponse:
        payload: Json = {
            "model": self.cfg.model,
            "messages": req.messages,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        if req.extra:
            payload.update(req.extra)

        raw = await self._post_json("/v1/chat/completions", payload)
        text = raw["choices"][0]["message"]["content"]
        return MinimaLlmResponse(request_id=req.request_id, text=text, raw=raw)


# ===== batch runner ======


    async def run_batched(self, requests: List[MinimaLlmRequest]) -> List[Result]:
        cfg = self.cfg

        in_q: asyncio.Queue[Tuple[int, MinimaLlmRequest]] = asyncio.Queue()
        out_q: asyncio.Queue[Tuple[int, Result]] = asyncio.Queue()
        n_total = len(requests)

        for i, r in enumerate(requests):
            in_q.put_nowait((i, r))

        async def worker() -> None:
            while True:
                try:
                    idx, req = in_q.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    resp = await self.prompt_one(req)
                    await out_q.put((idx, resp))
                except Exception as e:
                    fail = MinimaLlmFailure(
                        request_id=req.request_id,
                        error_type=type(e).__name__,
                        message=str(e),
                    )
                    await out_q.put((idx, fail))
                finally:
                    in_q.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(cfg.num_workers)]
        results: List[Optional[Result]] = [None] * n_total

        heartbeat = _Heartbeat(n_total=n_total, heartbeat_s=cfg.heartbeat_s, stall_s=cfg.stall_s)
        failure_tracker = _FailureTracker(
            max_failures=cfg.max_failures,
            print_first=cfg.print_first_failures,
            keep=cfg.keep_failure_summaries,
        )
        hb_task = asyncio.create_task(heartbeat.run())

        try:
            done = 0
            while done < n_total:
                idx, item = await out_q.get()
                if results[idx] is not None:
                    continue

                results[idx] = item
                done += 1

                is_failure = isinstance(item, MinimaLlmFailure)
                heartbeat.mark_completed(is_failure=is_failure)

                if is_failure:
                    failure_tracker.record(item)
                    if failure_tracker.should_abort():
                        for w in workers:
                            w.cancel()
                        raise failure_tracker.abort_exception()

            return [r for r in results if r is not None]

        finally:
            hb_task.cancel()
            for w in workers:
                w.cancel()
            await self.aclose()



    # async def run_batched_minimallm(
    #     self,
    #     requests: List[MinimaLlmRequest],
    #     *,
    #     num_workers: int = 64,
    #     max_failures: Optional[int] = None,     # abort if failures > max_failures
    #     heartbeat_s: float = 10.0,              # print status every N seconds
    #     stall_s: float = 300.0,                 # warn if no completions for this long
    #     print_first_failures: int = 5,
    #     keep_failure_summaries: int = 20,
    # ) -> List[Result]:


    #     def _fmt_s(seconds: float) -> str:
    #         seconds = max(0.0, seconds)
    #         if seconds < 90:
    #             return f"{int(seconds)}s"
    #         minutes = int(seconds // 60)
    #         if minutes < 90:
    #             return f"{minutes}m{int(seconds - minutes*60)}s"
    #         hours = minutes // 60
    #         return f"{hours}h{minutes % 60}m"
        
        
    #     in_q: asyncio.Queue[Tuple[int, MinimaLlmRequest]] = asyncio.Queue()
    #     out_q: asyncio.Queue[Tuple[int, Result]] = asyncio.Queue()

    #     n_total = len(requests)
    #     for i, r in enumerate(requests):
    #         in_q.put_nowait((i, r))

    #     async def worker() -> None:
    #         while True:
    #             try:
    #                 idx, req = in_q.get_nowait()
    #             except asyncio.QueueEmpty:
    #                 return
    #             try:
    #                 resp = await self.prompt_one(req)
    #                 await out_q.put((idx, resp))
    #             except Exception as e:
    #                 fail = MinimaLllmFailure(
    #                     request_id=req.request_id,
    #                     error_type=type(e).__name__,
    #                     message=str(e),
    #                 )
    #                 await out_q.put((idx, fail))
    #             finally:
    #                 in_q.task_done()

    #     workers = [asyncio.create_task(worker()) for _ in range(num_workers)]

    #     results: List[Optional[Result]] = [None] * n_total
    #     failures = 0
    #     failure_ring: List[MinimaLllmFailure] = []

    #     t0 = time.monotonic()
    #     done = 0
    #     ok = 0

    #     # updated whenever we receive an item from out_q (success or failure)
    #     last_completion_t = t0

    #     async def heartbeat() -> None:
    #         while True:
    #             await asyncio.sleep(heartbeat_s)
    #             now = time.monotonic()

    #             elapsed = now - t0
    #             since_last = now - last_completion_t

    #             rate = done / elapsed if elapsed > 0 else 0.0
    #             remaining = n_total - done
    #             eta = (remaining / rate) if rate > 0 else float("inf")

    #             stall_note = ""
    #             if since_last > stall_s:
    #                 stall_note = f"  STALL WARNING: no completions for {_fmt_s(since_last)}"

    #             print(
    #                 f"[{_fmt_s(elapsed)}] done {done}/{n_total} "
    #                 f"(ok {ok}, fail {failures}) "
    #                 f"rate {rate:.2f}/s "
    #                 f"ETA {_fmt_s(eta) if eta != float('inf') else '??'} "
    #                 f"since_last {_fmt_s(since_last)}"
    #                 f"{stall_note}"
    #             )

    #     hb_task = asyncio.create_task(heartbeat())

    #     try:
    #         while done < n_total:
    #             idx, item = await out_q.get()
    #             if results[idx] is not None:
    #                 continue

    #             results[idx] = item
    #             done += 1
    #             last_completion_t = time.monotonic()

    #             if isinstance(item, MinimaLllmFailure):
    #                 failures += 1

    #                 if failures <= print_first_failures:
    #                     print(f"Failure {failures} on request_id={item.request_id}")
    #                     print(f"{item.error_type}: {item.message[:1000]}")

    #                 failure_ring.append(item)
    #                 if len(failure_ring) > keep_failure_summaries:
    #                     failure_ring.pop(0)

    #                 if max_failures is not None and failures > max_failures:
    #                     for w in workers:
    #                         w.cancel()

    #                     summary_lines = [
    #                         f"{f.request_id}: {f.error_type}: {f.message[:300]}"
    #                         for f in failure_ring
    #                     ]
    #                     summary = "\n  " + "\n  ".join(summary_lines) if summary_lines else ""
    #                     raise RuntimeError(
    #                         f"Aborting batch: {failures} failures (limit {max_failures})."
    #                         f"{summary}"
    #                     )
    #             else:
    #                 ok += 1

    #         return [r for r in results if r is not None]

    #     finally:
    #         hb_task.cancel()
    #         for w in workers:
    #             w.cancel()
    #         await self.aclose()
