# parasail_batch.py
"""
Parasail Batch API support for MinimaLlm.

Provides async batch submission with:
- Automatic cache filtering (skip already-cached requests)
- Resumption support via state files
- Polling with progress updates
- Cache population from batch results
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .llm_config import MinimaLlmConfig
from .llm_protocol import MinimaLlmRequest

if TYPE_CHECKING:
    from .minima_llm import OpenAIMinimaLlm

# Type alias for JSON-serializable data
Json = Dict[str, Any]




# Here's the complete picture with both code paths:

#   ┌─────────────────────────────────────────────────────────────────────────────┐
#   │  NORMAL FLOW (non-interrupted)                                              │
#   └─────────────────────────────────────────────────────────────────────────────┘

#   run_dspy_batch_generic()  [minima_llm_dspy.py:924]
#       │
#       └── async with backend.batch_mode(prefix):
#               │
#               ├── Phase 1: _collect_requests_for_batch()
#               │   └── generate() returns BatchPendingResponse sentinels
#               │
#               └── Context EXIT triggers:  [minima_llm.py:1168-1170]
#                       │
#                       └── await self._batch_collector.submit_and_wait()
#                               │
#                               ├── _upload_batch() → Parasail
#                               │   └── _save_state() → parasail_batch_{prefix}.json
#                               │
#                               ├── _poll_until_done() → wait for completion
#                               │   └── _save_state() on each poll
#                               │
#                               └── _download_and_resolve()  [parasail_batch.py:344]
#                                       │
#                                       └── cache = self._backend._ensure_cache()
#                                           cache.put(cache_key, text, response_body)
#                                           ↓
#                                           Written to minima_llm.db ✓


#   ┌─────────────────────────────────────────────────────────────────────────────┐
#   │  INTERRUPTED FLOW (process killed during polling)                           │
#   └─────────────────────────────────────────────────────────────────────────────┘

#   1. Process starts normally...
#      └── _upload_batch() completes
#          └── State saved: parasail_batch_{prefix}.json (status: "in_progress")

#   2. Process killed during _poll_until_done()
#      └── Batch continues running on Parasail servers (up to 24h)

#   3. User runs recovery command:  [click_plus.py:654-686]

#      python -m judge batch-status --llm-config llm-config.yml --prefix my-batch
#          │
#          └── check_and_populate_cache(prefix, config)  [parasail_batch.py:555]
#                  │
#                  ├── backend = OpenAIMinimaLlm(config)  ← Creates backend
#                  │
#                  ├── collector._load_state()
#                  │   └── Reads parasail_batch_{prefix}.json
#                  │
#                  ├── If status == "completed":
#                  │       │
#                  │       ├── collector._download_file(output_file_id)
#                  │       │
#                  │       └── cache = backend._ensure_cache()  ← Same cache!
#                  │           cache.put(cache_key, text, response_body)
#                  │           ↓
#                  │           Written to minima_llm.db ✓
#                  │
#                  └── backend.aclose()  ← Syncs cache to disk

#   4. User re-runs original workflow:
#      └── Phase 3 retrieval hits cache (100% cache hits)

#   Key insight: Both paths write to the same minima_llm.db because they both use backend._ensure_cache().



# ----------------------------
# Data classes
# ----------------------------

@dataclass
class BatchState:
    """
    Persistent state for a Parasail batch job.

    Stored as JSON in state_dir for resumption after interruption.
    """
    prefix: str  # User-provided batch prefix
    batch_id: Optional[str] = None  # Parasail batch ID (assigned after upload)
    input_file_id: Optional[str] = None  # Parasail file ID for uploaded .jsonl
    output_file_id: Optional[str] = None  # Parasail file ID for results
    status: str = "pending"  # pending, validating, in_progress, finalizing, completed, failed, expired, cancelled
    created_at: float = field(default_factory=time.time)
    custom_id_to_cache_key: Dict[str, str] = field(default_factory=dict)

    def with_status(self, status: str) -> "BatchState":
        """Return new state with updated status."""
        return BatchState(
            prefix=self.prefix,
            batch_id=self.batch_id,
            input_file_id=self.input_file_id,
            output_file_id=self.output_file_id,
            status=status,
            created_at=self.created_at,
            custom_id_to_cache_key=self.custom_id_to_cache_key,
        )

    def with_output_file_id(self, output_file_id: str) -> "BatchState":
        """Return new state with output_file_id set."""
        return BatchState(
            prefix=self.prefix,
            batch_id=self.batch_id,
            input_file_id=self.input_file_id,
            output_file_id=output_file_id,
            status=self.status,
            created_at=self.created_at,
            custom_id_to_cache_key=self.custom_id_to_cache_key,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "prefix": self.prefix,
            "batch_id": self.batch_id,
            "input_file_id": self.input_file_id,
            "output_file_id": self.output_file_id,
            "status": self.status,
            "created_at": self.created_at,
            "custom_id_to_cache_key": self.custom_id_to_cache_key,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchState":
        """Deserialize from dict."""
        return cls(
            prefix=data["prefix"],
            batch_id=data.get("batch_id"),
            input_file_id=data.get("input_file_id"),
            output_file_id=data.get("output_file_id"),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", time.time()),
            custom_id_to_cache_key=data.get("custom_id_to_cache_key", {}),
        )


@dataclass
class BatchResult:
    """Result summary from a batch submission."""
    cached_count: int  # Requests skipped (already in cache)
    submitted_count: int  # Requests uploaded to Parasail
    completed_count: int  # Successful results from batch
    failed_requests: List[Tuple[str, Any]]  # (custom_id, error) for failures


# ----------------------------
# Batch Collector
# ----------------------------

class BatchCollector:
    """
    Collects requests during batch mode, handles submission and polling.

    Usage:
        collector = BatchCollector(config, "my-batch-prefix")
        future1 = collector.add_request(req1, cache_key1)
        future2 = collector.add_request(req2, cache_key2)
        # ... add more requests ...
        await collector.submit_and_wait()
        # All futures now resolved, results in cache
    """

    def __init__(self, config: MinimaLlmConfig, prefix: str, backend: "Optional[OpenAIMinimaLlm]" = None):
        self.config = config
        self.prefix = prefix
        self._backend = backend  # Use backend's cache if provided
        self._pending: List[Tuple[MinimaLlmRequest, str, asyncio.Future[bool]]] = []
        self._state_dir = self._get_state_dir()
        self._state_file = self._state_dir / f"parasail_batch_{prefix}.json"

    def _get_state_dir(self) -> Path:
        """Get directory for state files."""
        state_dir = self.config.parasail.state_dir or self.config.cache_dir
        if not state_dir:
            raise ValueError("Either parasail.state_dir or cache_dir must be set for batch mode")
        path = Path(state_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_cache_db_path(self) -> Optional[str]:
        """Get path to cache database."""
        if self.config.cache_dir:
            return os.path.join(self.config.cache_dir, "prompt_cache.db")
        return None

    def add_request(self, req: MinimaLlmRequest, cache_key: str) -> "asyncio.Future[bool]":
        """
        Add request to pending batch, return Future that resolves when result available.

        The Future resolves to True on success, or raises an exception on failure.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending.append((req, cache_key, future))
        return future

    def has_pending(self) -> bool:
        """Check if there are pending requests."""
        return len(self._pending) > 0

    @property
    def pending_count(self) -> int:
        """Number of pending requests."""
        return len(self._pending)

    def _load_state(self) -> Optional[BatchState]:
        """Load existing batch state from file."""
        if self._state_file.exists():
            try:
                with open(self._state_file) as f:
                    return BatchState.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load state file {self._state_file}: {e}")
        return None

    def _save_state(self, state: BatchState) -> None:
        """Save batch state to file."""
        with open(self._state_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def _make_custom_id(self, idx: int) -> str:
        """Generate custom_id for a request."""
        return f"{self.prefix}-{idx}"

    def _make_cache_key(self, req: MinimaLlmRequest) -> str:
        """Generate cache key from request parameters (same as OpenAIMinimaLlm)."""
        obj: Dict[str, Any] = {"model": self.config.model, "messages": req.messages}
        if req.temperature is not None:
            obj["temperature"] = req.temperature
        if req.max_tokens is not None:
            obj["max_tokens"] = req.max_tokens
        if req.extra:
            obj["extra"] = req.extra
        canonical = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    async def submit_and_wait(self) -> BatchResult:
        """
        Submit batch to Parasail, poll until done, populate cache, resolve futures.

        Returns:
            BatchResult with summary of the operation.
        """
        if not self._pending:
            return BatchResult(cached_count=0, submitted_count=0, completed_count=0, failed_requests=[])

        # Check for existing batch (resumption case)
        state = self._load_state()

        if state and state.status in ("validating", "in_progress", "finalizing"):
            print(f"Resuming existing batch {state.batch_id} (status: {state.status})")
        elif state and state.status == "completed":
            print(f"Batch {state.batch_id} already completed, fetching results...")
            return await self._download_and_resolve(state)
        else:
            # New batch - upload
            state = await self._upload_batch()

        # Poll until completion
        state = await self._poll_until_done(state)

        # Download results and resolve futures
        return await self._download_and_resolve(state)

    async def _upload_batch(self) -> BatchState:
        """Create .jsonl file, upload to Parasail, create batch job."""
        # Build .jsonl content
        lines = []
        custom_id_to_cache_key = {}

        for idx, (req, cache_key, _future) in enumerate(self._pending):
            custom_id = self._make_custom_id(idx)
            custom_id_to_cache_key[custom_id] = cache_key

            body: Dict[str, Any] = {
                "model": self.config.model,
                "messages": req.messages,
            }
            if req.temperature is not None:
                body["temperature"] = req.temperature
            if req.max_tokens is not None:
                body["max_tokens"] = req.max_tokens
            if req.extra:
                body.update(req.extra)

            line = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            lines.append(json.dumps(line))

        jsonl_content = "\n".join(lines)

        # Upload file
        print(f"Uploading batch file ({len(lines)} requests)...")
        input_file_id = await self._upload_file(jsonl_content)
        print(f"  File uploaded: {input_file_id}")

        # Create batch
        print(f"Creating batch job...")
        batch_response = await self._create_batch(input_file_id)
        batch_id = batch_response["id"]
        print(f"  Batch created: {batch_id}")

        # Save state for resumption
        state = BatchState(
            prefix=self.prefix,
            batch_id=batch_id,
            input_file_id=input_file_id,
            status=batch_response["status"],
            created_at=time.time(),
            custom_id_to_cache_key=custom_id_to_cache_key,
        )
        self._save_state(state)

        return state

    async def _poll_until_done(self, state: BatchState) -> BatchState:
        """Poll batch status until completed/failed/expired."""
        start_time = time.time()
        max_poll_s = self.config.parasail.max_poll_hours * 3600
        poll_interval = self.config.parasail.poll_interval_s

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > max_poll_s:
                raise TimeoutError(f"Batch {state.batch_id} not completed after {elapsed/3600:.1f}h")

            # Fetch status
            batch_info = await self._get_batch_status(state.batch_id)  # type: ignore[arg-type]
            state = state.with_status(batch_info["status"])
            self._save_state(state)

            status = batch_info["status"]
            if status == "completed":
                output_file_id = batch_info.get("output_file_id")
                if output_file_id:
                    state = state.with_output_file_id(output_file_id)
                    self._save_state(state)
                print(f"Batch {state.batch_id} completed!")
                return state
            elif status in ("failed", "expired", "cancelled"):
                errors = batch_info.get("errors")
                raise RuntimeError(f"Batch {state.batch_id} {status}: {errors}")

            # Progress update
            counts = batch_info.get("request_counts", {})
            total = counts.get("total", "?")
            completed = counts.get("completed", 0)
            failed = counts.get("failed", 0)
            print(f"Batch {state.batch_id}: {status} "
                  f"({completed}/{total} done, {failed} failed, "
                  f"{elapsed/60:.0f}m elapsed)")

            # Wait before next poll
            await asyncio.sleep(poll_interval)

    async def _download_and_resolve(self, state: BatchState) -> BatchResult:
        """Download results, populate cache, resolve all pending futures."""
        if not state.output_file_id:
            raise RuntimeError(f"Batch {state.batch_id} has no output_file_id")

        # Download results
        print(f"Downloading results from {state.output_file_id}...")
        output_content = await self._download_file(state.output_file_id)

        # Parse results into dict keyed by custom_id
        results: Dict[str, Dict[str, Any]] = {}
        for line in output_content.strip().split("\n"):
            if line:
                result = json.loads(line)
                results[result["custom_id"]] = result

        # Get cache from backend (BatchCollector always has backend when used properly)
        cache = self._backend._ensure_cache() if self._backend else None

        completed = 0
        failed_requests: List[Tuple[str, Any]] = []

        for idx, (_req, cache_key, future) in enumerate(self._pending):
            custom_id = self._make_custom_id(idx)

            if custom_id not in results:
                error = f"Request {custom_id} not in batch results"
                failed_requests.append((custom_id, error))
                if not future.done():
                    future.set_exception(RuntimeError(error))
                continue

            result = results[custom_id]

            if result.get("error"):
                error = result["error"]
                failed_requests.append((custom_id, error))
                if not future.done():
                    future.set_exception(RuntimeError(f"Batch request failed: {error}"))
                continue

            # Extract response
            response = result.get("response", {})
            response_body = response.get("body", {})
            choices = response_body.get("choices", [])

            if not choices:
                error = "No choices in response"
                failed_requests.append((custom_id, error))
                if not future.done():
                    future.set_exception(RuntimeError(error))
                continue

            text = choices[0].get("message", {}).get("content", "")

            # Write to cache (backend owns the cache, don't close it)
            if cache:
                cache.put(cache_key, text, response_body)

            completed += 1
            if not future.done():
                future.set_result(True)

        print(f"Batch results: {completed} completed, {len(failed_requests)} failed")

        return BatchResult(
            cached_count=0,  # Already filtered before calling submit_and_wait
            submitted_count=len(self._pending),
            completed_count=completed,
            failed_requests=failed_requests,
        )

    # ----------------------------
    # HTTP helpers for Parasail API
    # ----------------------------

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Parasail API calls."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def _upload_file(self, content: str) -> str:
        """
        Upload .jsonl file to Parasail Files API.

        POST /v1/files with multipart/form-data
        Returns: file_id
        """
        url = f"{self.config.base_url}/files"

        # Build multipart form data
        boundary = "----BatchUploadBoundary" + hashlib.md5(str(time.time()).encode()).hexdigest()[:16]
        body_parts = []

        # purpose field
        body_parts.append(f"--{boundary}".encode())
        body_parts.append(b'Content-Disposition: form-data; name="purpose"')
        body_parts.append(b"")
        body_parts.append(b"batch")

        # file field
        body_parts.append(f"--{boundary}".encode())
        body_parts.append(b'Content-Disposition: form-data; name="file"; filename="batch_input.jsonl"')
        body_parts.append(b"Content-Type: application/jsonl")
        body_parts.append(b"")
        body_parts.append(content.encode("utf-8"))

        body_parts.append(f"--{boundary}--".encode())

        body = b"\r\n".join(body_parts)

        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

        # Run blocking request in thread pool
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._do_request("POST", url, body, headers)
        )

        return response["id"]

    async def _create_batch(self, input_file_id: str) -> Dict[str, Any]:
        """
        Create batch job via POST /v1/batches.

        Returns: batch object with id, status, etc.
        """
        url = f"{self.config.base_url}/batches"
        payload = {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "1h",
            # "completion_window": "24h",
        }

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._do_request("POST", url, json.dumps(payload).encode(), self._get_headers())
        )

    async def _get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get batch status via GET /v1/batches/{batch_id}.

        Returns: batch object with status, request_counts, etc.
        """
        url = f"{self.config.base_url}/batches/{batch_id}"

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._do_request("GET", url, None, self._get_headers())
        )

    async def _cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        """
        Cancel batch via POST /v1/batches/{batch_id}/cancel.

        Returns: batch object with updated status.
        """
        url = f"{self.config.base_url}/batches/{batch_id}/cancel"

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._do_request("POST", url, None, self._get_headers())
        )

    async def _list_batches(self, limit: int = 100) -> Dict[str, Any]:
        """
        List batches via GET /v1/batches.

        Returns: object with 'data' list of batch objects.
        """
        url = f"{self.config.base_url}/batches?limit={limit}"

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._do_request("GET", url, None, self._get_headers())
        )

    async def _download_file(self, file_id: str) -> str:
        """
        Download file content via GET /v1/files/{file_id}/content.

        Returns: file content as string
        """
        url = f"{self.config.base_url}/files/{file_id}/content"

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._do_request_text("GET", url, self._get_headers())
        )

    def _do_request(self, method: str, url: str, body: Optional[bytes], headers: Dict[str, str]) -> Dict[str, Any]:
        """Perform HTTP request and return JSON response."""
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} from {url}: {body_text[:500]}")

    def _do_request_text(self, method: str, url: str, headers: Dict[str, str]) -> str:
        """Perform HTTP request and return text response."""
        req = urllib.request.Request(url, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code} from {url}: {body_text[:500]}")


# ----------------------------
# CLI Utility Functions
# ----------------------------

def check_and_populate_cache(prefix: str, config: MinimaLlmConfig) -> None:
    """
    Check batch status and populate cache if completed.

    Used for resumption after process interruption.

    Args:
        prefix: Batch prefix to check
        config: LLM config (required - use click command to get proper config from llm-config.yml)
    """
    from .minima_llm import OpenAIMinimaLlm

    # Create backend to get the correct cache (uses minima_llm.db)
    backend = OpenAIMinimaLlm(config)
    collector = BatchCollector(config, prefix, backend=backend)
    state = collector._load_state()

    if state is None:
        print(f"No batch state found for prefix '{prefix}'")
        return

    print(f"Batch: {state.batch_id}")
    print(f"Status: {state.status}")
    print(f"Created: {time.ctime(state.created_at)}")
    print(f"Requests: {len(state.custom_id_to_cache_key)}")

    if state.status == "completed":
        print("\nBatch already completed. Populating cache...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if not state.output_file_id:
                print("Error: No output_file_id in state")
                return

            output_content = loop.run_until_complete(collector._download_file(state.output_file_id))

            # Use backend's cache (correct filename: minima_llm.db)
            cache = backend._ensure_cache()
            if not cache:
                print("Error: No cache_dir configured")
                return

            completed = 0
            failed = 0

            for line in output_content.strip().split("\n"):
                if not line:
                    continue
                result = json.loads(line)
                custom_id = result["custom_id"]
                cache_key = state.custom_id_to_cache_key.get(custom_id)

                if not cache_key:
                    print(f"  Warning: Unknown custom_id {custom_id}")
                    continue

                if result.get("error"):
                    failed += 1
                    continue

                response = result.get("response", {})
                response_body = response.get("body", {})
                choices = response_body.get("choices", [])

                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    cache.put(cache_key, text, response_body)
                    completed += 1
                else:
                    failed += 1

            # Don't close cache - backend owns it
            print(f"Results added to cache: {completed} success, {failed} failed")

        finally:
            loop.close()
            # Close backend to sync cache
            loop2 = asyncio.new_event_loop()
            asyncio.set_event_loop(loop2)
            try:
                loop2.run_until_complete(backend.aclose())
            finally:
                loop2.close()

    elif state.status in ("validating", "in_progress", "finalizing"):
        print(f"\nBatch still processing. Checking current status...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            batch_info = loop.run_until_complete(collector._get_batch_status(state.batch_id))  # type: ignore
            counts = batch_info.get("request_counts", {})
            print(f"Current status: {batch_info['status']}")
            print(f"Progress: {counts.get('completed', 0)}/{counts.get('total', '?')} done, "
                  f"{counts.get('failed', 0)} failed")
        finally:
            loop.close()

    elif state.status in ("failed", "expired", "cancelled"):
        print(f"\nBatch {state.status}. Cannot retrieve results.")

    else:
        print(f"\nUnknown status: {state.status}")


def list_batch_states(config: Optional[MinimaLlmConfig] = None) -> None:
    """List all local batch state files."""
    if config is None:
        config = MinimaLlmConfig.from_env()

    state_dir = config.parasail.state_dir or config.cache_dir
    if not state_dir:
        print("No state_dir or cache_dir configured")
        return

    state_path = Path(state_dir)
    if not state_path.exists():
        print(f"State directory does not exist: {state_path}")
        return

    state_files = list(state_path.glob("parasail_batch_*.json"))
    if not state_files:
        print("No batch state files found")
        return

    print(f"Found {len(state_files)} local batch state file(s):\n")
    for sf in sorted(state_files):
        try:
            with open(sf) as f:
                state = BatchState.from_dict(json.load(f))
            print(f"  {state.prefix}")
            print(f"    Batch ID: {state.batch_id}")
            print(f"    Status: {state.status}")
            print(f"    Created: {time.ctime(state.created_at)}")
            print(f"    Requests: {len(state.custom_id_to_cache_key)}")
            print()
        except Exception as e:
            print(f"  {sf.name}: Error reading - {e}")
            print()


def list_remote_batches(config: Optional[MinimaLlmConfig] = None, limit: int = 100) -> None:
    """List batches from remote Parasail API."""
    if config is None:
        config = MinimaLlmConfig.from_env()

    collector = BatchCollector(config, "dummy")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(collector._list_batches(limit))
        batches = result.get("data", [])

        if not batches:
            print("No batches found on remote")
            return

        print(f"Found {len(batches)} batch(es) on Parasail:\n")
        for batch in batches:
            batch_id = batch.get("id", "?")
            status = batch.get("status", "?")
            created = batch.get("created_at")
            created_str = time.ctime(created) if created else "?"
            counts = batch.get("request_counts", {})
            total = counts.get("total", "?")
            completed = counts.get("completed", 0)
            failed = counts.get("failed", 0)

            print(f"  {batch_id}")
            print(f"    Status: {status}")
            print(f"    Created: {created_str}")
            print(f"    Progress: {completed}/{total} done, {failed} failed")
            print()
    finally:
        loop.close()


def cancel_batch(batch_id: str, config: Optional[MinimaLlmConfig] = None) -> None:
    """Cancel a batch on Parasail."""
    if config is None:
        config = MinimaLlmConfig.from_env()

    collector = BatchCollector(config, "dummy")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        print(f"Cancelling batch {batch_id}...")
        result = loop.run_until_complete(collector._cancel_batch(batch_id))
        status = result.get("status", "?")
        print(f"Batch {batch_id} status: {status}")
    except RuntimeError as e:
        print(f"Error: {e}")
    finally:
        loop.close()


