# Batch Processing Module

This module provides OpenAI/Parasail Batch API integration for MinimaLlm, enabling cost savings on large LLM workloads by submitting requests as batch jobs instead of real-time HTTP.

## Table of Contents

- [For Judge Implementors](#for-judge-implementors)
  - [Configuration](#configuration)
  - [Enabling/Disabling Batch Mode](#enablingdisabling-batch-mode)
  - [Understanding Judge Output](#understanding-judge-output)
  - [CLI Utility](#cli-utility)
- [Integration with MinimaLlm](#integration-with-minimalm)
  - [Signals from Batch Runner](#signals-from-batch-runner)
  - [Collecting Payload](#collecting-payload)
  - [Prompt Cache Integration](#prompt-cache-integration)
- [Reuse Outside MinimaLlm](#reuse-outside-minimalm)

---

## For Judge Implementors

### Configuration

Add batch configuration to your `llm-config.yml`:

```yaml
# llm-config.yml
base_url: "https://api.parasail.io/v1"
model: "meta-llama/Llama-3.3-70B-Instruct"
api_key: "your-api-key"
cache_dir: "./cache"

parasail:
  llm_batch_prefix: "project-name"     # Base prefix for batch state files
  state_dir: "./batch_state"     # Optional, defaults to cache_dir
  poll_interval_s: 30            # Seconds between status checks
  max_poll_hours: 24             # Maximum wait time
  max_batch_size: 50000          # Max requests per batch upload
```

Or via environment variables:

```bash
export OPENAI_BASE_URL="https://api.parasail.io/v1"
export OPENAI_MODEL="meta-llama/Llama-3.3-70B-Instruct"
export OPENAI_API_KEY="your-api-key"
export CACHE_DIR="./cache"
```

### Enabling/Disabling Batch Mode

Batch mode is **enabled** when `parasail.prefix` is set (either directly or computed from `llm_batch_prefix`).

To **disable** batch mode, either:
- Remove the `parasail` section from your config
- Ensure neither `prefix` nor `llm_batch_prefix` is set

**For direct use of the `llm` module:**

Set `parasail.prefix` directly in your config to the desired batch state identifier.

**For AutoJudge implementations (using MinimaLlm via click_plus):**

Set `parasail.llm_batch_prefix` and the full prefix is computed automatically:
```
{llm_batch_prefix}_{out_dir}_{filebase}_{config_name}
```

For example, running with `--out-dir ./output --filebase myrun`:
```
proj_output_myrun_default
```

This ensures different judge configurations/variants get isolated batch state files.

### Understanding Judge Output

When batch mode is active, you'll see output like:

```
Starting batch: 12348 items | workers=1 max_outstanding=1 rpm=10

Submitting batch 'proj_myrun_default' (24696 requests)...
Uploading batch file (24696 requests)...
  File uploaded: file-bi-abc123
Creating batch job...
  Batch created: batch-xyz789
Batch batch-xyz789: in_progress (1000/24696 done, 0 failed, 5m elapsed)
Batch batch-xyz789: in_progress (5000/24696 done, 0 failed, 10m elapsed)
...
Batch batch-xyz789 completed!
Downloading results from file-bo-def456...
Batch results: 24696 completed, 0 failed
```

**Key messages:**

| Message | Meaning |
|---------|---------|
| `Starting batch: N items` | Collection phase started, N requests queued |
| `Submitting batch 'prefix' (M requests)` | Uploading to Parasail API |
| `Batch ... in_progress (X/Y done)` | Polling for completion |
| `Batch ... completed!` | All requests processed |
| `Batch results: N completed, M failed` | Final summary |

**For large batches (>50k requests):**

```
Splitting 75000 requests into 2 batches (max 50000 each)
Uploading batch chunk 'prefix-0' (50000 requests)...
Uploading batch chunk 'prefix-1' (25000 requests)...
Polling 2 batch(es) in parallel...
```

### CLI Utility

The `parasail_batch` module includes a standalone CLI for batch management:

```bash
# Show status of all batches
python -m trec_auto_judge.llm.batch.parasail_batch

# Show all batches including already-processed ones
python -m trec_auto_judge.llm.batch.parasail_batch --all

# Download completed batch results and populate cache
python -m trec_auto_judge.llm.batch.parasail_batch --retrieve my-prefix

# Cancel a batch by prefix
python -m trec_auto_judge.llm.batch.parasail_batch --cancel my-prefix

# Cancel a remote batch by ID
python -m trec_auto_judge.llm.batch.parasail_batch --cancel-remote batch-abc123

# Cancel all local batches
python -m trec_auto_judge.llm.batch.parasail_batch --cancel-all
```

**Example output:**

```
Batch states in ./cache:

Active batches:

  proj_myrun_default
    Batch ID: batch-xyz789
    Status: in_progress
    Progress: 5000/24696 (20.3%) done, 0 failed
    Created: Mon Jan 13 10:30:00 2026

  (2 already-processed batch(es) hidden, use --all to show)

Commands:
  --retrieve PREFIX   Download completed batch and populate cache
  --cancel PREFIX     Cancel batch and delete local state
  --cancel-all        Cancel all local batches
  --help              Show all options
```

---

## Integration with MinimaLlm

This section explains how the batch module integrates with `MinimaLlm` for developers who want to understand or extend the integration.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  MinimaLlm                                                      │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ batch_mode() │───>│ BatchCollect │───>│ Parasail API │       │
│  │   context    │    │     or       │    │   (remote)   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │               │
│         │                   v                   v               │
│         │           ┌──────────────┐    ┌──────────────┐        │
│         └──────────>│ PromptCache  │<───│ State Files  │        │
│                     │  (SQLite)    │    │   (JSON)     │        │
│                     └──────────────┘    └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Signals from Batch Runner

The batch system needs these signals from the caller:

| Signal | Source | Purpose |
|--------|--------|---------|
| `prefix` | `batch_mode(prefix)` | Unique identifier for state files |
| `max_batch_size` | `config.parasail.max_batch_size` | Split threshold for chunking |
| `cache_key` | `_compute_cache_key(request)` | Link batch results to cache |

### Collecting Payload

When `batch_mode()` context is active:

1. **Collection Phase**: `generate()` returns `BatchPendingResponse` sentinels instead of making HTTP calls
2. **Request Queuing**: Each request is added to `BatchCollector._pending` with its cache key and a Future
3. **Context Exit**: `submit_and_wait()` is called automatically

```python
# Inside MinimaLlm.batch_mode()
async with backend.batch_mode(prefix):
    # Phase 1: Collection - requests are queued
    for item in items:
        result = await module(item)  # Returns BatchPendingResponse

    # Phase 2: Submit - happens on context exit
    # BatchCollector.submit_and_wait() uploads and polls
```

### Prompt Cache Integration

The batch module writes results directly to MinimaLlm's prompt cache:

```python
# After downloading batch results:
cache = backend._ensure_cache()  # Get MinimaLlm's SQLite cache
for custom_id, result in results.items():
    cache_key = state.custom_id_to_cache_key[custom_id]
    text = result["response"]["body"]["choices"][0]["message"]["content"]
    cache.put(cache_key, text, result["response"]["body"])
```

**Important considerations:**

1. **Cache Key Consistency**: The same `_compute_cache_key()` logic must be used for both batch requests and cache lookups
2. **Cache Ownership**: `BatchCollector` uses the backend's cache instance, doesn't create its own
3. **Sync on Close**: Call `backend.aclose()` to ensure cache is synced to disk
4. **State File Cleanup**: State files remain after processing for resumption; use `--cancel` to delete

---

## Reuse Outside MinimaLlm

The `BatchCollector` class can be adapted for other LLM backends:

### Requirements

1. **OpenAI-compatible batch API**: Must support `/v1/files` and `/v1/batches` endpoints
2. **Cache interface**: Implement `get(key)` and `put(key, text, metadata)` methods
3. **Request format**: Must produce `MinimaLlmRequest` objects

### Minimal Integration Example

```python
from trec_auto_judge.llm.batch import BatchCollector, BatchState
from trec_auto_judge.llm import MinimaLlmConfig

class MyBackend:
    def __init__(self, config: MinimaLlmConfig):
        self.cfg = config
        self._cache = MyCache(config.cache_dir)

    def _ensure_cache(self):
        return self._cache

    async def batch_mode(self, prefix: str):
        collector = BatchCollector(self.cfg, prefix, backend=self)

        # Check for completed batch from previous run
        await collector.populate_cache_if_completed()

        try:
            yield collector
        finally:
            if collector.has_pending():
                await collector.submit_and_wait()
```

### Key Extension Points

| Class/Function | Purpose | Extension Point |
|----------------|---------|-----------------|
| `BatchCollector` | Request collection and submission | Subclass for custom upload logic |
| `BatchState` | Resumption state | Add fields for custom metadata |
| `_upload_file()` | File upload to API | Override for different auth |
| `_poll_until_done()` | Status polling | Customize polling behavior |

### State File Format

State files are JSON with this structure:

```json
{
  "prefix": "proj_myrun_default",
  "batch_id": "batch-xyz789",
  "input_file_id": "file-bi-abc123",
  "output_file_id": "file-bo-def456",
  "status": "completed",
  "created_at": 1736784600.0,
  "custom_id_to_cache_key": {
    "proj_myrun_default-0": "a1b2c3...",
    "proj_myrun_default-1": "d4e5f6...",
    ...
  }
}
```

This enables:
- **Resumption**: Detect in-progress batches after process restart
- **Cache population**: Map batch results back to cache keys
- **Debugging**: Inspect batch state for troubleshooting
