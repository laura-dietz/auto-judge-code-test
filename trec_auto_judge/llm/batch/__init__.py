# trec_auto_judge/llm/batch/__init__.py
"""
Batch processing module for LLM requests.

Provides Parasail Batch API integration for 50% cost savings on large workloads.
See README.md for usage documentation.
"""

from .parasail_batch import (
    # Core classes
    BatchCollector,
    BatchState,
    BatchResult,
    # CLI utility functions
    check_and_populate_cache,
    batch_status_overview,
    cancel_batch,
    cancel_all_batches,
    cancel_all_local_batches,
    list_remote_batches,
)

__all__ = [
    # Core classes
    "BatchCollector",
    "BatchState",
    "BatchResult",
    # CLI utility functions
    "check_and_populate_cache",
    "batch_status_overview",
    "cancel_batch",
    "cancel_all_batches",
    "cancel_all_local_batches",
    "list_remote_batches",
]
