# trec_auto_judge/llm/batch/__init__.py
"""
Batch processing module for LLM requests.

Provides Parasail Batch API integration for 50% cost savings on large workloads.
See README.md for usage documentation.
"""

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


def __getattr__(name):
    """Lazy import to avoid circular import issues when running parasail_batch as __main__."""
    if name in __all__:
        from . import parasail_batch
        return getattr(parasail_batch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
