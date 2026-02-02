# EvalResult Module Design

## Overview

The `eval_results` module provides a clean, type-safe container for IR evaluation results. It handles loading, storing, verifying, and serializing evaluation metrics across multiple runs, topics, and measures.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Public API                              │
│  load(), write(), EvalResult, EvalResultBuilder                 │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   io.py       │    │  builder.py   │    │verification.py│
│   (B. IO)     │    │  (D. Builder) │    │(C. Verification)│
│               │    │               │    │               │
│ - load()      │───▶│ - add()       │───▶│ - check_*()   │
│ - write()     │    │ - filter()    │    │ - diagnostics │
│ - formats     │    │ - build()     │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
                              │
                              ▼
                     ┌───────────────┐
                     │ eval_result.py│
                     │ (A. EvalResult)│
                     │               │
                     │ - EvalResult  │
                     │ - EvalEntry   │
                     │ - MeasureSpecs│
                     └───────────────┘
```

## Components

### A. EvalResult (`eval_result.py`)

Immutable data container. Read-only after construction.

**Key Types:**

| Type | Purpose |
|------|---------|
| `EvalEntry` | Single cell: `(run_id, topic_id, measure) → value` |
| `MeasureSpecs` | Defines expected measures for per-topic and aggregate entries |
| `EvalResult` | Immutable container of entries + specs |
| `ALL_TOPIC_ID` | Constant `"all"` for aggregate rows |

**MeasureSpecs Design:**

Per-topic and aggregate entries can have *different* measures:
- `per_topic`: measures in regular topic entries (e.g., `ndcg`, `recall`, `top_word`)
- `aggregate`: measures in `ALL_TOPIC_ID` entries (e.g., `ndcg`, `recall`, `character_count_max`)

This separation supports:
1. String measures that only exist per-topic (can't be averaged)
2. Aggregate-only measures computed across all topics (e.g., `character_count_max`)

### B. IO (`io.py`)

Load and write in various formats.

**Supported Formats:**

| Format | Columns | Notes |
|--------|---------|-------|
| `trec_eval` | measure, topic_id, value | run_id from filename |
| `tot` | run_id, measure, topic_id, value | TREC-style 4-column |
| `ir_measures` | run_id, topic_id, measure, value | ir_measures output |
| `jsonl` | JSON with all fields | Most portable |

**Key Design Decisions:**

1. **Specs are inferred from data**: `MeasureSpecs.infer()` examines entries to determine which measures exist in per-topic vs aggregate rows
2. **No default values for critical params**: `recompute_aggregates`, `verify`, `on_missing` must be explicit
3. **Written files have no headers**: Simplifies roundtrip testing

### C. Verification (`verification.py`)

Validation checks called by `Builder.build()`.

**Active Checks (used by default):**

| Check | Purpose |
|-------|---------|
| `check_same_topics_per_run` | All runs must have same topic set |
| `check_measures_match_specs` | Entries must have expected measures |
| `check_aggregates_match_specs` | Aggregates must exist for all runs × measures |

**Legacy Checks (not used by default):**

| Check | Purpose |
|-------|---------|
| `check_complete_grid` | All (run, topic, measure) combinations exist |
| `check_no_extra_aggregates` | No aggregates without per-topic data |
| `check_consistent_dtypes` | Values match inferred dtype |

**Failure Handling:**

```python
on_fail: Literal["error", "warn", "ignore"]
```

- `"error"`: Raise `VerificationError`
- `"warn"`: Print to stderr, continue
- `"ignore"`: Silent, continue

**Diagnostic Messages:**

When measures mismatch, detailed diagnostics show:
- Which measures are missing
- How many entries are affected
- Where the measure IS present (for debugging)

Example:
```
Per-topic measure mismatches:
  - 'character_count_max': missing from 1220/1281 entries (only in: lichen, radish)
```

### D. Builder (`builder.py`)

Mutable construction, filtering, and aggregation.

**Lifecycle:**

```
            ┌──────────────────────────────────────────────┐
            │               EvalResultBuilder              │
            │                                              │
  add() ───▶│  _entries: List[_RawEntry]  (uncasted)      │
            │  _specs: MeasureSpecs                        │
            │                                              │
            └─────────────────────┬────────────────────────┘
                                  │ build()
                                  ▼
            ┌──────────────────────────────────────────────┐
            │  1. Cast values (string → float)             │
            │  2. Compute OR preserve aggregates           │
            │  3. Run verification                         │
            │  4. Return immutable EvalResult              │
            └──────────────────────────────────────────────┘
```

**Critical: `compute_aggregates` behavior**

| Value | Behavior |
|-------|----------|
| `True` | REPLACE all aggregates with freshly computed macro-averages |
| `False` | PRESERVE existing aggregate entries from input |

**Warning:** `compute_aggregates=True` discards aggregate-only measures (like `character_count_max`) because they cannot be recomputed from per-topic data.

## Data Flow

### Loading from File

```
File → _parse_file() → [EvalEntry...] → MeasureSpecs.infer()
                                              │
                                              ▼
                               EvalResultBuilder(specs)
                                              │
                                         builder.add()
                                              │
                                         builder.build()
                                              │
                                              ▼
                                         EvalResult
```

### Roundtrip Test

```
load(path, ...) → EvalResult → write(result, tmp) → load(tmp, ...) → compare
                     │                                    │
                     └────────────────────────────────────┘
                              Must be identical
```

## Design Principles

### 1. Explicit Over Implicit

No hidden defaults. Critical parameters (`compute_aggregates`, `verify`, `on_missing`) must be passed explicitly to prevent confusion about what processing is applied.

### 2. Fail-Fast Verification

Verification catches data issues early with detailed diagnostics. Errors include:
- Exact counts of affected entries
- Examples of where data IS correct (for debugging)

### 3. Separate Per-Topic from Aggregate

Different measures can exist in per-topic vs aggregate entries. This supports:
- String measures (per-topic only, can't average)
- Aggregate-only measures (computed differently)

### 4. Immutable Result

`EvalResult` is frozen after construction. All mutation goes through `EvalResultBuilder`, ensuring data integrity.

## Common Patterns

### Load and Preserve Existing Aggregates

```python
result = load(
    path,
    format="ir_measures",
    recompute_aggregates=False,  # Keep original aggregates
    verify=True,
    on_missing="warn",
)
```

### Load and Recompute Aggregates

```python
result = load(
    path,
    format="ir_measures",
    recompute_aggregates=True,  # Replace with macro-averages
    verify=True,
    on_missing="warn",
)
```

### Build from Records

```python
builder = EvalResultBuilder(specs)
builder.add_records(
    records=my_data,
    run_id=lambda r: r.run_name,
    topic_id=lambda r: r.query_id,
    measures=lambda r: {"ndcg": r.ndcg, "recall": r.recall},
)
result = builder.build(
    compute_aggregates=True,
    verify=True,
    on_missing="error",
)
```

## Known Limitations

1. **Aggregation is macro-average only**: No support for other aggregation methods (micro-average, median, etc.)

2. **No incremental updates**: Must rebuild entire `EvalResult` to add entries

3. **Memory-resident**: All entries loaded into memory; not suitable for very large result sets

4. **String measures not aggregated**: By design, but may surprise users expecting all measures to have aggregates
