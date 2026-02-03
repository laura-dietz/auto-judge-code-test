# EvalResult Module Design

## Overview

The `eval_results` module provides a clean, type-safe container for IR evaluation results. It replaces the older `Leaderboard` module with a simpler design focused on:

1. **Immutable data** - `EvalResult` is frozen after construction
2. **Explicit parameters** - No hidden defaults for critical operations
3. **Separate filtering semantics** - Topic filtering requires recompute; run filtering does not

## Migration from Leaderboard

| Old (Leaderboard) | New (EvalResult) |
|-------------------|------------------|
| `Leaderboard.load()` | `load()` function |
| `LeaderboardSpec` + `MeasureSpec` | `MeasureSpecs` (inferred from data) |
| `LeaderboardBuilder` | `EvalResultBuilder` |
| `entry.values[measure]` | `entry.measure`, `entry.value` (one per entry) |
| `lb.all_topic_id` | `ALL_TOPIC_ID` constant |
| `filter_and_recompute()` | `filter_runs()` OR `filter_topics_and_recompute()` |

Key difference: EvalResult has **one measure per entry**, while Leaderboard had multiple measures per entry.

## Architecture

```
                         Public API
  load(), write(), EvalResult, EvalResultBuilder
                              |
        +---------------------+---------------------+
        v                     v                     v
+---------------+    +---------------+    +---------------+
|   io.py       |    |  builder.py   |    |verification.py|
|   (B. IO)     |    |  (D. Builder) |    |(C. Verification)|
|               |    |               |    |               |
| - load()      |--->| - add()       |--->| - check_*()   |
| - write()     |    | - filter()    |    | - diagnostics |
| - formats     |    | - build()     |    |               |
+---------------+    +---------------+    +---------------+
                              |
                              v
                     +---------------+
                     | eval_result.py|
                     | (A. EvalResult)|
                     |               |
                     | - EvalResult  |
                     | - EvalEntry   |
                     | - MeasureSpecs|
                     +---------------+
```

## Components

### A. EvalResult (`eval_result.py`)

Immutable data container. Read-only after construction.

**Key Types:**

| Type | Purpose |
|------|---------|
| `EvalEntry` | Single cell: `(run_id, topic_id, measure) -> value` |
| `MeasureSpecs` | Defines expected measures for per-topic and aggregate entries |
| `EvalResult` | Immutable container of entries + specs |
| `ALL_TOPIC_ID` | Constant `"all"` for aggregate rows |

**Entry Structure:**

Each `EvalEntry` holds a single measure-value pair:
```python
EvalEntry(run_id="run1", topic_id="t1", measure="ndcg", value=0.75)
```

This differs from Leaderboard which stored all measures in one entry.

**MeasureSpecs Design:**

Per-topic and aggregate entries can have *different* measures:
- `per_topic`: measures in regular topic entries (e.g., `ndcg`, `recall`, `top_word`)
- `aggregate`: measures in `ALL_TOPIC_ID` entries (e.g., `ndcg`, `recall`)

String measures exist only in `per_topic` (cannot be averaged).

### B. IO (`io.py`)

Load and write in various formats.

**Supported Formats:**

| Format | Columns | Notes |
|--------|---------|-------|
| `trec_eval` | measure, topic_id, value | run_id from filename |
| `tot` | run_id, measure, topic_id, value | 4-column |
| `ir_measures` | run_id, topic_id, measure, value | ir_measures output |
| `ranking` | topic_id, Q0, run_id, rank, value, measure | TREC ranking format |
| `jsonl` | JSON with all fields | Most portable |

**Critical Parameters (all keyword-only, required):**

```python
load(
    path,
    format,
    has_header=False,
    drop_aggregates=False,
    *,
    recompute_aggregates: bool,  # REQUIRED
    verify: bool,                 # REQUIRED
    on_missing: Literal["error", "warn", "ignore"],  # REQUIRED
)
```

No defaults for critical params - forces explicit choice.

### C. Verification (`verification.py`)

Validation checks called by `Builder.build()`.

**Active Checks:**

| Check | Purpose |
|-------|---------|
| `check_same_topics_per_run` | All runs must have same topic set |
| `check_measures_match_specs` | Entries must have expected measures |
| `check_aggregates_match_specs` | Aggregates must exist for all runs x measures |

**Failure Handling:**

```python
on_fail: Literal["error", "warn", "ignore"]
```

### D. Builder (`builder.py`)

Mutable construction, filtering, and aggregation.

**Lifecycle:**

```
            EvalResultBuilder
                    |
  add() --->  _entries: List[_RawEntry]
                    |
               build()
                    |
        1. Cast values (string -> float)
        2. Compute OR preserve aggregates
        3. Run verification
        4. Return immutable EvalResult
```

**Critical: `compute_aggregates` behavior**

| Value | Behavior |
|-------|----------|
| `True` | REPLACE all aggregates with macro-averages |
| `False` | PRESERVE existing aggregate entries |

**Warning:** `compute_aggregates=True` discards aggregate-only measures.

## Filtering Operations

### The Two Filtering Dimensions

**1. Topic filtering** - Changes which topics are included
- Aggregates become INVALID (they're averages over old topic set)
- MUST recompute aggregates
- Use: `filter_topics_and_recompute(topic_ids=...)`

**2. Run filtering** - Changes which runs are included
- Aggregates remain VALID (each run's aggregate is independent)
- Should NOT recompute (preserves original aggregate values)
- Use: `filter_runs(run_ids=...)`

### Method Summary

| Method | Recomputes? | Use Case |
|--------|-------------|----------|
| `filter_runs(run_ids)` | No | Select subset of runs, preserve aggregates |
| `filter_topics_and_recompute(topic_ids)` | Yes | Select topics, get fresh aggregates |
| `filter_and_recompute(run_ids, topic_ids)` | Yes | Both filters, always recomputes |

### Why This Matters

When computing correlations, you often want to:
1. Filter to common runs between truth and eval
2. Preserve the original aggregate values (not recompute)

If you recompute when filtering runs, you may get different aggregate values than the original file (especially if the original used weighted averaging).

## Aggregate Discrepancy Detection

File aggregates may differ from simple macro-averages when:
- Upstream uses weighted averaging
- Different aggregation formulas (f1 vs f1_macro)
- Data errors

**Detection Methods:**

```python
# Check for discrepancies
discrepancies = result.check_aggregate_discrepancies(tolerance=0.001)
# Returns: {run_id: {measure: (original, recomputed, diff)}}

# Human-readable report
report = result.report_aggregate_discrepancies()
print(report)
```

## Common Patterns

### Load and Preserve Aggregates (Recommended Default)

```python
result = load(
    path,
    format="ir_measures",
    drop_aggregates=False,
    recompute_aggregates=False,
    verify=True,
    on_missing="warn",
)
```

### Load and Recompute Aggregates

```python
result = load(
    path,
    format="ir_measures",
    drop_aggregates=True,
    recompute_aggregates=True,
    verify=True,
    on_missing="warn",
)
```

### Filter Runs Without Recompute

```python
# Keep only specific runs, preserve their aggregate values
filtered = result.filter_runs({"run_a", "run_b", "run_c"})
```

### Filter Topics With Recompute

```python
# Keep only specific topics, get fresh aggregates
filtered = result.filter_topics_and_recompute(topic_ids={"t1", "t2"})
```

### Compute Correlation Between Truth and Eval

```python
from trec_auto_judge.evaluation_v2 import LeaderboardEvaluator

evaluator = LeaderboardEvaluator(
    truth_leaderboard=truth_path,
    truth_format="ir_measures",
    eval_format="tot",
    on_missing="skip",
    # Don't drop/recompute - preserve original aggregates
    truth_drop_aggregate=False,
    eval_drop_aggregate=False,
)

# evaluate() handles filtering internally
results = evaluator.evaluate(eval_path)
```

## Design Principles

### 1. Explicit Over Implicit

No hidden defaults. Critical parameters must be passed explicitly.

### 2. Fail-Fast Verification

Detailed diagnostics show exactly what's wrong and where.

### 3. Separate Per-Topic from Aggregate

Different measures can exist in per-topic vs aggregate entries.

### 4. Immutable Result

`EvalResult` is frozen. All mutation goes through `EvalResultBuilder`.

### 5. Preserve vs Recompute Awareness

Topic filtering invalidates aggregates; run filtering does not.

## Known Limitations

1. **Aggregation is macro-average only** - No micro-average, median, etc.

2. **No incremental updates** - Must rebuild to add entries

3. **Memory-resident** - All entries in memory

4. **String measures not aggregated** - By design

5. **One entry per (run_id, topic_id, measure)** - Different from Leaderboard's multi-measure entries