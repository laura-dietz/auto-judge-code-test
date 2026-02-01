# MinimalJudge: A Simple AutoJudge Example

This directory contains a minimal, self-documented example of an AutoJudge implementation. Use it as a starting template for building your own judge.

## Quick Start

There are two ways to run a judge:

### Option 1: Via `trec-auto-judge` CLI (Recommended)

When `judge_class` is specified in `workflow.yml`, you can run directly with the framework CLI:

```bash
# Run with default settings
trec-auto-judge run \
    --workflow ./trec25/judges/minimaljudge/workflow.yml \
    --rag-responses /path/to/responses/ \
    --rag-topics /path/to/topics.jsonl \
    --out-dir ./output/ \
    --llm-config /path/to/llm-config.yml

# Run a specific variant
trec-auto-judge run --workflow workflow.yml --variant strict ...

# Run a parameter sweep
trec-auto-judge run --workflow workflow.yml --sweep keyword-sweep ...
```

### Option 2: Via Judge-Specific Script

You can also run the judge directly via its Python script. This requires a small CLI wrapper:

```python
# minimal_judge.py (at the bottom of the file)
if __name__ == "__main__":
    from trec_auto_judge import auto_judge_to_click_command

    judge = MinimalJudge()
    cli = auto_judge_to_click_command(judge, "minimaljudge")
    cli()
```

Then run:

```bash
# Run with default settings
python ./trec25/judges/minimaljudge/minimal_judge.py run \
    --rag-responses /path/to/responses/ \
    --rag-topics /path/to/topics.jsonl \
    --out-dir ./output/ \
    --llm-config /path/to/llm-config.yml \
    --workflow ./trec25/judges/minimaljudge/workflow.yml

# Run a specific variant
python ./minimal_judge.py run --workflow workflow.yml --variant strict ...
```

The `auto_judge_to_click_command` creates a CLI with `run`, `nuggify`, and `judge` subcommands.

## Files

| File | Purpose |
|------|---------|
| `minimal_judge.py` | Judge implementation (this is what you modify) |
| `workflow.yml` | Configuration: judge_class, lifecycle flags, settings, variants, sweeps |
| `README.md` | This documentation |

## The AutoJudge Protocol

Every judge implements three methods:

```python
class AutoJudge(Protocol):
    nugget_banks_type: Type[NuggetBanksProtocol]  # Declare nugget format

    def create_nuggets(self, rag_responses, rag_topics, llm_config, nugget_banks=None, **kwargs):
        """Create or refine nugget banks. Returns NuggetBanks or None."""
        ...

    def create_qrels(self, rag_responses, rag_topics, llm_config, nugget_banks=None, **kwargs):
        """Create relevance judgments. Returns Qrels or None."""
        ...

    def judge(self, rag_responses, rag_topics, llm_config, nugget_banks=None, qrels=None, **kwargs):
        """Score responses and produce leaderboard. Returns Leaderboard."""
        ...
```

## Creating a Leaderboard

### Step 1: Define the LeaderboardSpec

The spec declares what measures your judge produces:

```python
from trec_auto_judge import LeaderboardSpec, MeasureSpec

MINIMAL_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("SCORE"),              # dtype=float (default): cast to float, aggregate via mean
    MeasureSpec("HAS_KEYWORDS", bool), # dtype=bool: cast to 1.0/0.0, aggregate via mean
))
```

The `dtype` parameter determines casting, aggregation, and default behavior:
- `float` (default): cast to float, aggregate via mean, default 0.0
- `int`: cast to float, aggregate via mean, default 0.0
- `bool`: cast to 1.0/0.0, aggregate via mean, default 0.0
- `str`: keep as string, aggregate via first value, default ""

### Step 2: Build the Leaderboard

Use `LeaderboardBuilder` to collect per-topic scores:

```python
from trec_auto_judge import LeaderboardBuilder

def judge(self, rag_responses, rag_topics, llm_config, **kwargs):
    builder = LeaderboardBuilder(MINIMAL_SPEC)

    for response in rag_responses:
        # Calculate your scores
        score = calculate_score(response)
        has_keywords = check_keywords(response)

        # Add one row per (run_id, topic_id)
        builder.add(
            run_id=response.metadata.run_id,
            topic_id=response.metadata.topic_id,
            values={
                "SCORE": score,
                "HAS_KEYWORDS": has_keywords,
            },
        )

    # Build leaderboard with aggregate "all" rows
    expected_topic_ids = [t.request_id for t in rag_topics]
    leaderboard = builder.build(
        expected_topic_ids=expected_topic_ids,
        on_missing="fix_aggregate",  # or "default", "warn", "error"
    )

    # Verify the leaderboard
    leaderboard.verify(expected_topic_ids=expected_topic_ids, warn=True, on_missing="fix_aggregate")

    return leaderboard
```

The builder automatically:
- Validates measure names (typos fail fast)
- Casts values according to the spec
- Computes aggregate "all" rows using each measure's aggregator

## Creating NuggetBanks

### Step 1: Declare the Nugget Format

```python
from trec_auto_judge.nugget_data import NuggetBanks, NuggetBanksProtocol

class MyJudge(AutoJudge):
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks
```

### Step 2: Create NuggetBank per Topic

```python
from trec_auto_judge.nugget_data import NuggetBank, NuggetQuestion

def create_nuggets(self, rag_responses, rag_topics, llm_config, **kwargs):
    banks = []

    for topic in rag_topics:
        # Create a bank for this topic
        bank = NuggetBank(
            query_id=topic.request_id,
            title_query=topic.title or topic.request_id,
        )

        # Create questions (typically via LLM)
        questions = [
            NuggetQuestion.from_lazy(
                query_id=topic.request_id,
                question="What is the main topic?",
                gold_answers=["Expected answer"],  # Optional
            ),
        ]

        # Add to bank
        bank.add_nuggets(questions)
        banks.append(bank)

    # Combine into multi-topic container
    return NuggetBanks.from_banks_list(banks)
```

## Creating Qrels

### Step 1: Define the QrelsSpec

```python
from trec_auto_judge import QrelsSpec, doc_id_md5

class GradeRecord:
    def __init__(self, topic_id: str, text: str, grade: int):
        self.topic_id = topic_id
        self.text = text
        self.grade = grade

QRELS_SPEC = QrelsSpec[GradeRecord](
    topic_id=lambda r: r.topic_id,
    doc_id=lambda r: doc_id_md5(r.text),  # Hash text as doc_id
    grade=lambda r: r.grade,
    on_duplicate="keep_max",  # or "error", "keep_last"
)
```

### Step 2: Build Qrels

```python
from trec_auto_judge import build_qrels

def create_qrels(self, rag_responses, rag_topics, llm_config, **kwargs):
    records = []

    for response in rag_responses:
        # Grade the response (typically via LLM)
        grade = grade_response(response)
        records.append(GradeRecord(
            topic_id=response.metadata.topic_id,
            text=response.get_report_text(),
            grade=grade,
        ))

    return build_qrels(records=records, spec=QRELS_SPEC)
```

## Configuring with workflow.yml

### Judge Class

To enable running via `trec-auto-judge run`, specify the judge class:

```yaml
judge_class: "trec25.judges.minimaljudge.minimal_judge.MinimalJudge"
```

This is a dotted import path to your AutoJudge class. When specified, the framework can dynamically load and run your judge without a separate script.

### Lifecycle Flags

Control which phases run:

```yaml
create_nuggets: true    # Call create_nuggets()
create_qrels: true      # Call create_qrels()
judge: true             # Call judge()
```

Data flow control:

```yaml
judge_uses_nuggets: true   # Pass nuggets to judge()
judge_uses_qrels: true     # Pass qrels to judge()
qrels_uses_nuggets: true   # Pass nuggets to create_qrels()
```

### Settings

Settings are passed to judge methods as `**kwargs`:

```yaml
# Shared settings (all phases)
settings:
  filebase: "{_name}"

# Phase-specific settings
nugget_settings:
  questions_per_topic: 3

judge_settings:
  keyword_bonus: 0.2
```

### Variants

Named configurations that override defaults:

```yaml
variants:
  strict:
    nugget_settings:
      questions_per_topic: 5
    judge_settings:
      keyword_bonus: 0.1
```

Run with: `--variant strict`

### Sweeps

Grid search over parameter combinations:

```yaml
sweeps:
  grid-search:
    nugget_settings:
      questions_per_topic: [2, 3, 5]
    judge_settings:
      keyword_bonus: [0.1, 0.2, 0.3]
```

Run with: `--sweep grid-search` (runs 3 x 3 = 9 configurations)

## CLI Subcommands

The judge CLI provides three subcommands:

| Command | Description |
|---------|-------------|
| `run` | Execute according to workflow.yml (DEFAULT) |
| `nuggify` | Create nuggets only |
| `judge` | Judge with existing nuggets |

```bash
# Default: run according to workflow
./minimal_judge.py --rag-responses ... --out-dir ...

# Create nuggets only
./minimal_judge.py nuggify --rag-responses ... --store-nuggets nuggets.jsonl

# Judge with existing nuggets
./minimal_judge.py judge --rag-responses ... --nugget-banks nuggets.jsonl --output leaderboard.txt
```

## Output Files

Given `filebase: "minimal"`, the framework generates:

| File | Condition |
|------|-----------|
| `minimal.nuggets.jsonl` | `create_nuggets: true` |
| `minimal.qrels` | `create_qrels: true` |
| `minimal.judgment.json` | `judge: true` |
| `minimal.config.yml` | `judge: true` |

## Verification

All output structures have verification:

```python
# Leaderboard verification
leaderboard.verify(
    expected_topic_ids=expected_topic_ids,
    warn=True,  # Print warnings vs raise errors
    on_missing="fix_aggregate",  # or "default", "warn", "error"
)

# NuggetBanks verification
nugget_banks.verify(
    expected_topic_ids=expected_topic_ids,
    warn=True,
)

# Qrels verification
qrels.verify(
    expected_topic_ids=expected_topic_ids,
    warn=True,
)
```

## Tips

1. **Start simple**: Get the basic flow working before adding LLM calls
2. **Use verification**: Catch errors early with strict verification
3. **Test with variants**: Use workflow variants to test different configurations
4. **Check expected_topic_ids**: Always verify against expected topics to catch missing data
5. **Use filebase**: The `{_name}` variable automatically names outputs by variant

## See Also

- `trec_auto_judge/workflow/README.md` - Full workflow documentation
- `trec25/judges/prefnugget/` - Complex nugget-based judge example
- `trec25/judges/umbrela/` - Qrels-based judge example