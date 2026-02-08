# Direct Prompt Judge

> **Setup and usage**: See [main judges README](../README.md) for installation, environment setup, and running instructions.

LLM-based judge that evaluates RAG response relevance using direct prompting. Uses different prompt strategies based on dataset type.

## What This Judge Does

**Response-Level Assessment** (no citation validation):

1. **Create Qrels**: Grades each response using LLM prompts (0-3 scale)
   - **UMBRELA prompt** for RAG & RAGTIME datasets (relevance grading)
   - **DRAGUN prompt** for DRAGUN dataset (trustworthiness assessment)

2. **Judge**: Aggregates grades into a leaderboard with measures:
   - `AVG_GRADE`: Average score (0-3)
   - `IS_RELEVANT`: Percentage of responses with grade >= 2

**Note**: This judge evaluates response quality/relevance only. It does NOT validate citations or check if cited documents actually support claims. For citation validation, use the [citation judge](../citation_judge/README.md).

## Prompts

### UMBRELA Prompt (RAG & RAGTIME)

Used for relevance grading on a 0-3 scale:
- **Grade 0**: Completely irrelevant or wrong
- **Grade 1**: Partially relevant but incomplete
- **Grade 2**: Relevant with minor issues
- **Grade 3**: Highly relevant and complete

The prompt asks the LLM to evaluate how well the response answers the query, considering:
- Completeness of answer
- Accuracy of information
- Relevance to query

### DRAGUN Prompt (DRAGUN)

Used for trustworthiness assessment of news article reports:
- **Grade 0**: Misleading or false assessment
- **Grade 1**: Partially accurate but incomplete
- **Grade 2**: Mostly accurate trustworthiness evaluation
- **Grade 3**: Comprehensive and accurate trustworthiness analysis

The prompt asks the LLM to evaluate how well the report assesses the news article's trustworthiness.

## Dataset Selection

The judge requires **explicit dataset specification** via the `--dataset` flag:

```bash
--dataset rag      # Uses UMBRELA prompt, extracts query from topic.title
--dataset ragtime  # Uses UMBRELA prompt, extracts query from topic.title + problem_statement + background
--dataset dragun   # Uses DRAGUN prompt, extracts query from topic.body (news article)
```

See [main README](../README.md#dataset-types) for details on dataset formats.

## Quick Start

```bash
cd trec26/judges

# Set environment variables (see main README)
export OPENAI_BASE_URL="your-endpoint"
export OPENAI_MODEL="your-model"
export OPENAI_API_KEY="your-key"

# Run on RAGTIME dataset
python run_judge.py \
  --judge direct_prompt \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir ./output/ \
  --dataset ragtime

# Test with limited data
python run_judge.py \
  --judge direct_prompt \
  --rag-topics path/to/topics.jsonl \
  --rag-responses path/to/runs/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir ./output/ \
  --dataset ragtime \
  --max-topics 2 \
  --max-runs 1 \
  --name test
```

See [EXAMPLES.md](../EXAMPLES.md) for more examples across all datasets.

## Output Format

### Qrels (default.qrels)
TREC format: `topic_id system doc_id grade`

```
28 system abc123... 2.0
28 system def456... 3.0
29 system ghi789... 1.0
```

### Leaderboard (default.leaderboard.txt)
```
run_01 28 AVG_GRADE 2.5
run_01 28 IS_RELEVANT 1.0
run_01 all AVG_GRADE 2.0
run_01 all IS_RELEVANT 0.75
```

### Debug Logs (when using --name)
JSON Lines format with prompts and LLM outputs:

```jsonl
{"event": "INPUT", "run_id": "run1", "topic_id": "28", "query": "...", "response": "...", "prompt": "..."}
{"event": "OUTPUT", "run_id": "run1", "topic_id": "28", "output": 3, "grade": 3}
```

## Customization

### Modify Prompts

Edit `direct_prompt_judge.py`:

**UMBRELA Prompt** (RAG/RAGTIME):
```python
class UmbrelaPrompt(dspy.Signature):
    """UMBRELA prompting framework for passage grading."""

    __doc__ = dedent("""
        # Modify this prompt for your use case
        Given a query and a passage, provide a score from 0-3...
    """)
```

**DRAGUN Prompt** (DRAGUN):
```python
class DragunPrompt(dspy.Signature):
    """DRAGUN trustworthiness assessment prompt."""

    __doc__ = dedent("""
        # Modify this prompt for trustworthiness evaluation
        Given a news article and a report, evaluate...
    """)
```

### Add New Measures

Edit the spec in `direct_prompt_judge.py`:

```python
UMBRELA_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("AVG_GRADE", aggregate=mean_of_floats, cast=float, default=0.0),
    MeasureSpec("IS_RELEVANT", aggregate=mean_of_bools, cast=bool, default=False),
    # Add your new measure:
    MeasureSpec("PERFECT_MATCHES", aggregate=mean_of_bools, cast=bool, default=False),
))
```

Then add it in the `judge()` method:

```python
builder.add(
    run_id=response.metadata.run_id,
    topic_id=response.metadata.topic_id,
    AVG_GRADE=float(grade),
    IS_RELEVANT=(grade >= relevance_threshold),
    PERFECT_MATCHES=(grade == 3)  # Your new measure
)
```

### Add Variants

Edit `workflow.yml`:

```yaml
variants:
  strict:
    judge_settings:
      relevance_threshold: 3  # Only grade 3 is relevant

  lenient:
    judge_settings:
      relevance_threshold: 1  # Grade >= 1 is relevant

  my_variant:
    judge_settings:
      relevance_threshold: 2
      # Add custom settings
```

Run with variant:
```bash
python direct_prompt_judge.py run \
  --rag-responses path/to/runs/ \
  --rag-topics path/to/topics.jsonl \
  --workflow workflow.yml \
  --variant strict \
  --out-dir output/
```

## Combining with Citation Judge

To evaluate both content relevance AND citation quality, run both judges:

```bash
# 1. Run content relevance
python run_judge.py \
  --judge direct_prompt \
  --rag-topics topics.jsonl \
  --rag-responses runs/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir output/ \
  --dataset ragtime \
  --name content

# 2. Run citation validation
python run_judge.py \
  --judge citation \
  --rag-topics topics.jsonl \
  --rag-responses runs/ \
  --workflow citation_judge/workflow.yml \
  --use-env-llm \
  --out-dir output/ \
  --dataset ragtime \
  --name citations

# 3. Analyze both
cat output/content.leaderboard.txt
cat output/citations.leaderboard.txt
```

## Limitations

- **No Citation Validation**: This judge evaluates response quality only. It does NOT:
  - Check if cited documents exist
  - Verify if citations actually support the claims
  - Assess citation accuracy or completeness

  Use the [citation judge](../citation_judge/README.md) for citation validation.

- **LLM-dependent**: Results vary based on LLM model and prompt design
- **Cost**: Requires API calls for each response (use `--max-topics` for testing)

## Files

```
direct_prompt/
├── direct_prompt_judge.py   # Judge implementation
├── workflow.yml             # Configuration and variants
├── llm-config.yml          # LLM settings (optional)
├── requirements.txt        # Dependencies
└── README.md               # This file
```
