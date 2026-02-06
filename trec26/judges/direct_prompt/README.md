# Direct Prompt Judge - Starter Kit

A simple LLM-based judge for evaluating RAG responses using direct prompting. Uses different prompt strategies based on dataset type.

## What This Judge Does

**Response-Level Assessment** (no citation validation):

1. **Create Qrels**: Grades each response using LLM prompts (0-3 scale)
   - **UMBRELA prompt** for RAG & RAGTIME datasets (relevance grading)
   - **DRAGUN prompt** for DRAGUN dataset (trustworthiness assessment)

2. **Judge**: Aggregates grades into a leaderboard with measures:
   - `AVG_GRADE`: Average score (0-3)
   - `IS_RELEVANT`: Percentage of responses with grade >= 2

**Note**: This judge evaluates response quality/relevance only. It does NOT validate citations or check if cited documents actually support claims.

## Files

```
direct_prompt/
├── direct_prompt_judge.py   # Judge implementation
├── workflow.yml             # Configuration
├── llm-config.yml          # LLM settings (free models)
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Setup

### 1. Create Virtual Environment

```bash
cd trec26/judges/direct_prompt
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup a Free LLM (Choose One)

#### Option A: Ollama (Local, Completely Free)

```bash
# Install Ollama: https://ollama.ai
# Download a model
ollama pull llama3.2:3b

# Verify it's running
curl http://localhost:11434/api/tags
```

The `llm-config.yml` is already configured for Ollama by default.

#### Option B: Groq (Cloud, Free Tier)

1. Sign up at https://console.groq.com
2. Get your free API key
3. Update `llm-config.yml`:

```yaml
base_url: "https://api.groq.com/openai/v1"
model: "llama-3.2-3b-preview"
api_key: "your-groq-api-key"
```

Or set environment variable:
```bash
export GROQ_API_KEY="your-groq-api-key"
```

#### Option C: Together.ai (Cloud, Free Tier)

1. Sign up at https://together.ai
2. Get your free API key
3. Update `llm-config.yml`:

```yaml
base_url: "https://api.together.xyz/v1"
model: "meta-llama/Llama-3.2-3B-Instruct-Turbo"
api_key: "your-together-api-key"
```

## Running the Judge

### Basic Usage

```bash
# Activate environment
source venv/bin/activate

# Run with test data
python direct_prompt_judge.py run \
  --rag-responses /path/to/runs/ \
  --rag-topics /path/to/topics.jsonl \
  --workflow workflow.yml \
  --llm-config llm-config.yml \
  --out-dir output/

# Check outputs
ls output/
# default.qrels              # Response grades
# default.judgment.json      # Leaderboard (JSON)
# default.leaderboard.txt    # Leaderboard (TREC format)
# default.config.yml         # Run configuration
```

### Dataset-Specific Prompts

Use the `--dataset` flag to explicitly control which prompt to use:

```bash
# RAG dataset (uses UMBRELA prompt)
python direct_prompt_judge.py run \
  --rag-responses /path/to/rag_runs/ \
  --rag-topics /path/to/rag_topics.jsonl \
  --workflow workflow.yml \
  --llm-config llm-config.yml \
  --dataset rag \
  --out-dir output/rag/

# RAGTIME dataset (uses UMBRELA prompt)
python direct_prompt_judge.py run \
  --rag-responses /path/to/ragtime_runs/ \
  --rag-topics /path/to/ragtime_topics.jsonl \
  --workflow workflow.yml \
  --llm-config llm-config.yml \
  --dataset ragtime \
  --out-dir output/ragtime/

# DRAGUN dataset (uses DRAGUN trustworthiness prompt)
python direct_prompt_judge.py run \
  --rag-responses /path/to/dragun_runs/ \
  --rag-topics /path/to/dragun_topics.jsonl \
  --workflow workflow.yml \
  --llm-config llm-config.yml \
  --dataset dragun \
  --out-dir output/dragun/
```

**Note:** Without `--dataset`, the judge auto-detects based on topic fields.

### Run with Variants

```bash
# Strict mode (only grade 3 is relevant)
python direct_prompt_judge.py run \
  --rag-responses ../../../tests/resources/spot-check-fully-local/runs/ \
  --rag-topics ../../../tests/resources/example-rag-topics.jsonl \
  --workflow workflow.yml \
  --variant strict \
  --out-dir output/

# Lenient mode (grade >= 1 is relevant)
python direct_prompt_judge.py run \
  --rag-responses ../../../tests/resources/spot-check-fully-local/runs/ \
  --rag-topics ../../../tests/resources/example-rag-topics.jsonl \
  --workflow workflow.yml \
  --variant lenient \
  --out-dir output/
```

### Using run_judge.py Helper

For easier management with filtering and debug logging:

```bash
cd trec26/judges

# Run with environment-based LLM config (recommended)
python run_judge.py \
  --judge direct_prompt \
  --rag-topics /path/to/topics.jsonl \
  --rag-responses /path/to/runs/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir output/ \
  --dataset dragun \
  --name dragun_llm

# Or with limited topics/runs for testing
python run_judge.py \
  --judge direct_prompt \
  --rag-topics /path/to/topics.jsonl \
  --rag-responses /path/to/runs/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir output/ \
  --dataset rag \
  --max-topics 5 \
  --max-runs 3 \
  --name test_run
```

**Note**: When using `--name`, debug logs are automatically created as `{name}.jsonl` in the `--out-dir` directory.

## Output Formats

### Qrels (default.qrels)

TREC format: `topic_id system doc_id grade`

```
28 system abc123... 2.0
28 system def456... 3.0
29 system ghi789... 1.0
```

### Leaderboard (default.leaderboard.txt)

```
my_best_run_01 28 AVG_GRADE 2.5
my_best_run_01 28 IS_RELEVANT 1.0
my_best_run_01 29 AVG_GRADE 1.0
my_best_run_01 29 IS_RELEVANT 0.0
my_best_run_01 all AVG_GRADE 1.75
my_best_run_01 all IS_RELEVANT 0.5
```

### Debug Logs (when using --name or --debug-log)

JSON Lines format with structured logging (automatically created in `--out-dir` when using `--name`):

```jsonl
{"event": "session_start", "timestamp": "2026-02-06T...", "message": "Direct Prompt Judge Debug Session Started"}
{"event": "INPUT", "timestamp": "2026-02-06T...", "run_id": "run1", "topic_id": "28", "query": "...", "response": "...", "prompt": "..."}
{"event": "OUTPUT", "timestamp": "2026-02-06T...", "run_id": "run1", "topic_id": "28", "output": 3, "grade": 3}
```

## Supported Datasets

| Dataset | Prompt Used | Evaluates |
|---------|------------|-----------|
| **RAG** | UMBRELA | Query → Response relevance (0-3) |
| **RAGTIME** | UMBRELA | Report request → Report relevance (0-3) |
| **DRAGUN** | DRAGUN | News article → Trustworthiness assessment quality (0-3) |

### How Dataset Selection Works

The judge requires **explicit dataset specification** - no auto-detection.

**You MUST use the `--dataset` flag** when running via run_judge.py:
```bash
--dataset dragun  # Uses DRAGUN prompt
--dataset rag     # Uses UMBRELA prompt
--dataset ragtime # Uses UMBRELA prompt
```

**Or manually set `topic_format` in workflow.yml** before running:
```yaml
qrels_settings:
  topic_format: dragun  # or: rag, ragtime

judge_settings:
  topic_format: dragun  # or: rag, ragtime
```

**Important**: The `topic_format` setting must be in **both** `qrels_settings` and `judge_settings`:
- `qrels_settings.topic_format` controls which prompt is used (UMBRELA vs DRAGUN)
- `judge_settings.topic_format` controls how queries are extracted from topics

**Note**: Auto-detection has been removed. The judge will raise an error if `topic_format` is not explicitly set.

## Customizing the Judge

### Modify Prompts

Edit `direct_prompt_judge.py`:

**UMBRELA Prompt** (for RAG/RAGTIME):
```python
class UmbrelaPrompt(dspy.Signature):
    """UMBRELA prompting framework for passage grading."""

    __doc__ = dedent("""
        # Modify this prompt for your use case
        Given a query and a passage, provide a score...
    """)
```

**DRAGUN Prompt** (for DRAGUN):
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
  my_variant:
    judge_settings:
      relevance_threshold: 2
      # Add custom settings
```

## Limitations

- **No Citation Validation**: This judge evaluates response quality only. It does NOT:
  - Check if cited documents exist
  - Verify if citations actually support the claims
  - Assess citation accuracy or completeness



## Next Steps

1. **Test with your data**: Replace test data paths with your RAG responses
2. **Try different datasets**: Use `--dataset` flag for explicit control
3. **Compare models**: Test different LLMs with the same prompts
4. **Customize**: Modify prompts, measures, or add new variants
5. **Debug**: Use `--debug-log` to inspect prompt inputs/outputs
