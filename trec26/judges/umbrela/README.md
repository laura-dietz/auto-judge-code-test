# Simple UMBRELA Judge - Starter Kit

A simple working implementation of the UMBRELA framework for evaluating RAG responses.

## What This Judge Does

1. **Create Qrels**: Grades each RAG response using UMBRELA (0-3 scale)
2. **Judge**: Aggregates grades into a leaderboard with measures:
   - `AVG_GRADE`: Average UMBRELA score (0-3)
   - `IS_RELEVANT`: Percentage of responses with grade >= 2

## Files

```
umbrela/
├── umbrela_judge.py   # Judge implementation
├── workflow.yml       # Configuration
├── llm-config.yml     # LLM settings (free models)
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## Setup

### 1. Create Virtual Environment

```bash
cd trec26/judges/umbrela
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

export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_MODEL="llama3.2:3b"

# Run with test data
python umbrela_judge.py run \
  --rag-responses /path/to/runs/ \
  --rag-topics /path/to/topics.jsonl \
  --workflow workflow.yml \
  --llm-config llm-config.yml \
  --out-dir output/

python umbrela_judge.py run \
  --rag-responses ./toy_data/runs/ \
  --rag-topics ./toy_data/topics.jsonl \
  --workflow workflow.yml \
  --llm-config llm-config.yml \
  --out-dir toy_output/

# Check outputs
# ls output/
# default.qrels              # Passage grades
# default.judgment.json      # Leaderboard (JSON)
# default.leaderboard.txt    # Leaderboard (TREC format)
# default.config.yml         # Run configuration
```

### Run with Variants

```bash
# Strict mode (only grade 3 is relevant)
python umbrela_judge.py run \
  --rag-responses ../../../tests/resources/spot-check-fully-local/runs/ \
  --rag-topics ../../../tests/resources/example-rag-topics.jsonl \
  --workflow workflow.yml \
  --variant strict \
  --out-dir output/

# Lenient mode (grade >= 1 is relevant)
python umbrela_judge.py run \
  --rag-responses ../../../tests/resources/spot-check-fully-local/runs/ \
  --rag-topics ../../../tests/resources/example-rag-topics.jsonl \
  --workflow workflow.yml \
  --variant lenient \
  --out-dir output/
```



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

### Configuration (default.config.yml)

Tracks everything needed to reproduce the run:

```yaml
name: default
create_qrels: true
judge: true
llm_model: llama3.2:3b
timestamp: 2026-01-30T...
git:
  commit: abc123...
  dirty: false
settings:
  filebase: default
judge_settings:
  relevance_threshold: 2
```

## Customizing the Judge

### Modify the UMBRELA Prompt

Edit `umbrela_judge.py`, class `UmbrelaPrompt`:

```python
class UmbrelaPrompt(dspy.Signature):
    """UMBRELA prompting framework for passage grading."""

    __doc__ = dedent("""
        # Modify this prompt for your use case
        Given a query and a passage, provide a score...
    """)
```

### Add New Measures

Edit the `UMBRELA_SPEC`:

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


## Next Steps

1. **Test with your data**: Replace test data paths with your RAG responses
2. **Try different models**: Compare results across models
3. **Customize**: Modify prompt, measures, or add new variants


