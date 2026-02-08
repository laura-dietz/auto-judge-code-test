# Citation Judge - Auto-ARGUE Framework

A citation validation judge that evaluates whether citations actually support the claims made in RAG responses.

## What This Judge Does

**Citation-Level Validation** using Auto-ARGUE framework:

1. **Create Qrels**: Assesses each citation using Auto-ARGUE attestation prompt
   - Checks if cited document exists in documents dictionary
   - Uses LLM to verify if cited document supports the sentence claim

2. **Judge**: Aggregates citation assessments into leaderboard with measures:
   - `CITATION_ACCURACY`: Percentage of citations that exist in documents dictionary
   - `CITATION_SUPPORT`: Percentage of citations that support their sentences (via LLM)
   - `AVG_CITATIONS`: Average number of citations per response
   - `PERFECT_CITATIONS`: Boolean - true if all citations exist AND support claims

**Note**: This judge focuses exclusively on citation validation. For content relevance, use `direct_prompt` judge.

## Files

```
citation_judge/
├── citation_judge.py    # Judge implementation
├── workflow.yml         # Configuration
└── README.md           # This file
```

## How It Works

### 1. Citation Extraction

For each response, the judge:
- Parses the `responses` array (each segment has `text` and `citations`)
- Extracts citation IDs from each sentence
- Checks if citation exists in `documents` dictionary

### 2. Citation Attestation (Auto-ARGUE)

For each existing citation, the judge uses the Auto-ARGUE attestation prompt:

```
Sentence: [sentence with citation]
Document: [cited document text]
Answer (YES/NO): [Does document support sentence?]
```

### 3. Metrics Calculation

- **CITATION_ACCURACY** = (citations that exist) / (total citations)
- **CITATION_SUPPORT** = (citations that support) / (total citations)
- **AVG_CITATIONS** = total citations in response
- **PERFECT_CITATIONS** = all citations exist AND all support (100% on both metrics)

## Running the Judge

### Using run_judge.py (Recommended)

```bash
cd trec26/judges

# Set environment variables for your LLM provider
export OPENAI_BASE_URL="https://api.together.xyz/v1"
export OPENAI_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
export OPENAI_API_KEY="your-key"

# Run citation judge
python run_judge.py \
  --judge citation \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow citation_judge/workflow.yml \
  --use-env-llm \
  --out-dir ./output/ \
  --dataset ragtime \
  --name ragtime_citations
```

### Test with Limited Data

```bash
# Test with 2 topics, 1 run
python run_judge.py \
  --judge citation \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow citation_judge/workflow.yml \
  --use-env-llm \
  --out-dir ./citation_test/ \
  --dataset ragtime \
  --max-topics 2 \
  --max-runs 1 \
  --name test_citations
```

## Output Formats

### Qrels (default.qrels)

TREC format: `topic_id system doc_id grade`

```
28 system abc123... 1.0   # Citation exists and supports
28 system def456... 0.0   # Citation doesn't support
29 system ghi789... 0.0   # Citation doesn't exist
```

### Leaderboard (default.leaderboard.txt)

```
run_01 28 CITATION_ACCURACY 0.92
run_01 28 CITATION_SUPPORT 0.78
run_01 28 AVG_CITATIONS 5.0
run_01 28 PERFECT_CITATIONS 0.0
run_01 all CITATION_ACCURACY 0.85
run_01 all CITATION_SUPPORT 0.72
run_01 all AVG_CITATIONS 4.5
run_01 all PERFECT_CITATIONS 0.0
```

### Debug Logs (when using --name or --debug-log)

JSON Lines format with structured logging:

```jsonl
{"event": "session_start", "timestamp": "2026-02-06T...", "message": "Citation Judge Debug Session Started"}
{"event": "CITATION_INPUT", "timestamp": "2026-02-06T...", "run_id": "run1", "topic_id": "28", "citation_id": "doc123", "citation_exists": true}
{"event": "CITATION_OUTPUT", "timestamp": "2026-02-06T...", "run_id": "run1", "topic_id": "28", "citation_supports": true}
```

## Supported Datasets

Works with all three datasets:

| Dataset | Citation Structure |
|---------|-------------------|
| **RAG** | `responses` array with `citations` + `documents` dictionary |
| **RAGTIME** | `responses` array with `citations` + `documents` dictionary |
| **DRAGUN** | `responses` array with `citations` + `documents` dictionary |

All datasets use the same citation format, so the judge works identically across them.

## Combining with Content Relevance

To get both citation AND content evaluation:

### Option 1: Run Separately

```bash
# Run content relevance judge
python run_judge.py \
  --judge direct_prompt \
  --rag-topics topics.jsonl \
  --rag-responses runs/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir output/ \
  --dataset ragtime \
  --name ragtime_content

# Run citation judge
python run_judge.py \
  --judge citation \
  --rag-topics topics.jsonl \
  --rag-responses runs/ \
  --workflow citation_judge/workflow.yml \
  --use-env-llm \
  --out-dir output/ \
  --dataset ragtime \
  --name ragtime_citations
```

Then merge the leaderboards manually or analyze separately.

### Option 2: Combine Metrics (Future)

Create a combined judge that runs both and reports all metrics together.

## Example Workflow

```bash
# 1. Test with small data first
python run_judge.py \
  --judge citation \
  --rag-topics topics.jsonl \
  --rag-responses runs/ \
  --workflow citation_judge/workflow.yml \
  --use-env-llm \
  --out-dir test_output/ \
  --dataset ragtime \
  --max-topics 2 \
  --max-runs 1 \
  --name test

# 2. Check debug log to verify citations are being validated
cat test_output/test.log | jq '.event' | sort | uniq -c

# 3. Review leaderboard to see metrics
cat test_output/test.leaderboard.txt

# 4. If looks good, scale up to full dataset
python run_judge.py \
  --judge citation \
  --rag-topics topics.jsonl \
  --rag-responses runs/ \
  --workflow citation_judge/workflow.yml \
  --use-env-llm \
  --out-dir output/ \
  --dataset ragtime \
  --name full_citations
```

## Customizing

### Modify Attestation Prompt

Edit [citation_judge.py](citation_judge.py:119) to change the Auto-ARGUE prompt:

```python
class AttestationPrompt(dspy.Signature):
    """Auto-ARGUE attestation prompt for checking if sentence is supported by document."""

    __doc__ = dedent("""
        # Modify the prompt here
        Your custom instructions...
    """)
```

### Add New Metrics

Edit the spec in [citation_judge.py](citation_judge.py:157):

```python
CITATION_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("CITATION_ACCURACY", aggregate=mean_of_floats, cast=float, default=0.0),
    MeasureSpec("CITATION_SUPPORT", aggregate=mean_of_floats, cast=float, default=0.0),
    MeasureSpec("AVG_CITATIONS", aggregate=mean_of_floats, cast=float, default=0.0),
    MeasureSpec("PERFECT_CITATIONS", aggregate=mean_of_bools, cast=bool, default=False),
    # Add your new measure:
    MeasureSpec("CITATION_PRECISION", aggregate=mean_of_floats, cast=float, default=0.0),
))
```

## Limitations

- **Sentence splitting**: Uses simple regex-based sentence splitting. For better accuracy, could integrate spacy or nltk.
- **LLM dependency**: Requires LLM for attestation checks - costs scale with number of citations.
- **No composite scoring**: Currently reports citation metrics separately from content relevance. Manual combination needed.

## Next Steps

1. **Test with your data**: Run on a small subset first
2. **Compare with direct_prompt**: Run both judges to see content vs. citation quality
3. **Analyze patterns**: Use debug logs to identify common citation errors
4. **Tune thresholds**: Decide what citation accuracy/support levels are acceptable for your use case
