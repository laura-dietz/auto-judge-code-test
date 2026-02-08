# Citation Judge - Auto-ARGUE Framework

> **Setup and usage**: See [main judges README](../README.md) for installation, environment setup, and running instructions.

Citation validation judge that evaluates whether citations actually support the claims made in RAG responses.

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

**Note**: This judge focuses exclusively on citation validation. For content relevance, use the [direct prompt judge](../direct_prompt/README.md).

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

The LLM evaluates whether the document content actually supports/entails the claim made in the sentence.

### 3. Metrics Calculation

- **CITATION_ACCURACY** = (citations that exist) / (total citations)
- **CITATION_SUPPORT** = (citations that support) / (total citations)
- **AVG_CITATIONS** = total citations in response
- **PERFECT_CITATIONS** = all citations exist AND all support (100% on both metrics)

## Quick Start

```bash
cd trec26/judges

# Set environment variables (see main README)
export OPENAI_BASE_URL="your-endpoint"
export OPENAI_MODEL="your-model"
export OPENAI_API_KEY="your-key"

# Run citation judge on RAGTIME
python run_judge.py \
  --judge citation \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow citation_judge/workflow.yml \
  --use-env-llm \
  --out-dir ./output/ \
  --dataset ragtime

# Test with limited data
python run_judge.py \
  --judge citation \
  --rag-topics path/to/topics.jsonl \
  --rag-responses path/to/runs/ \
  --workflow citation_judge/workflow.yml \
  --use-env-llm \
  --out-dir ./output/ \
  --dataset ragtime \
  --max-topics 2 \
  --max-runs 1 \
  --name test
```

See [EXAMPLES.md](../EXAMPLES.md) for more examples across all datasets.

## Supported Datasets

Works with all three datasets - they all use the same citation format:

| Dataset | Citation Structure |
|---------|-------------------|
| **RAG** | `responses` array with `citations` + `documents` dictionary |
| **RAGTIME** | `responses` array with `citations` + `documents` dictionary |
| **DRAGUN** | `responses` array with `citations` + `documents` dictionary |

The judge handles both citation formats:
- **List format**: `citations: ["doc1", "doc2"]` (RAG/DRAGUN)
- **Dict format**: `citations: {"doc1": 0.9, "doc2": 0.8}` (RAGTIME)

## Output Format

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

### Debug Logs (when using --name)
JSON Lines format with structured logging:

```jsonl
{"event": "CITATION_INPUT", "run_id": "run1", "topic_id": "28", "citation_id": "doc123", "citation_exists": true, "prompt_template": "..."}
{"event": "CITATION_OUTPUT", "run_id": "run1", "topic_id": "28", "citation_supports": true}
```

Debug logs include:
- Full Auto-ARGUE prompts sent to LLM
- Document text for each citation
- LLM responses (YES/NO)
- Citation metrics

## Customization

### Modify Attestation Prompt

Edit [citation_judge.py](citation_judge.py) to change the Auto-ARGUE prompt:

```python
class AttestationPrompt(dspy.Signature):
    """Auto-ARGUE attestation prompt for checking if sentence is supported by document."""

    __doc__ = dedent("""
        # Modify the prompt here
        You are an expert at determining if statements are supported by a document.
        Your task is to determine if a sentence's claims are supported by a provided document.
        A sentence is supported by a document if and only if is entailed by the document.
        Respond with ONLY 'YES' or 'NO' in English.

        # Add your custom instructions...
    """)
```

### Add New Metrics

Edit the spec in [citation_judge.py](citation_judge.py):

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

Then update the `judge()` method to compute and add the new metric.

## Combining with Direct Prompt Judge

To evaluate both content relevance AND citation quality:

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

# 3. Analyze both leaderboards
cat output/content.leaderboard.txt
cat output/citations.leaderboard.txt
```

This gives you:
- **Content quality** (AVG_GRADE, IS_RELEVANT) from direct_prompt
- **Citation quality** (CITATION_ACCURACY, CITATION_SUPPORT) from citation judge

## Limitations

- **Sentence splitting**: Uses simple regex-based sentence splitting. For better accuracy, could integrate spacy or nltk.
- **LLM dependency**: Requires LLM for attestation checks - costs scale with number of citations.
- **No composite scoring**: Currently reports citation metrics separately from content relevance. Manual combination needed.

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
cat test_output/test.jsonl | jq '.event' | sort | uniq -c

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

## Files

```
citation_judge/
├── citation_judge.py    # Judge implementation
├── workflow.yml         # Configuration
└── README.md           # This file
```
