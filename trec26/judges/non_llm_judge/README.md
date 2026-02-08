# Non-LLM Judge

> **Setup and usage**: See [main judges README](../README.md) for installation and running instructions.

A deterministic judge that scores RAG responses using multiple non-LLM metrics.

## Overview

This judge evaluates responses based on:
- **Length-based scoring**: Evaluates response completeness based on word count
- **Keyword matching**: Measures presence of query terms in the response
- **BM25 scoring**: Traditional IR ranking metric
- **Query coverage**: Percentage of query words found in the response

All metrics are combined with configurable weights to produce a final grade on a 0-3 scale.


## Metrics

### 1. Length Score (0-1)
Evaluates response length with optimal range preferences:
- Below `min_length` (default 50 words): Penalized for being too short
- Between `min_length` and `optimal_length` (default 200 words): Linear increase
- Between `optimal_length` and `max_length` (default 500 words): Slight penalty
- Above `max_length`: Penalized for being too verbose

### 2. Keyword Score (0-1)
Percentage of unique query terms found in the response.
- Example: Query "What is quantum computing?" -> 4 terms
- If response contains ["what", "quantum", "computing"] -> 3/4 = 0.75

### 3. BM25 Score (0-1)
Simplified BM25 ranking score (normalized):
- Uses standard BM25 parameters: k1=1.5, b=0.75
- Considers term frequency in response
- Normalized to 0-1 range for combination

### 4. Coverage Score (0-1)
Similar to keyword score but considers query term frequency:
- Counts how many query term occurrences are covered
- Example: Query "what is is" has 3 tokens
- If response has "is" once → 1/3 = 0.33

### Combined Score (0-3)
All metrics are weighted and combined:
```
combined = (
    length_score * length_weight +
    keyword_score * keyword_weight +
    bm25_score * bm25_weight +
    coverage_score * coverage_weight
)
final_grade = combined * 3.0  # Scale to 0-3 range
```

Default weights (configurable):
- Length: 20%
- Keyword: 30%
- BM25: 30%
- Coverage: 20%

## Quick Start

```bash
cd trec26/judges

# Run non-LLM judge (no LLM or environment variables needed)
python run_judge.py \
  --judge non_llm \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow non_llm_judge/workflow.yml \
  --out-dir ./output/ \
  --dataset ragtime

# Test with limited data
python run_judge.py \
  --judge non_llm \
  --rag-topics path/to/topics.jsonl \
  --rag-responses path/to/runs/ \
  --workflow non_llm_judge/workflow.yml \
  --out-dir ./output/ \
  --dataset ragtime \
  --max-topics 3 \
  --max-runs 2
```

See [EXAMPLES.md](../EXAMPLES.md) and [main README](../README.md) for more examples.

### Using the CLI Directly





```bash
python ./non_llm_judge.py \
    --workflow workflow.yml \
    --rag-topics ../umbrela/toy_data/topics.jsonl \
    --rag-responses ../umbrela/toy_data/runs/ \
    --out-dir ./toy_output/ 
```

## Configuration

Edit `workflow.yml` to adjust settings:

```yaml
judge_settings:
  # Metric weights (should sum to 1.0)
  length_weight: 0.2
  keyword_weight: 0.3
  bm25_weight: 0.3
  coverage_weight: 0.2

  # Length thresholds (in words)
  min_length: 50
  optimal_length: 200
  max_length: 500
```

## Output

The judge produces a leaderboard with the following measures:

- **AVG_GRADE**: Overall grade (0-3 scale)
<!-- - **LENGTH_SCORE**: Average length score
- **KEYWORD_SCORE**: Average keyword matching score
- **BM25_SCORE**: Average BM25 score
- **COVERAGE_SCORE**: Average query coverage score -->

Each measure is reported per topic and as an aggregate ("all" topics).

## Example Results

```
run_high_quality    AVG_GRADE  all  0.44
run_medium_quality  AVG_GRADE  all  0.50
run_low_quality     AVG_GRADE  all  0.00
```


## Files

- [`non_llm_judge.py`](non_llm_judge.py): Main judge implementation
- [`workflow.yml`](workflow.yml): Configuration and variants
- [`README.md`](README.md): This file
