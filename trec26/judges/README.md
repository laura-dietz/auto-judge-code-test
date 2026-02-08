# TREC 2026 Judges - Quick Start Guide

Automated judging for RAG responses using LLM and non-LLM methods.

## Overview

This directory contains three judges for evaluating RAG system responses:

- **Non-LLM Judge**: Fast, deterministic scoring using BM25, keyword matching, and length metrics
- **Direct Prompt Judge**: LLM-based relevance assessment using UMBRELA (RAG/RAGTIME) or DRAGUN prompts
- **Citation Judge**: Validates citations using Auto-ARGUE attestation framework

## Prerequisites

- **Python 3.10+**
- **Install dependencies**: From project root, run:
  ```bash
  pip install -e ".[dspy]"
  ```
  *(Or use uv, poetry, pip-tools - your choice)*

- **LLM endpoint** (only for direct_prompt and citation judges):
  - Local: [Ollama](https://ollama.ai) at `http://localhost:11434/v1`
  - Cloud: Together.ai, Groq, OpenAI, or any OpenAI-compatible endpoint

## Environment Variables

For judges that use LLMs (direct_prompt, citation), set these variables:

```bash
export OPENAI_BASE_URL="your-llm-endpoint"  # Example: https://api.together.xyz/v1
export OPENAI_MODEL="your-model-name"        # Example: meta-llama/Llama-3.2-3B-Instruct-Turbo
export OPENAI_API_KEY="your-api-key"
```

Alternatively, use `--llm-config path/to/config.yml` to specify a configuration file.

## Quick Start

### Test Without LLM (Non-LLM Judge)

```bash
cd trec26/judges

python run_judge.py \
  --judge non_llm \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow non_llm_judge/workflow.yml \
  --out-dir ./output/ \
  --dataset ragtime \
  --max-topics 3 \
  --max-runs 2
```

### Test With LLM (Direct Prompt Judge)

```bash
cd trec26/judges

# Set environment variables first (see above)

python run_judge.py \
  --judge direct_prompt \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir ./output/ \
  --dataset ragtime \
  --max-topics 3 \
  --max-runs 2 \
  --name my_test
```

### Test Citation Validation

```bash
cd trec26/judges

# Set environment variables first (see above)

python run_judge.py \
  --judge citation \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow citation_judge/workflow.yml \
  --use-env-llm \
  --out-dir ./output/ \
  --dataset ragtime \
  --max-topics 3 \
  --max-runs 2 \
  --name citation_test
```

## Using run_judge.py

The `run_judge.py` script provides a unified interface for all judges with dataset filtering and debug logging.

### Key Arguments

| Argument | Description | Required |
|----------|-------------|----------|
| `--judge` | Which judge to use: `non_llm`, `direct_prompt`, or `citation` | ✓ |
| `--rag-topics` | Path to topics file (JSONL) | ✓ |
| `--rag-responses` | Path to runs directory | ✓ |
| `--workflow` | Path to workflow config file | ✓ |
| `--dataset` | Dataset type: `rag`, `ragtime`, or `dragun` | ✓ |
| `--out-dir` | Output directory for results | ✓ |
| `--llm-config` | Path to LLM config YAML (if not using `--use-env-llm`) | |
| `--use-env-llm` | Create LLM config from environment variables | |
| `--name` | Name for output files (e.g., `my_test` → `my_test.qrels`) | |
| `--max-topics` | Limit to first N topics (for testing) | |
| `--max-runs` | Limit to first N runs (for testing) | |
| `--debug-log` | Path to debug log file (JSONL format) | |

**Notes:**
- Use `--use-env-llm` OR `--llm-config`, not both
- When using `--name`, debug logs are automatically created as `{name}.jsonl` in `--out-dir`
- `--max-topics` and `--max-runs` create a filtered subset - great for fast iteration

### Dataset Types

The `--dataset` flag controls which prompt/format to use:

| Dataset | Prompt Used | Topic Format |
|---------|-------------|--------------|
| `rag` | UMBRELA | Uses `topic.title` as query |
| `ragtime` | UMBRELA | Uses `topic.title + problem_statement + background` |
| `dragun` | DRAGUN | Uses `topic.body` (news article text) |

## Debug Mode

Test judges on a small subset with detailed logging:

```bash
# Test with 2 topics, 1 run, with debug logging
python run_judge.py \
  --judge direct_prompt \
  --rag-topics path/to/topics.jsonl \
  --rag-responses path/to/runs/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir ./debug_output/ \
  --dataset ragtime \
  --max-topics 2 \
  --max-runs 1 \
  --name debug_test
```

This creates:
- `debug_output/debug_test.qrels` - Grades per response
- `debug_output/debug_test.leaderboard.txt` - Aggregated scores
- `debug_output/debug_test.judgment.json` - Full judgment data
- `debug_output/debug_test.jsonl` - Debug log (prompts, outputs, etc.)

### Understanding Debug Logs

Debug logs use JSON Lines format (one JSON object per line):

```jsonl
{"event": "session_start", "timestamp": "2026-02-08T...", "message": "..."}
{"event": "INPUT", "run_id": "run1", "topic_id": "28", "query": "...", "response": "...", "prompt": "..."}
{"event": "OUTPUT", "run_id": "run1", "topic_id": "28", "grade": 3}
```

Parse with `jq`:
```bash
# Count events by type
cat debug_test.jsonl | jq -r '.event' | sort | uniq -c

# Extract all prompts
cat debug_test.jsonl | jq -r 'select(.event=="INPUT") | .prompt'

# Filter by topic
cat debug_test.jsonl | jq 'select(.topic_id=="28")'
```

## Output Files

All judges produce standard output files in `--out-dir`:

| File | Format | Description |
|------|--------|-------------|
| `{name}.qrels` | TREC | Relevance judgments: `topic_id system doc_id grade` |
| `{name}.leaderboard.txt` | TREC | Aggregated scores per run and topic |
| `{name}.judgment.json` | JSON | Complete judgment data with metadata |
| `{name}.config.yml` | YAML | Configuration used for this run |
| `{name}.jsonl` | JSONL | Debug log (if `--name` is provided) |

## Available Judges

### Non-LLM Judge
**Fast, deterministic scoring** using BM25, keyword matching, and length analysis.

- **No LLM required** - runs in seconds
- **Metrics**: Combined score (0-3) from length, keywords, BM25, coverage
- **Use case**: Quick baseline, fast iteration, cost-free evaluation
- **Details**: See [non_llm_judge/README.md](non_llm_judge/README.md)

### Direct Prompt Judge
**LLM-based relevance assessment** using UMBRELA or DRAGUN prompts.

- **UMBRELA prompt** for RAG & RAGTIME (relevance grading 0-3)
- **DRAGUN prompt** for DRAGUN (trustworthiness assessment 0-3)
- **Metrics**: AVG_GRADE, IS_RELEVANT
- **Use case**: Human-like relevance judgment, content quality assessment
- **Details**: See [direct_prompt/README.md](direct_prompt/README.md)

### Citation Judge
**Citation validation** using Auto-ARGUE attestation framework.

- **Checks if citations exist** in documents dictionary
- **Uses LLM to verify citations support claims** (Auto-ARGUE)
- **Metrics**: CITATION_ACCURACY, CITATION_SUPPORT, AVG_CITATIONS, PERFECT_CITATIONS
- **Use case**: Verify citation quality, detect hallucinated citations
- **Details**: See [citation_judge/README.md](citation_judge/README.md)

## Examples

For comprehensive examples covering all datasets and judges, see [EXAMPLES.md](EXAMPLES.md).

### Quick Examples by Dataset

**RAGTIME:**
```bash
python run_judge.py \
  --judge direct_prompt \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir ./output/ \
  --dataset ragtime
```

**RAG:**
```bash
python run_judge.py \
  --judge direct_prompt \
  --rag-topics ../../dataset/rag-export/trec_rag_2025_queries.jsonl \
  --rag-responses ../../dataset/rag-export/runs/generation/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir ./output/ \
  --dataset rag
```

**DRAGUN:**
```bash
python run_judge.py \
  --judge direct_prompt \
  --rag-topics ../../dataset/dragun-export/trec-2025-dragun-topics.jsonl \
  --rag-responses ../../dataset/dragun-export/runs/repgen/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir ./output/ \
  --dataset dragun
```

## Tips

- **Start small**: Use `--max-topics 2 --max-runs 1` for fastest testing
- **Use cheap models**: For testing, use 3B models like `meta-llama/Llama-3.2-3B-Instruct-Turbo`
- **Name your runs**: Use `--name` to organize outputs and auto-generate debug logs
- **Check logs first**: Review debug logs before running on full dataset
- **Combine judges**: Run multiple judges on same data for comprehensive evaluation

## Combining Judges

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

## Troubleshooting

**"topic_format is required"**: Use `--dataset` flag to specify dataset type

**LLM connection errors**: Verify `OPENAI_BASE_URL` and `OPENAI_API_KEY` are set correctly

**Empty results**: Check that topic IDs in responses match topic IDs in topics file

**Slow execution**: Use `--max-topics` and `--max-runs` to test on subset first

## Next Steps

1. **Test with small data**: Use `--max-topics 2` to verify setup
2. **Review debug logs**: Check prompts and outputs make sense
3. **Scale up**: Remove limits and run on full dataset
4. **Customize**: Modify prompts, metrics, or add new variants (see judge-specific READMEs)
