# Judge Examples - Complete Reference

> **New to judges?** Start with [README.md](README.md) for setup, overview, and quick start.

This file contains comprehensive examples for running all judges across all datasets.

## Features

- **Limit topics**: Process only first N topics (`--max-topics`)
- **Limit runs**: Process only first N runs (`--max-runs`)
- **Debug logging**: Save detailed information to JSONL log file
- **Filtered dataset**: Automatically creates filtered subset in `temp_filtered/`

## Setup

See [README.md](README.md#prerequisites) for installation and environment variable configuration.

## Usage Examples

### Test Non-LLM Judge (Quick)

```bash
cd trec26/judges

python run_judge.py \
  --judge non_llm \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow non_llm_judge/workflow.yml \
  --out-dir ./debug_output/ \
  --max-topics 3 \
  --max-runs 2 \
  --debug-log debug_non_llm.log
```

**Result**: Processes 3 topics × 2 runs = ~6 responses in seconds

### Test Direct Prompt Judge (With LLM)

**Prerequisites**: Set environment variables (see [README.md](README.md#environment-variables))

```bash
cd trec26/judges

# Use --use-env-llm to pull config from environment variables
python run_judge.py \
  --judge direct_prompt \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir ./debug_output/ \
  --max-topics 2 \
  --max-runs 1 \
  --name ragtime_debug \
  --dataset ragtime
```

**Result**: Processes 2 topics × 1 run = ~2 responses with LLM calls

**Cost**: ~$0.001 (very cheap for testing!)

**Notes**:
- The `--use-env-llm` flag creates the LLM config from environment variables instead of using `llm-config.yml`
- The `--name` flag names your output files (e.g., `ragtime_debug.qrels`) and auto-generates debug log in `--out-dir`
- The `--dataset` flag explicitly sets which prompt to use (UMBRELA for ragtime, DRAGUN for dragun)

### Test Citation Judge (With LLM - Auto-ARGUE)

**Prerequisites**: Set environment variables (see [README.md](README.md#environment-variables))

```bash
cd trec26/judges

# Use --use-env-llm to pull config from environment variables
python run_judge.py \
  --judge citation \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow citation_judge/workflow.yml \
  --use-env-llm \
  --out-dir ./debug_output/ \
  --max-topics 2 \
  --max-runs 1 \
  --name ragtime_citations \
  --dataset ragtime
```

**Result**: Validates citations for 2 topics × 1 run using Auto-ARGUE attestation prompt

**Cost**: ~$0.002 per citation (varies with number of citations)

**Notes**:
- Citation judge validates that cited documents actually support the claims
- Uses Auto-ARGUE attestation prompt for each citation
- Reports metrics: CITATION_ACCURACY, CITATION_SUPPORT, AVG_CITATIONS, PERFECT_CITATIONS

## Examples for Each Dataset

### RAGTIME (Test 5 topics, 3 runs)
```bash
# With non-LLM judge
python run_judge.py \
  --judge non_llm \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow non_llm_judge/workflow.yml \
  --out-dir ./ragtime_test/ \
  --max-topics 5 \
  --max-runs 3 \
  --name ragtime_nonllm \
  --dataset ragtime

# With direct prompt judge (uses UMBRELA prompt)
python run_judge.py \
  --judge direct_prompt \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir ./ragtime_test/ \
  --max-topics 5 \
  --max-runs 3 \
  --name ragtime_llm \
  --dataset ragtime

# With citation judge (uses Auto-ARGUE attestation)
python run_judge.py \
  --judge citation \
  --rag-topics ../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --rag-responses ../../dataset/ragtime-export/runs/repgen/ \
  --workflow citation_judge/workflow.yml \
  --use-env-llm \
  --out-dir ./ragtime_test/ \
  --max-topics 5 \
  --max-runs 3 \
  --name ragtime_citations \
  --dataset ragtime
```

### RAG (Test 5 topics, 3 runs)
```bash
# With non-LLM judge
python run_judge.py \
  --judge non_llm \
  --rag-topics ../../dataset/rag-export/trec_rag_2025_queries.jsonl \
  --rag-responses ../../dataset/rag-export/runs/generation/ \
  --workflow non_llm_judge/workflow.yml \
  --out-dir ./rag_test/ \
  --max-topics 5 \
  --max-runs 3 \
  --name rag_nonllm \
  --dataset rag

# With direct prompt judge (uses UMBRELA prompt)
python run_judge.py \
  --judge direct_prompt \
  --rag-topics ../../dataset/rag-export/trec_rag_2025_queries.jsonl \
  --rag-responses ../../dataset/rag-export/runs/generation/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir ./rag_test/ \
  --max-topics 5 \
  --max-runs 3 \
  --name rag_llm \
  --dataset rag

# With citation judge (uses Auto-ARGUE attestation)
python run_judge.py \
  --judge citation \
  --rag-topics ../../dataset/rag-export/trec_rag_2025_queries.jsonl \
  --rag-responses ../../dataset/rag-export/runs/generation/ \
  --workflow citation_judge/workflow.yml \
  --use-env-llm \
  --out-dir ./rag_test/ \
  --max-topics 5 \
  --max-runs 3 \
  --name rag_citations \
  --dataset rag
```

### DRAGUN (Test 3 topics, 2 runs)
```bash
# With non-LLM judge
python run_judge.py \
  --judge non_llm \
  --rag-topics ../../dataset/dragun-export/trec-2025-dragun-topics.jsonl \
  --rag-responses ../../dataset/dragun-export/runs/repgen/ \
  --workflow non_llm_judge/workflow.yml \
  --out-dir ./dragun_test/ \
  --max-topics 3 \
  --max-runs 2 \
  --name dragun_nonllm \
  --dataset dragun

# With direct prompt judge (uses DRAGUN prompt)
python run_judge.py \
  --judge direct_prompt \
  --rag-topics ../../dataset/dragun-export/trec-2025-dragun-topics.jsonl \
  --rag-responses ../../dataset/dragun-export/runs/repgen/ \
  --workflow direct_prompt/workflow.yml \
  --use-env-llm \
  --out-dir ./dragun_test/ \
  --max-topics 3 \
  --max-runs 2 \
  --name dragun_llm \
  --dataset dragun

# With citation judge (uses Auto-ARGUE attestation)
python run_judge.py \
  --judge citation \
  --rag-topics ../../dataset/dragun-export/trec-2025-dragun-topics.jsonl \
  --rag-responses ../../dataset/dragun-export/runs/repgen/ \
  --workflow citation_judge/workflow.yml \
  --use-env-llm \
  --out-dir ./dragun_test/ \
  --max-topics 3 \
  --max-runs 2 \
  --name dragun_citations \
  --dataset dragun
```

## Outputs

### Standard Output Directory
```
debug_output/
├── temp_filtered/           # Filtered dataset (auto-created)
│   ├── topics_filtered.jsonl
│   ├── runs_filtered/
│   └── workflow_custom.yml  # (if using --name or --dataset)
├── ragtime_test.qrels       # Qrels (grades per response) - named via --name
├── ragtime_test.leaderboard.txt  # Leaderboard results
├── ragtime_test.judgment.json    # Judgment data (JSON)
├── ragtime_test.config.yml       # Run configuration
└── ragtime_test.jsonl       # Debug log (auto-generated from --name)
```

### Debug Log File
**Format**: JSONL (JSON Lines) for easy parsing

Contains:
- Configuration used
- Topics and runs selected
- Queries extracted from topics
- Responses being judged
- Scores computed
- (For Direct Prompt Judge) LLM prompts and outputs in structured JSON format

Example JSONL entries:
```jsonl
{"event": "session_start", "timestamp": "2026-02-06T...", "message": "Direct Prompt Judge Debug Session Started"}
{"event": "INPUT", "timestamp": "2026-02-06T...", "run_id": "run1", "topic_id": "28", "query": "...", "response": "...", "prompt": "..."}
{"event": "OUTPUT", "timestamp": "2026-02-06T...", "run_id": "run1", "topic_id": "28", "output": 3, "grade": 3}
```

## Quick Validation Workflow

1. **Test with 2-3 topics/runs first** to verify everything works
2. **Check debug log** to see prompts and responses
3. **Review output leaderboard** to see scores make sense
4. **If good, scale up** to full dataset

## Arguments

- `--judge`: Which judge (`non_llm`, `direct_prompt`, or `citation`)
- `--rag-topics`: Path to topics file
- `--rag-responses`: Path to runs directory
- `--workflow`: Workflow config file
- `--llm-config`: LLM config file (direct_prompt and citation only, optional if using --use-env-llm)
- `--use-env-llm`: Create LLM config from environment variables (direct_prompt and citation only)
- `--out-dir`: Where to save results
- `--max-topics`: Limit to first N topics (optional)
- `--max-runs`: Limit to first N runs (optional)
- `--name`: Name for output files (replaces `{_name}` in workflow, auto-generates debug log as `{name}.jsonl` in `--out-dir`) (optional)
- `--dataset`: Dataset type - `rag`, `ragtime`, or `dragun` - explicitly sets which prompt to use (required)
- `--debug-log`: Save debug info to JSONL file (optional, auto-generated in `--out-dir` if --name is provided)

## Tips

- Start with `--max-topics 2 --max-runs 1` for fastest testing
- Use Direct Prompt Judge with 8B model to save costs (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo`)
- Use `--name` to organize your test runs with descriptive names (e.g., `--name ragtime_test_v1`)
- Use `--dataset` to explicitly control which prompt is used (UMBRELA for rag/ragtime, DRAGUN for dragun)
- Debug logs (JSONL format) can get large - use limits!
- The filtered dataset is saved in `{out-dir}/temp_filtered/`
- Parse debug logs with `jq` for analysis: `cat debug.jsonl | jq '.event'`
