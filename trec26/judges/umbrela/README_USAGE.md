# Running UMBRELA Judge

## Quick Start

### 1. Setup
```bash
cd trec26/judges/umbrela
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Environment Variables

Make sure you have an LLM running. Choose one:

**Option A: Ollama (Local)**
```bash
# Start ollama first: ollama run llama3.2:3b
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_MODEL="llama3.2:3b"
```

**Option B: Groq (Cloud)**
```bash
export OPENAI_BASE_URL="https://api.groq.com/openai/v1"
export OPENAI_MODEL="llama-3.2-3b-preview"
export OPENAI_API_KEY="your-groq-api-key"
```

### 3. Run on RAGTIME Dataset
```bash
python umbrela_judge.py run \
  --rag-responses ../../../dataset/ragtime-export/runs/repgen/ \
  --rag-topics ../../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
  --workflow workflow.yml \
  --llm-config llm-config.yml \
  --out-dir ragtime_output/
```

Results will be saved to `ragtime_output/default.eval.txt`

### Test with Toy Data
```bash
python umbrela_judge.py run \
  --rag-responses ./toy_data/runs/ \
  --rag-topics ./toy_data/topics.jsonl \
  --workflow workflow.yml \
  --llm-config llm-config.yml \
  --out-dir toy_output/
```

**Note:** UMBRELA judge uses an LLM, so it will be slower than the non-LLM judge. For 61 RAGTIME runs with 122 topics (~7400 responses), expect significant runtime.
