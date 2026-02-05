# Running Non-LLM Judge

## Quick Start

### 1. Setup
```bash
cd trec26/judges/non_llm_judge
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_MODEL="dummy"
```

### 3. Run on RAGTIME Dataset
```bash
python ./non_llm_judge.py \
    --workflow workflow.yml \
    --rag-topics ../../../dataset/ragtime-export/RAGTIME-data/ragtime25_main_eng.jsonl \
    --rag-responses ../../../dataset/ragtime-export/runs/repgen/ \
    --out-dir ./ragtime_output/
```

Results will be saved to `ragtime_output/default.eval.txt`

### Test with Toy Data
```bash
python ./non_llm_judge.py \
    --workflow workflow.yml \
    --rag-topics ../umbrela/toy_data/topics.jsonl \
    --rag-responses ../umbrela/toy_data/runs/ \
    --out-dir ./toy_output/
```
