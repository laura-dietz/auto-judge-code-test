"""
Debug logging utility for judges.
Saves detailed information about queries, responses, prompts, and outputs in JSON format.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict


class DebugLogger:
    """Logger for detailed judge debugging in JSONL format."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = Path(log_file) if log_file else None
        self.enabled = log_file is not None

        if self.enabled:
            # Ensure log file uses .jsonl extension
            if not str(self.log_file).endswith('.jsonl'):
                self.log_file = self.log_file.with_suffix('.jsonl')

            # Write header entry
            self._write_json({
                "event": "session_start",
                "timestamp": datetime.now().isoformat(),
                "message": "Direct Prompt Judge Debug Session Started"
            })

    def _write_json(self, data: Dict):
        """Write a JSON line to the log file."""
        if not self.enabled:
            return

        with open(self.log_file, 'a') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

    def log(self, section: str, data: Any):
        """Log a section with data in JSON format."""
        if not self.enabled:
            return

        # Parse section to extract run_id and topic_id if present
        # Format: "INPUT [run_id / topic_id]" or "OUTPUT [run_id / topic_id]"
        run_id = None
        topic_id = None
        event_type = section

        if '[' in section and ']' in section:
            event_type = section.split('[')[0].strip()
            ids = section.split('[')[1].split(']')[0]
            if '/' in ids:
                run_id, topic_id = [x.strip() for x in ids.split('/')]
            else:
                topic_id = ids.strip()

        log_entry = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
        }

        if run_id:
            log_entry["run_id"] = run_id
        if topic_id:
            log_entry["topic_id"] = topic_id

        # Add data fields
        if isinstance(data, dict):
            log_entry.update(data)
        else:
            log_entry["data"] = str(data)

        self._write_json(log_entry)

    def log_query(self, topic_id: str, query: str):
        """Log extracted query."""
        self.log(f"QUERY [{topic_id}]", {
            "query": query,
            "query_length": len(query)
        })

    def log_response(self, run_id: str, topic_id: str, response: str):
        """Log response being judged."""
        self.log(f"RESPONSE [{run_id} / {topic_id}]", {
            "response": response,
            "response_length": len(response)
        })

    def log_llm_prompt(self, run_id: str, topic_id: str, prompt: dict):
        """Log LLM prompt."""
        self.log(f"PROMPT [{run_id} / {topic_id}]", {
            "prompt": prompt
        })

    def log_llm_output(self, run_id: str, topic_id: str, output: dict):
        """Log LLM output."""
        self.log(f"OUTPUT [{run_id} / {topic_id}]", {
            "output": output
        })

    def log_scores(self, run_id: str, topic_id: str, scores: dict):
        """Log computed scores."""
        self.log(f"SCORES [{run_id} / {topic_id}]", {
            "scores": scores
        })
