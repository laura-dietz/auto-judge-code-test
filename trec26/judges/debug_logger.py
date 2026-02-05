"""
Debug logging utility for judges.
Saves detailed information about queries, responses, prompts, and outputs.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Any


class DebugLogger:
    """Logger for detailed judge debugging."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = Path(log_file) if log_file else None
        self.enabled = log_file is not None

        if self.enabled:
            # Create log file with header
            with open(self.log_file, 'w') as f:
                f.write(f"=== Judge Debug Log ===\n")
                f.write(f"Started: {datetime.now()}\n")
                f.write("=" * 80 + "\n\n")

    def log(self, section: str, data: Any):
        """Log a section with data."""
        if not self.enabled:
            return

        with open(self.log_file, 'a') as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"{section}\n")
            f.write(f"{'=' * 80}\n")

            if isinstance(data, dict):
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")
            elif isinstance(data, str):
                f.write(data + "\n")
            else:
                f.write(str(data) + "\n")

            f.write("\n")

    def log_query(self, topic_id: str, query: str):
        """Log extracted query."""
        self.log(f"QUERY [{topic_id}]", {
            "topic_id": topic_id,
            "query_text": query,
            "query_length": len(query)
        })

    def log_response(self, run_id: str, topic_id: str, response: str):
        """Log response being judged."""
        self.log(f"RESPONSE [{run_id} / {topic_id}]", {
            "run_id": run_id,
            "topic_id": topic_id,
            "response_text": response[:500] + "..." if len(response) > 500 else response,
            "response_length": len(response)
        })

    def log_llm_prompt(self, run_id: str, topic_id: str, prompt: dict):
        """Log LLM prompt."""
        self.log(f"LLM PROMPT [{run_id} / {topic_id}]", prompt)

    def log_llm_output(self, run_id: str, topic_id: str, output: dict):
        """Log LLM output."""
        self.log(f"LLM OUTPUT [{run_id} / {topic_id}]", output)

    def log_scores(self, run_id: str, topic_id: str, scores: dict):
        """Log computed scores."""
        self.log(f"SCORES [{run_id} / {topic_id}]", scores)
