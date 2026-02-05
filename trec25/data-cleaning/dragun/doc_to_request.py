#!/usr/bin/env python3
"""
Throw-away script: Convert document JSON to Request format using LLM summarization.

Usage:
    python doc_to_request.py --llm-config llm-config.yml --input docs.jsonl --output requests.jsonl
"""

import json
import asyncio
from pathlib import Path
from typing import Optional
from textwrap import dedent

import dspy

from trec_auto_judge.request import Request
from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm
from minima_llm.dspy_adapter import MinimaLlmDSPyLM


# --- DSPy Signature ---

class DocumentToRequest(dspy.Signature):
    """
    Given a web document (article), derive a query that described the information need of a fact-checker. 
    
    1. Create a search query (query_title) the fact checker would search for. 
    2. Summarize the main challenges for a fact checker in the problem_statement.  Think about what information the fact checker needs from this article.
    3. Give a user_background for the fact checker.
    """

    # Inputs
    doc_title: str = dspy.InputField(desc="Title of the web document")
    doc_body: str = dspy.InputField(desc="Body text of the web document (may be truncated)")
    doc_url: str = dspy.InputField(desc="URL of the document for context")

    # Outputs
    query_title: str = dspy.OutputField(
        desc="A concise search query (5-15 words) that a user would type to find this article"
    )
    problem_statement: str = dspy.OutputField(
        desc="One or two paragraph description the article, focusing on information the fact checker would need to verify."
    )
    user_background: str = dspy.OutputField(
        desc="User_background for the fact checker"
    )


# --- Conversion Logic ---

async def convert_doc_to_request(
    doc: dict,
    predictor: dspy.Module,
    max_body_chars: int = 4000,
) -> Request:
    """Convert a single document to Request format using LLM."""

    # Truncate body if too long
    body = doc.get("body", "")[:max_body_chars]

    # Call LLM
    result = await predictor.acall(
        doc_title=doc.get("title", ""),
        doc_body=body,
        doc_url=doc.get("url", ""),
    )

    return Request(
        request_id=doc.get("docid", doc.get("url", "unknown")),
        title=result.query_title,
        problem_statement=result.problem_statement,
        background=result.user_background,
        collection_ids=["msmarco_v2.1"],
    )


async def process_documents(
    docs: list[dict],
    llm_config: MinimaLlmConfig,
) -> list[Request]:
    """Process all documents through the LLM."""

    backend = OpenAIMinimaLlm(llm_config)
    lm = MinimaLlmDSPyLM(backend)

    with dspy.context(lm=lm):
        predictor = dspy.ChainOfThought(DocumentToRequest)

        results = []
        for i, doc in enumerate(docs):
            print(f"Processing {i+1}/{len(docs)}: {doc.get('docid', 'unknown')}")
            try:
                request = await convert_doc_to_request(doc, predictor)
                results.append(request)
            except Exception as e:
                print(f"  Error: {e}")
                continue

    await backend.aclose()
    return results


# --- CLI ---

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert documents to Request format using LLM")
    # parser.add_argument("--llm-config", type=Path, required=True, help="Path to llm-config.yml")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file with documents")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file for requests")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents to process")

    args = parser.parse_args()

    # Load config
    config = MinimaLlmConfig.from_env()
    print(f"Using model: {config.model} from {config.base_url}")

    # Load documents
    docs = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))

    if args.limit:
        docs = docs[:args.limit]

    print(f"Loaded {len(docs)} documents")

    # Process
    requests = asyncio.run(process_documents(docs, config))

    # Write output
    with open(args.output, "w") as f:
        for req in requests:
            f.write(req.model_dump_json() + "\n")

    print(f"Wrote {len(requests)} requests to {args.output}")


if __name__ == "__main__":
    main()