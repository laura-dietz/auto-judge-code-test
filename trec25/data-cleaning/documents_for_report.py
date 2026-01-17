#!/usr/bin/env python3
"""
Fetch documents for Report citations and populate Report.documents field.

Usage:
    python documents_for_report.py --input reports_dir/ --output enriched_dir/ --collection ragtime-mt
"""

import argparse
from pathlib import Path
from typing import Dict, Set

from trec_auto_judge.report import (
    Report,
    NeuclirReportSentence,
    RagtimeReportSentence,
    Rag24ReportSentence,
    load_report,
    write_pydantic_json_list,
)
from trec_auto_judge.document.document import Document, fetch_document_service


def extract_doc_ids_from_report(report: Report) -> Set[str]:
    """Extract all unique document IDs from a Report's citations."""
    doc_ids: Set[str] = set()

    for sentence in report.responses or []:
        if sentence.citations is None:
            continue

        if isinstance(sentence, RagtimeReportSentence):
            # Dict[str, float] - keys are doc IDs
            doc_ids.update(sentence.citations.keys())

        elif isinstance(sentence, NeuclirReportSentence):
            # List[str] or List[int] - direct doc IDs
            doc_ids.update(str(c) for c in sentence.citations)

        elif isinstance(sentence, Rag24ReportSentence):
            # List[int] - indices into report.references
            if report.references:
                for idx in sentence.citations:
                    if 0 <= idx < len(report.references):
                        doc_ids.add(report.references[idx])

    # Add any references not already in doc_ids
    if report.references:
        for ref in report.references:
            ref_str = str(ref)
            if ref_str not in doc_ids:
                doc_ids.add(ref_str)

    return doc_ids


def fetch_documents_for_report(
    report: Report,
    collection_handle: str,
    **fetch_kwargs,
) -> Dict[str, Document]:
    """Fetch documents for all citations in a report.

    Returns dict of doc_id -> Document.
    """
    doc_ids = extract_doc_ids_from_report(report)

    if not doc_ids:
        return {}

    return fetch_document_service(
        doc_ids=list(doc_ids),
        collection_handle=collection_handle,
        **fetch_kwargs,
    )


def populate_report_documents(
    report: Report,
    collection_handle: str,
    **fetch_kwargs,
) -> None:
    """Fetch and populate report.documents in place."""
    report.documents = fetch_documents_for_report(
        report, collection_handle, **fetch_kwargs
    )


def process_directory(
    input_dir: Path,
    output_dir: Path,
    collection_handle: str,
    **fetch_kwargs,
) -> None:
    """Process all jsonl files in input_dir, fetch documents, write to output_dir."""
    input_files = [f for f in input_dir.iterdir() if f.is_file()]

    if not input_files:
        print(f"No files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(input_files)} jsonl files to process")

    for input_file in input_files:
        output_file = output_dir / input_file.name
        if output_file.exists():
            print(f"\nSkipping {input_file.name} (already exists in output)")
            continue

        print(f"\nProcessing {input_file.name}...")
        reports = load_report(input_file)
        print(f"  Loaded {len(reports)} reports")

        for i, report in enumerate(reports):
            topic_id = report.metadata.topic_id
            doc_ids = extract_doc_ids_from_report(report)
            print(f"  [{i+1}/{len(reports)}] topic={topic_id}: fetching {len(doc_ids)} documents")

            if doc_ids:
                populate_report_documents(report, collection_handle, **fetch_kwargs)
                fetched_count = len(report.documents or {})
                print(f"    Fetched {fetched_count} documents")
                if fetched_count < len(doc_ids):
                    missing = doc_ids - set(report.documents.keys())
                    print(f"    WARNING: Failed to fetch {len(missing)} documents: {missing}")

        write_pydantic_json_list(reports, output_file)
        print(f"  Wrote {len(reports)} reports to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch documents for Report citations"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory containing .jsonl files with Reports",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output directory for enriched Reports",
    )
    parser.add_argument(
        "--collection", "-c",
        type=str,
        required=True,
        help="Collection handle (e.g., ragtime-mt, msmarco2.1, rag, biogen25)",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Document service host",
        required=True
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True
    )   
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=3.0,
        help="Max requests per second (default: 3.0)",
    )

    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"Input path is not a directory: {args.input}")

    
    # doc_ids = ["msmarco_v2.1_doc_14_451442966#8_952051598", "msmarco_v2.1_doc_14_451442966"]
    doc_ids =["msmarco_v2.1_doc_07_1499440455#17_2768952297","msmarco_v2.1_doc_37_169878285#8_339258694"]
    # ds = fetch_document_service(doc_ids=doc_ids,
    #                        collection_handle= args.collection,
    #                         host=args.host,
    #                         port=args.port,
    #                         rate_limit=args.rate_limit,
    #                         max_retries=args.max_retries)
    # print(ds)

    process_directory(
        input_dir=args.input,
        output_dir=args.output,
        collection_handle=args.collection,
        host=args.host,
        port=args.port,
        rate_limit=args.rate_limit,
        max_retries=args.max_retries,
        parallel = False
    )


if __name__ == "__main__":
    main()