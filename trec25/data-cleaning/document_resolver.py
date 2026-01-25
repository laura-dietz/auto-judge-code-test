#!/usr/bin/env python3
"""
Resolve documents for Report citations and populate Report.documents field.

Supports multiple document sources:
- pull: Fetch from remote document service (HTTP)
- ingest: Load from local corpus archive (jsonl.gz)
- export-docno: Extract doc_ids from reports (no document fetching)

Usage:
    python document_resolver.py pull --input reports_dir/ --output enriched_dir/ \\
        --collection ragtime-mt --host localhost --port 8080

    python document_resolver.py ingest --input reports_dir/ --output enriched_dir/ \\
        --corpus corpus.jsonl.gz

    python document_resolver.py export-docno --input reports_dir/ --docno-out docnos.txt
"""

import gzip
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Protocol, Set, Union

import click

from trec_auto_judge.report import (
    Report,
    extract_doc_ids_from_report,
    load_report,
    write_pydantic_json_list,
)
from trec_auto_judge.utils import format_preview
from trec_auto_judge.document.document import Document, fetch_document_service


# =============================================================================
# Protocol
# =============================================================================

class DocumentResolver(Protocol):
    """Protocol for resolving document IDs to Document objects."""

    def resolve(self, doc_ids: Set[str]) -> tuple[Dict[str, Document], Set[str]]:
        """Resolve document IDs to Documents.

        Returns:
            Tuple of (resolved documents dict, set of failed doc_ids).
        """
        ...

    def close(self) -> None:
        """Release any resources. Default is no-op."""
        ...


# =============================================================================
# Implementations
# =============================================================================

class RemoteDocumentResolver(DocumentResolver):
    """Resolve documents via remote HTTP service."""

    def __init__(
        self,
        collection_handle: str,
        host: str,
        port: int,
        max_retries: int = 5,
        rate_limit: float = 3.0,
    ):
        self.collection_handle = collection_handle
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.rate_limit = rate_limit

    def resolve(self, doc_ids: Set[str]) -> tuple[Dict[str, Document], Set[str]]:
        if not doc_ids:
            return {}, set()
        result = fetch_document_service(
            doc_ids=list(doc_ids),
            collection_handle=self.collection_handle,
            host=self.host,
            port=self.port,
            max_retries=self.max_retries,
            rate_limit=self.rate_limit,
            parallel=False,
        )
        failed = doc_ids - set(result.keys())
        return result, failed

    def close(self) -> None:
        pass


class ArchiveDocumentResolver(DocumentResolver):
    """Resolve documents from local corpus archive (jsonl.gz)."""

    def __init__(self, corpus_paths: list[Path]):
        self.corpus: Dict[str, Any] = {}
        for path in corpus_paths:
            self.corpus.update(self._load_corpus_dict(path))

    @staticmethod
    def _load_corpus_dict(source: Union[str, Path]) -> Dict[str, Any]:
        """Load corpus JSONL.gz file as a docno -> payload dictionary."""
        open_fn = gzip.open if str(source).endswith(".gz") else open
        corpus: Dict[str, Any] = {}
        with open_fn(source, mode="rt", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc["docno"]] = doc
        return corpus

    @staticmethod
    def _create_doc(doc_id: str, payload: Any) -> Document | None:
        """Convert payload to Document, handling various formats.

        Returns None if the payload cannot be converted to a valid Document.
        """
        if isinstance(payload, str):
            return Document(id=doc_id, text=payload)
        elif "id" in payload and "text" in payload:
            return Document(**{k: v for k, v in payload.items() if k in Document.model_fields})
        elif "segment" in payload:
            return Document(
                id=doc_id,
                text=payload["segment"],
                **{k: v for k, v in payload.items() if k in Document.model_fields and k not in ("id", "text")}
            )
        else:
            # No text content found
            return None

    def resolve(self, doc_ids: Set[str]) -> tuple[Dict[str, Document], Set[str]]:
        """Resolve doc_ids to Documents.

        Returns:
            Tuple of (resolved documents dict, set of failed doc_ids).
        """
        if not doc_ids:
            return {}, set()

        result: Dict[str, Document] = {}
        failed: Set[str] = set()

        for doc_id in doc_ids:
            if doc_id not in self.corpus:
                failed.add(doc_id)
                continue

            doc = self._create_doc(doc_id, self.corpus[doc_id])
            if doc is not None:
                result[doc_id] = doc
            else:
                failed.add(doc_id)

        return result, failed

    def close(self) -> None:
        self.corpus.clear()


# =============================================================================
# Shared Utilities
# =============================================================================

def populate_report_documents(report: Report, resolver: DocumentResolver, cited_only: bool = True) -> Set[str]:
    """Resolve and populate report.documents in place.

    Returns:
        Set of failed doc_ids that could not be resolved (excludes already present).
    """
    doc_ids = extract_doc_ids_from_report(report, cited_only=cited_only)
    if not doc_ids:
        return set()

    # Only resolve doc_ids not already present
    already_present = set(report.documents.keys()) if report.documents else set()
    to_resolve = doc_ids - already_present

    if not to_resolve:
        return set()

    documents, failed = resolver.resolve(to_resolve)
    if report.documents is None:
        report.documents = {}
    report.documents.update(documents)
    return failed


def process_directory(
    input_dir: Path,
    output_dir: Path,
    resolver: DocumentResolver,
    cited_only: bool = True,
    fail_fast: bool = False,
    docno_out: Path | None = None,
    skip_existing_files: bool = False,
) -> bool:
    """Process all jsonl files in input_dir, resolve documents, write to output_dir.

    Args:
        input_dir: Input directory with .jsonl report files.
        output_dir: Output directory for enriched reports.
        resolver: Document resolver to use.
        cited_only: Only resolve cited documents.
        fail_fast: Stop on first resolution failure.
        docno_out: Optional file to write failed doc_ids.
        skip_existing_files: Skip files that already exist in output directory.

    Returns:
        True if all documents resolved, False if any failures.
    """
    input_files = [f for f in input_dir.iterdir() if f.is_file()]

    if not input_files:
        print(f"No files found in {input_dir}")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(input_files)} jsonl files to process")

    all_failed: Set[str] = set()

    for input_file in input_files:
        output_file = output_dir / input_file.name
        if skip_existing_files and output_file.exists():
            print(f"\nSkipping {input_file.name} (already exists in output)")
            continue

        print(f"\nProcessing {input_file.name}...")
        reports = load_report(input_file)
        print(f"  Loaded {len(reports)} reports")

        # Track failures per file
        file_failed: Set[str] = set()
        affected_topics: Set[str] = set()

        for i, report in enumerate(reports):
            topic_id = report.metadata.topic_id
            doc_ids = extract_doc_ids_from_report(report, cited_only=cited_only)
            # print(f"  [{i+1}/{len(reports)}] topic={topic_id}: resolving {len(doc_ids)} documents")

            if doc_ids:
                failed = populate_report_documents(report, resolver, cited_only=cited_only)
                fetched_count = len(report.documents or {})
                # print(f"    Resolved {fetched_count} documents")

                if failed:
                    file_failed.update(failed)
                    affected_topics.add(topic_id)
                    all_failed.update(failed)
                    if fail_fast:
                        print("Stopping due to --fail-fast", file=sys.stderr)
                        if docno_out:
                            write_docnos(sorted(list(all_failed)), docno_out)
                            print(f"  Wrote failed docnos to {docno_out}")
                        return False

        # Print per-file summary if there were failures
        if file_failed:
            print(
                f"  FAILED {len(file_failed)} docs in topics: {format_preview(sorted(affected_topics), limit=5)}",
                file=sys.stderr
            )

        write_pydantic_json_list(reports, output_file)
        print(f"  Wrote {len(reports)} reports to {output_file}")

    # Summary
    if all_failed:
        print(
            f"\nSummary: {len(all_failed)} unique documents failed: "
            f"{format_preview(sorted(list(all_failed)), limit=5)}",
            file=sys.stderr
        )
        if docno_out:
            write_docnos(sorted(list(all_failed)), docno_out)
            print(f"  Wrote failed docnos to {docno_out}")
        return False
    else:
        print("\nAll documents resolved successfully.")
        return True


# =============================================================================
# Export-only Utilities
# =============================================================================

def write_docnos(docnos: list, output_path: Path) -> None:
    """Write docnos to file, one per line."""
    with open(output_path, "w") as f:
        for docno in docnos:
            f.write(docno + "\n")


def process_directory_check(
    input_dir: Path,
    cited_only: bool = True,
    docno_out: Path | None = None,
) -> bool:
    """Check that all reports have all required documents resolved.

    Args:
        input_dir: Directory containing enriched report jsonl files.
        cited_only: If True, only check cited documents. If False, also check references.
        docno_out: Optional path to write missing docnos to.

    Returns:
        True if all documents are present, False otherwise.
    """
    input_files = [f for f in input_dir.iterdir() if f.is_file()]

    if not input_files:
        print(f"No files found in {input_dir}")
        return True

    print(f"Found {len(input_files)} jsonl files to check")

    all_missing: Set[str] = set()
    total_reports = 0

    for input_file in input_files:
        print(f"\nChecking {input_file.name}...")
        reports = load_report(input_file)
        print(f"  Loaded {len(reports)} reports")

        for report in reports:
            total_reports += 1
            doc_ids = extract_doc_ids_from_report(report, cited_only=cited_only)
            present = set(report.documents.keys()) if report.documents else set()
            missing = doc_ids - present

            if missing:
                all_missing.update(missing)
                print(
                    f"  MISSING in topic={report.metadata.topic_id}: "
                    f"{len(missing)} docs: {format_preview(list(missing), limit=5)}",
                    file=sys.stderr
                )

    print(f"\nSummary: {total_reports} reports checked")
    if not all_missing:
        print("All documents present.")
        return True
    else:
        print(
            f"FAILED: {len(all_missing)} unique documents missing: "
            f"{format_preview(sorted(list(all_missing)), limit=5)}",
            file=sys.stderr
        )
        if docno_out:
            write_docnos(sorted(list(all_missing)), docno_out)
            print(f"  Wrote missing docnos to {docno_out}")
        return False


def process_directory_export(input_dir: Path, docno_out: Path, cited_only: bool = True) -> None:
    """Extract docnos from all jsonl files in input_dir, write sorted unique list to docno_out."""
    input_files = [f for f in input_dir.iterdir() if f.is_file()]

    if not input_files:
        print(f"No files found in {input_dir}")
        return

    print(f"Found {len(input_files)} jsonl files to process")

    collected_docnos: Set[str] = set()
    for input_file in input_files:
        print(f"\nProcessing {input_file.name}...")
        reports = load_report(input_file)
        print(f"  Loaded {len(reports)} reports")

        for report in reports:
            doc_ids = extract_doc_ids_from_report(report, cited_only=cited_only)
            collected_docnos.update(doc_ids)

    write_docnos(sorted(list(collected_docnos)), docno_out)
    print(f"  Exported {len(collected_docnos)} docnos to {docno_out}")


# =============================================================================
# CLI - Shared Option Decorators
# =============================================================================

def common_options(f: Callable) -> Callable:
    """Options shared by all commands: --input, --cited-only"""
    f = click.option(
        "--cited-only/--no-cited-only",
        default=True,
        help="Only process cited documents (default: True).",
    )(f)
    f = click.option(
        "--input", "-i",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        required=True,
        help="Input directory containing .jsonl files with Reports.",
    )(f)
    return f


def output_dir_option(f: Callable) -> Callable:
    """--output for commands that write enriched reports."""
    f = click.option(
        "--skip-existing-files",
        is_flag=True,
        default=False,
        help="Skip files that already exist in output directory.",
    )(f)
    f = click.option(
        "--output", "-o",
        type=click.Path(file_okay=False, path_type=Path),
        required=True,
        help="Output directory for enriched Reports.",
    )(f)
    return f


def docno_out_option(required: bool = False) -> Callable[[Callable], Callable]:
    """--docno-out for commands that export doc_ids."""
    def decorator(f: Callable) -> Callable:
        f = click.option(
            "--docno-out",
            type=click.Path(dir_okay=False, path_type=Path),
            required=required,
            help="Output file for doc_ids (one per line).",
        )(f)
        return f
    return decorator


# =============================================================================
# CLI - Commands
# =============================================================================

@click.group()
def cli():
    """Resolve documents for Report citations."""
    pass


@cli.command()
@common_options
@output_dir_option
@docno_out_option(required=False)
@click.option("--fail-fast", is_flag=True, default=False, help="Stop on first resolution failure.")
@click.option("--collection", "-c", required=True, help="Collection handle (e.g., ragtime-mt, msmarco2.1, rag, biogen25).")
@click.option("--host", required=True, help="Document service host.")
@click.option("--port", type=int, required=True, help="Document service port.")
@click.option("--max-retries", type=int, default=5, show_default=True, help="Max retries for failed requests.")
@click.option("--rate-limit", type=float, default=3.0, show_default=True, help="Max requests per second.")
def pull(input: Path, output: Path, skip_existing_files: bool, cited_only: bool, docno_out: Path | None, fail_fast: bool, collection: str, host: str, port: int, max_retries: int, rate_limit: float):
    """Fetch documents from remote HTTP service."""
    resolver = RemoteDocumentResolver(
        collection_handle=collection,
        host=host,
        port=port,
        max_retries=max_retries,
        rate_limit=rate_limit,
    )
    try:
        ok = process_directory(input, output, resolver, cited_only=cited_only, fail_fast=fail_fast, docno_out=docno_out, skip_existing_files=skip_existing_files)
    finally:
        resolver.close()
    if not ok:
        raise SystemExit(1)


@cli.command()
@common_options
@output_dir_option
@docno_out_option(required=False)
@click.option("--fail-fast", is_flag=True, default=False, help="Stop on first resolution failure.")
@click.option("--corpus", "-c", type=click.Path(exists=True, path_type=Path), multiple=True, required=True, help="Corpus archive file(s) (jsonl or jsonl.gz).")
def ingest(input: Path, output: Path, skip_existing_files: bool, cited_only: bool, docno_out: Path | None, fail_fast: bool, corpus: tuple[Path, ...]):
    """Enrich reports with document content from corpus archive."""
    resolver = ArchiveDocumentResolver(list(corpus))
    try:
        ok = process_directory(input, output, resolver, cited_only=cited_only, fail_fast=fail_fast, docno_out=docno_out, skip_existing_files=skip_existing_files)
    finally:
        resolver.close()
    if not ok:
        raise SystemExit(1)


@cli.command("export-docno")
@common_options
@docno_out_option(required=True)
def export_docno(input: Path, cited_only: bool, docno_out: Path):
    """Extract doc_ids from reports (sorted, deduplicated)."""
    process_directory_export(input_dir=input, docno_out=docno_out, cited_only=cited_only)


@cli.command()
@common_options
@docno_out_option(required=False)
def check(input: Path, cited_only: bool, docno_out: Path | None):
    """Verify all reports have all required documents resolved."""
    ok = process_directory_check(
        input_dir=input,
        cited_only=cited_only,
        docno_out=docno_out,
    )
    raise SystemExit(0 if ok else 1)


# =============================================================================
# Debug / Testing
# =============================================================================

# Test doc_ids for manual testing:
# doc_ids = ["msmarco_v2.1_doc_14_451442966#8_952051598", "msmarco_v2.1_doc_14_451442966"]
# doc_ids = ["msmarco_v2.1_doc_07_1499440455#17_2768952297", "msmarco_v2.1_doc_37_169878285#8_339258694"]
#
# Example usage with fetch_document_service:
# ds = fetch_document_service(
#     doc_ids=doc_ids,
#     collection_handle=collection,
#     host=host,
#     port=port,
#     rate_limit=rate_limit,
#     max_retries=max_retries,
# )
# print(ds)
#
# Example usage with corpus dict:
# for d in doc_ids:
#     print(d, corpus[d])


if __name__ == "__main__":
    cli()