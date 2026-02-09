"""
Shared utilities for rubric/nugget-based AutoJudge implementations.

Provides:
- NuggetGradeData: Data model for nugget grading
- GradeNuggetAnswer: DSPy signature for grading passages against questions
- Grade aggregation and coverage computation
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import dspy
from pydantic import BaseModel

from autojudge_base import Report
from autojudge_base.nugget_data import (
    NuggetBank,
    NuggetBanks,
    NuggetQuestion,
    Reference,
)


# =============================================================================
# Data Models
# =============================================================================


class NuggetGradeData(BaseModel):
    """Combined input/output for grading a nugget against a passage."""

    # Input fields
    run_id: str
    query_id: str
    nugget_id: str
    question: str
    passage: str
    # Optional: document/paragraph identifiers
    doc_id: Optional[str] = None
    paragraph_idx: Optional[int] = None
    # Output fields (populated by LLM)
    grade: int = 0
    reasoning: Optional[str] = None
    confidence: Optional[float] = None


# =============================================================================
# DSPy Signature
# =============================================================================


def _parse_grade(s: str) -> int:
    """Extract grade 0-5 from string."""
    m = re.search(r"\b([0-5])\b", s)
    if not m:
        return 0  # Default to 0 if no valid grade found
    return int(m.group(1))


class GradeNuggetAnswer(dspy.Signature):
    __doc__ = dedent(
        """
        Grade how well a passage answers a specific question.

        Can the question be answered based on the available context? Choose one:
        - 5: The answer is highly relevant, complete, and accurate.
        - 4: The answer is mostly relevant and complete but may have minor gaps or inaccuracies.
        - 3: The answer is partially relevant and complete, with noticeable gaps or inaccuracies.
        - 2: The answer has limited relevance and completeness, with significant gaps or inaccuracies.
        - 1: The answer is minimally relevant or complete, with substantial shortcomings.
        - 0: The answer is not relevant or complete at all.
        """
    )

    question: str = dspy.InputField(desc="The question to be answered")
    passage: str = dspy.InputField(desc="The passage that may contain the answer")

    grade: Literal["0", "1", "2", "3", "4", "5"] = dspy.OutputField(
        desc="Grade from 0-5 indicating how well the passage answers the question"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of the grade"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score from 0.0 to 1.0 indicating how certain you are"
    )

    @classmethod
    def convert_prompt_output(
        cls, prediction: dspy.Prediction, data: NuggetGradeData
    ) -> None:
        """Convert DSPy Prediction output to NuggetGradeData."""
        data.grade = _parse_grade(prediction.grade)
        data.reasoning = getattr(prediction, "reasoning", None)
        data.confidence = getattr(prediction, "confidence", None)


# =============================================================================
# Grade Data Preparation
# =============================================================================


def prepare_nugget_grade_data(
    rag_responses: Sequence[Report],
    nugget_banks: NuggetBanks,
) -> tuple[List[NuggetGradeData], Dict[str, int]]:
    """
    Prepare grading data for all response-nugget pairs.

    Args:
        rag_responses: RAG responses to grade
        nugget_banks: Nugget banks containing questions per topic

    Returns:
        Tuple of (grade_data list, nuggets_per_topic dict)
    """
    # Pre-compute nugget counts per topic from the bank
    nuggets_per_topic: Dict[str, int] = {
        topic_id: len(bank.nuggets_as_list())
        for topic_id, bank in nugget_banks.banks.items()
    }

    grade_data: List[NuggetGradeData] = []

    for response in rag_responses:
        metadata = response.metadata
        run_id = metadata.run_id
        topic_id = metadata.topic_id
        text = response.get_report_text()

        bank = nugget_banks.banks.get(topic_id)
        if bank is None:
            print(f"Warning: No nugget bank for topic {topic_id}, skipping")
            continue

        # Create grade data for each nugget question
        for nugget in bank.nuggets_as_list():
            if isinstance(nugget, NuggetQuestion):
                data = NuggetGradeData(
                    run_id=run_id,
                    query_id=topic_id,
                    nugget_id=nugget.question_id or nugget.question,
                    question=nugget.question,
                    passage=text,
                )
                grade_data.append(data)

    return grade_data, nuggets_per_topic


def prepare_nugget_grade_data_for_documents(
    rag_responses: Sequence[Report],
    nugget_banks: NuggetBanks,
    use_paragraphs: bool = False,
    document_ids: Optional[Set[str]] = None,
) -> tuple[List[NuggetGradeData], Dict[str, int]]:
    """
    Prepare grading data for document-nugget pairs.

    Instead of grading the full response text, this creates grade data for
    each document in response.documents, optionally at the paragraph level.

    Args:
        rag_responses: RAG responses containing documents to grade
        nugget_banks: Nugget banks containing questions per topic
        use_paragraphs: If True, create separate entries for each paragraph
                        in each document. If False, use full document text.
        document_ids: Optional set of document IDs to process. If None,
                      all documents are processed.

    Returns:
        Tuple of (grade_data list, nuggets_per_topic dict)
    """
    # Pre-compute nugget counts per topic from the bank
    nuggets_per_topic: Dict[str, int] = {
        topic_id: len(bank.nuggets_as_list())
        for topic_id, bank in nugget_banks.banks.items()
    }

    grade_data: List[NuggetGradeData] = []
    doc_counter = 0
    unique_doc_ids:Set[str] = set()
    unique_text:Set[str] = set()

    for response in rag_responses:
        metadata = response.metadata
        run_id = metadata.run_id
        topic_id = metadata.topic_id

        bank = nugget_banks.banks.get(topic_id)
        if bank is None:
            print(f"Warning: No nugget bank for topic {topic_id}, skipping")
            continue

        if not response.documents:
            print(f"Warning: No documents for topic {topic_id}, run {run_id}, skipping")
            continue

        # Iterate over documents
        for doc_id, document in response.documents.items():
            # Skip if document_ids filter is set and this doc is not in it
            if document_ids is not None and doc_id not in document_ids:
                continue

            unique_doc_ids.add(doc_id)
            if use_paragraphs:
                # Create grade data for each paragraph
                paragraphs = document.get_paragraphs()
                for para_idx, paragraph in enumerate(paragraphs):
                    if not paragraph.strip():
                        continue
                    doc_counter+=1
                    unique_text.add(paragraph.strip())
                    for nugget in bank.nuggets_as_list():
                        if isinstance(nugget, NuggetQuestion):
                            data = NuggetGradeData(
                                run_id=run_id,
                                query_id=topic_id,
                                nugget_id=nugget.question_id or nugget.question,
                                question=nugget.question,
                                passage=paragraph.strip(),
                                doc_id=doc_id,
                                paragraph_idx=para_idx,
                            )
                            grade_data.append(data)
            else:
                # Use full document text
                text = document.get_text()
                doc_counter+=1
                unique_text.add(text.strip())
                for nugget in bank.nuggets_as_list():
                    if isinstance(nugget, NuggetQuestion):
                        data = NuggetGradeData(
                            run_id=run_id,
                            query_id=topic_id,
                            nugget_id=nugget.question_id or nugget.question,
                            question=nugget.question,
                            passage=text.strip(),
                            doc_id=doc_id,
                        )
                        grade_data.append(data)

    print(f"Grade data for {len(unique_doc_ids)} unique doc_ids, {len(unique_text)} unique texts and {doc_counter} many text chunks.")
    return grade_data, nuggets_per_topic


# =============================================================================
# Grade Aggregation
# =============================================================================


class NuggetAggregateResult(BaseModel):
    """Aggregated nugget grading results for a single (run_id, topic_id) pair."""

    run_id: str
    topic_id: str
    coverage_score: float
    avg_grade: float
    max_grade: int
    covered_count: int
    total_nuggets: int
    graded_nuggets: int
    nugget_grades: Dict[str, Dict[str, Any]]  # nugget_id -> {grade, reasoning}


def compute_nugget_aggregates(
    grade_data: List[NuggetGradeData],
    nuggets_per_topic: Dict[str, int],
    grade_threshold: int = 3,
) -> Dict[str, NuggetAggregateResult]:
    """
    Compute coverage aggregates from nugget grading data.

    Args:
        grade_data: List of graded nugget-response pairs
        nuggets_per_topic: Total nugget count per topic from the bank
        grade_threshold: Minimum grade to count as "covered"

    Returns:
        Dict mapping "run_id:topic_id" -> NuggetAggregateResult
    """
    # Group grades by response
    response_data: Dict[str, Dict[str, Any]] = {}

    for data in grade_data:
        response_key = f"{data.run_id}:{data.query_id}"
        if response_key not in response_data:
            response_data[response_key] = {
                "run_id": data.run_id,
                "topic_id": data.query_id,
                "nugget_grades": {},
                "grades_list": [],
            }

        response_data[response_key]["nugget_grades"][data.nugget_id] = { # replace with higher grade
            "grade": data.grade,
            "reasoning": data.reasoning,
        }
        response_data[response_key]["grades_list"].append(data.grade) # keep document_id information

    # Compute aggregates using total nuggets in bank as denominator
    aggregates: Dict[str, NuggetAggregateResult] = {}

    for response_key, rd in response_data.items():
        topic_id = rd["topic_id"]
        total_in_bank = nuggets_per_topic.get(topic_id, 0)
        grades = rd["grades_list"]

        if total_in_bank > 0:
            covered = sum(1 for g in grades if g >= grade_threshold)
            aggregates[response_key] = NuggetAggregateResult(
                run_id=rd["run_id"],
                topic_id=topic_id,
                coverage_score=covered / total_in_bank,
                avg_grade=sum(grades) / total_in_bank if grades else 0.0,
                max_grade=max(grades) if grades else 0,
                covered_count=covered,
                total_nuggets=total_in_bank,
                graded_nuggets=len(grades),
                nugget_grades=rd["nugget_grades"],
            )
        else:
            aggregates[response_key] = NuggetAggregateResult(
                run_id=rd["run_id"],
                topic_id=topic_id,
                coverage_score=0.0,
                avg_grade=0.0,
                max_grade=0,
                covered_count=0,
                total_nuggets=0,
                graded_nuggets=0,
                nugget_grades=rd["nugget_grades"],
            )

    return aggregates


def compute_nugget_aggregates_for_documents(
    grade_data: List[NuggetGradeData],
    nuggets_per_topic: Dict[str, int],
    grade_threshold: int = 3,
) -> Dict[str, NuggetAggregateResult]:
    """
    Compute coverage aggregates from document-level nugget grading data.

    For each (run_id, topic_id, nugget_id), takes the MAX grade across all
    documents/paragraphs, then computes aggregates from those max grades.

    Args:
        grade_data: List of graded nugget-document pairs (with doc_id set)
        nuggets_per_topic: Total nugget count per topic from the bank
        grade_threshold: Minimum grade to count as "covered"

    Returns:
        Dict mapping "run_id:topic_id" -> NuggetAggregateResult
    """
    # Step 1: Group by (run_id, topic_id, nugget_id) and take max grade
    # Key: (run_id, topic_id, nugget_id) -> {grade, reasoning, doc_id}
    max_grades: Dict[tuple[str, str, str], Dict[str, Any]] = {}

    for data in grade_data:
        key = (data.run_id, data.query_id, data.nugget_id)
        if key not in max_grades or data.grade > max_grades[key]["grade"]:
            max_grades[key] = {
                "grade": data.grade,
                "reasoning": data.reasoning,
                "doc_id": data.doc_id,
                "paragraph_idx": data.paragraph_idx,
            }

    # Step 2: Group max grades by response (run_id, topic_id)
    response_data: Dict[str, Dict[str, Any]] = {}

    for (run_id, topic_id, nugget_id), grade_info in max_grades.items():
        response_key = f"{run_id}:{topic_id}"
        if response_key not in response_data:
            response_data[response_key] = {
                "run_id": run_id,
                "topic_id": topic_id,
                "nugget_grades": {},
                "grades_list": [],
            }

        response_data[response_key]["nugget_grades"][nugget_id] = {
            "grade": grade_info["grade"],
            "reasoning": grade_info["reasoning"],
            "doc_id": grade_info["doc_id"],
            "paragraph_idx": grade_info["paragraph_idx"],
        }
        response_data[response_key]["grades_list"].append(grade_info["grade"])

    # Step 3: Compute aggregates using total nuggets in bank as denominator
    aggregates: Dict[str, NuggetAggregateResult] = {}

    for response_key, rd in response_data.items():
        topic_id = rd["topic_id"]
        total_in_bank = nuggets_per_topic.get(topic_id, 0)
        grades = rd["grades_list"]

        if total_in_bank > 0:
            covered = sum(1 for g in grades if g >= grade_threshold)
            aggregates[response_key] = NuggetAggregateResult(
                run_id=rd["run_id"],
                topic_id=topic_id,
                coverage_score=covered / total_in_bank,
                avg_grade=sum(grades) / total_in_bank if grades else 0.0,
                max_grade=max(grades) if grades else 0,
                covered_count=covered,
                total_nuggets=total_in_bank,
                graded_nuggets=len(grades),
                nugget_grades=rd["nugget_grades"],
            )
        else:
            aggregates[response_key] = NuggetAggregateResult(
                run_id=rd["run_id"],
                topic_id=topic_id,
                coverage_score=0.0,
                avg_grade=0.0,
                max_grade=0,
                covered_count=0,
                total_nuggets=0,
                graded_nuggets=0,
                nugget_grades=rd["nugget_grades"],
            )

    return aggregates


# =============================================================================
# Nugget Bank Construction
# =============================================================================


def build_nugget_banks(
    questions_by_topic: Dict[str, Tuple[str, List[str]]],
    max_per_topic: Optional[int] = None,
) -> NuggetBanks:
    """Build NuggetBanks with normalization, deduplication, and limits.

    Args:
        questions_by_topic: topic_id -> (title_query, list of question strings)
        max_per_topic: Optional limit on nuggets per topic

    Returns:
        NuggetBanks with deduplicated, normalized questions per topic.
        question_id is auto-generated as MD5 hash of the question text.
    """
    banks = []
    total = 0

    for topic_id, (title, questions) in questions_by_topic.items():
        bank = NuggetBank(query_id=topic_id, title_query=title)
        seen: Set[str] = set()
        nuggets = []

        for q in questions:
            if max_per_topic and len(nuggets) >= max_per_topic:
                break
            normalized = q.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            nuggets.append(NuggetQuestion(query_id=topic_id, question=normalized))

        bank.add_nuggets(nuggets)
        banks.append(bank)
        total += len(nuggets)

    print(f"Created {total} nuggets across {len(banks)} topics")
    return NuggetBanks.from_banks_list(banks)


# =============================================================================
# Nugget-Relevant Document Export
# =============================================================================


# Re-export from autojudge_base for backwards compatibility
from autojudge_base.nugget_doc_models import NuggetDocEntry, TopicNuggetDocs, write_nugget_docs_collaborator  # noqa: F401


def collect_nugget_relevant_docs(
    grade_data: List[NuggetGradeData],
    grade_threshold: int = 3,
) -> Dict[str, TopicNuggetDocs]:
    """Collect nugget-relevant documents from document-level grading data.

    Groups by (query_id, question), collects doc_ids where grade >= threshold.
    Run-agnostic: aggregates across all runs.

    Args:
        grade_data: List of graded nugget-document pairs (with doc_id set)
        grade_threshold: Minimum grade to count as "covered"

    Returns:
        Dict mapping topic_id -> TopicNuggetDocs
    """
    # Group by (query_id, question) -> set of doc_ids meeting threshold
    relevant: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for data in grade_data:
        if data.doc_id is None:
            continue
        if data.grade >= grade_threshold:
            relevant[(data.query_id, data.question)].add(data.doc_id)

    # Build TopicNuggetDocs per topic
    topic_entries: Dict[str, List[NuggetDocEntry]] = defaultdict(list)
    for (query_id, question), doc_ids in sorted(relevant.items()):
        topic_entries[query_id].append(
            NuggetDocEntry(question=question, doc_ids=sorted(doc_ids))
        )

    return {
        topic_id: TopicNuggetDocs(topic_id=topic_id, entries=entries)
        for topic_id, entries in topic_entries.items()
    }


def nugget_docs_to_nugget_banks(
    topics: Dict[str, TopicNuggetDocs],
) -> NuggetBanks:
    """Convert nugget-doc references to NuggetBanks format.

    Each NuggetDocEntry becomes a NuggetQuestion with:
    - references: list of Reference(doc_id=...) for each relevant doc
    - answers: None (open-ended question)
    """
    banks = []
    for topic_id, topic in topics.items():
        bank = NuggetBank(query_id=topic_id, title_query=topic_id)
        nuggets = []
        for entry in topic.entries:
            refs = [Reference(doc_id=doc_id) for doc_id in entry.doc_ids]
            nuggets.append(
                NuggetQuestion(
                    query_id=topic_id,
                    question=entry.question,
                    references=refs,
                )
            )
        bank.add_nuggets(nuggets)
        banks.append(bank)
    return NuggetBanks.from_banks_list(banks)