"""
Shared utilities for rubric/nugget-based AutoJudge implementations.

Provides:
- NuggetGradeData: Data model for nugget grading
- GradeNuggetAnswer: DSPy signature for grading passages against questions
- Grade aggregation and coverage computation
"""

import re
from textwrap import dedent
from typing import Any, Dict, List, Literal, Optional, Sequence

import dspy
from pydantic import BaseModel

from trec_auto_judge import Report
from trec_auto_judge.nugget_data import NuggetBanks, NuggetQuestion


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
    reasoning: Optional[str] = dspy.OutputField(
        desc="Brief explanation of the grade", default=None, required=False
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

        response_data[response_key]["nugget_grades"][data.nugget_id] = {
            "grade": data.grade,
            "reasoning": data.reasoning,
        }
        response_data[response_key]["grades_list"].append(data.grade)

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