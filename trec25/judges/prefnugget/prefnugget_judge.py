#!/usr/bin/env python3
"""
PrefNuggetJudge: Extract differentiating nuggets from preference comparisons.

First runs pairwise comparisons (via pref_common), then extracts NuggetQuestion
objects explaining WHY the better response won.

This judge is primarily a nugget creator - judge() returns (None, None).
"""
import json
from itertools import groupby
from typing import Any, Dict, List, Optional, Sequence, Set, Type

import dspy
from pydantic import BaseModel

from trec_auto_judge import MinimaLlmConfig

from trec_auto_judge import *
from trec_auto_judge.nugget_data import (
    NuggetBank, NuggetBanks, NuggetQuestion
)

# Import shared utilities
from trec_auto_judge.llm.minima_llm_dspy import run_dspy_batch_generic
from trec25.judges.shared.pref_common import (
    compute_pref_aggregates,
    prepare_prompts,
    run_pref_judgment_batch,
)
from trec25.judges.shared.rubric_common import (
    NuggetGradeData,
    GradeNuggetAnswer,
    prepare_nugget_grade_data,
    compute_nugget_aggregates,
)


# =============================================================================
# Leaderboard & Qrels Specs (judge-specific)
# =============================================================================

PREFNUGGET_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("NUGGET_COVERAGE", aggregate=mean_of_floats, cast=float, default=0.0),
    MeasureSpec("AVG_GRADE", aggregate=mean_of_floats, cast=float, default=0.0),
    MeasureSpec("MAX_GRADE", aggregate=mean_of_floats, cast=float, default=0.0),
    MeasureSpec("COVERED_COUNT", aggregate=mean_of_floats, cast=float, default=0.0),
))


PREFNUGGET_QRELS: QrelsSpec[NuggetGradeData] = QrelsSpec[NuggetGradeData](
    topic_id=lambda r: r.query_id,
    doc_id=lambda r: doc_id_md5(r.passage),
    grade=lambda r: float(r.grade),
    on_duplicate="keep_max"
)


# =============================================================================
# DSPy Signature (for nugget extraction - specific to PrefNuggetJudge)
# =============================================================================


class ExtractDifferentiatingNuggets(dspy.Signature):
    """
    For a query as title, problem statement, and user background, you are given Winner and Loser RAG responses. Generate brief, atomic questions
    that target query-essential information which the Winner answers well and the Loser omits or mishandles.

    Only include differences that change the answer to the query (correctness, completeness,
    usefulness). Avoid generic quality questions.
    """

    query_title: str = dspy.InputField(desc="Query title")
    query_background: str = dspy.InputField(desc="Background context for the query")
    winner_passage: str = dspy.InputField(desc="The passage that won the comparison")
    loser_passage: str = dspy.InputField(desc="The passage that lost the comparison")

    differentiating_questions: list[str] = dspy.OutputField(
        desc="List of atomic questions identifying what made the winner better and which must be answered to address the query."
    )


# =============================================================================
# Data Model (for nugget extraction - specific to PrefNuggetJudge)
# =============================================================================


class PrefNuggetData(BaseModel):
    """Data model for extracting differentiating nuggets from comparison pairs."""

    # Input fields (for DSPy signature)
    query_id: str
    query_title: str
    query_background: str = ""
    winner_run_id: str
    loser_run_id: str
    winner_passage: str
    loser_passage: str

    # Output fields (populated by LLM)
    differentiating_questions: List[str] = []



# =============================================================================
# PrefNuggetJudge Implementation
# =============================================================================


class PrefNuggetJudge(AutoJudge):
    """
    AutoJudge that extracts differentiating nuggets from PrefJudge comparisons.

    Requires responses to have evaldata with 'better_than' lists from PrefJudge.
    Produces NuggetBanks containing NuggetQuestion objects.
    """

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def __init__(self):
        pass

    def create_nuggets(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanks] = None,
        max_questions_per_pair: int = 5,
        num_pivot: int = 0,
        num_others: int = 8,
        no_dupes:bool = True,
        **kwargs,
    ) -> Optional[NuggetBanks]:
        """
        Extract differentiating nuggets from pairwise preference comparisons.

        First runs pairwise comparisons (like PrefJudge), then extracts
        NuggetQuestion objects explaining WHY the better response won.

        Args:
            rag_responses: Responses to compare
            rag_topics: Topics being evaluated
            llm_config: LLM configuration
            nugget_banks: Ignored (not used for refinement)
            max_questions_per_pair: Max questions to extract per comparison
            num_pivot: Number of pivot responses (compared against all)
            num_others: Max number of non-pivot comparisons to sample

        Returns:
            NuggetBanks with differentiating questions per topic
        """
        # Build lookup structures
        rag_topic_dict: Dict[str, Request] = {t.request_id: t for t in rag_topics}
        rag_response_by_topic: Dict[str, List[Report]] = {
            topic: list(responses)
            for topic, responses in groupby(
                sorted(rag_responses, key=lambda r: r.metadata.topic_id),
                key=lambda r: r.metadata.topic_id,
            )
        }
        responses_by_key: Dict[str, Report] = {
            f"{r.metadata.run_id}:{r.metadata.topic_id}": r for r in rag_responses
        }

        # Step 1: Run pairwise preference comparisons
        print(f"PrefNuggetJudge: Running pairwise comparisons (num_pivot={num_pivot}, num_others={num_others})...")
        grade_data = prepare_prompts(
            rag_topic_dict=rag_topic_dict,
            rag_response_by_topic=rag_response_by_topic,
            num_pivot=num_pivot,
            num_others=num_others,
            no_dupes=no_dupes
        )

        if not grade_data:
            print("PrefNuggetJudge: No comparison pairs generated")
            return None

        grade_data = run_pref_judgment_batch(grade_data, llm_config)
        print(f"PrefNuggetJudge: Completed {len(grade_data)} pairwise comparisons")

        # Include pairs in reverse for position bias handling
        grade_data = grade_data + [data.flip() for data in grade_data]

        # Compute aggregates (better_than/worse_than lists)
        aggregates = compute_pref_aggregates(grade_data)

        # Step 2: Extract comparison pairs from aggregates
        extraction_data: List[PrefNuggetData] = []
        seen_pairs: Set[tuple[str, str, str]] = set()  # (topic_id, winner, loser)

        for _key, agg in aggregates.items():
            topic_id = agg.topic_id
            winner_run_id = agg.run_id
            winner_key = f"{winner_run_id}:{topic_id}"
            winner_response = responses_by_key.get(winner_key)

            if not winner_response:
                continue

            request = rag_topic_dict.get(topic_id)
            if not request:
                continue

            # This response beat these runs
            for loser_run_id in agg.better_than:
                pair_key = (topic_id, winner_run_id, loser_run_id)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    loser_key = f"{loser_run_id}:{topic_id}"
                    loser_response = responses_by_key.get(loser_key)
                    if loser_response:
                        extraction_data.append(
                            PrefNuggetData(
                                query_id=topic_id,
                                query_title=request.title or "",
                                query_background=request.background or "",
                                winner_run_id=winner_run_id,
                                loser_run_id=loser_run_id,
                                winner_passage=winner_response.get_report_text(),
                                loser_passage=loser_response.get_report_text(),
                            )
                        )

        if not extraction_data:
            print("PrefNuggetJudge: No winner/loser pairs found after comparison")
            return None

        print(
            f"PrefNuggetJudge: Extracting nuggets from {len(extraction_data)} comparison pairs..."
        )

        # Output converter
        def convert_output(
            prediction: dspy.Prediction, data: PrefNuggetData
        ) -> None:
            questions = (
                prediction.differentiating_questions
                if hasattr(prediction, "differentiating_questions")
                else []
            )
            # DSPy may return list as JSON string - parse it
            if isinstance(questions, str):
                try:
                    parsed = json.loads(questions)
                    if isinstance(parsed, list):
                        questions = [str(q).strip() for q in parsed if q]
                    else:
                        questions = [q.strip() for q in questions.split("\n") if q.strip()]
                except json.JSONDecodeError:
                    questions = [q.strip() for q in questions.split("\n") if q.strip()]
            data.differentiating_questions = questions[:max_questions_per_pair]

        # Run LLM extraction
        extraction_data = run_dspy_batch_generic(
            extraction_data,
            ExtractDifferentiatingNuggets,
            convert_output,
            llm_config,
        )
        print("PrefNuggetJudge: Finished extracting nuggets")

        # Aggregate by topic
        results_by_topic: Dict[str, List[PrefNuggetData]] = {}
        for data in extraction_data:
            results_by_topic.setdefault(data.query_id, []).append(data)

        # Build NuggetBanks
        banks: List[NuggetBank] = []
        total_nuggets = 0
        for topic_id, topic_results in results_by_topic.items():
            request = rag_topic_dict.get(topic_id)
            bank = self._aggregate_nuggets_for_topic(
                topic_id=topic_id,
                title_query=request.title if request else topic_id,
                extraction_results=topic_results,
            )
            banks.append(bank)
            total_nuggets += len(bank.nuggets_as_list())

        print(
            f"PrefNuggetJudge: Created {total_nuggets} nuggets across {len(banks)} topics"
        )

        return NuggetBanks.from_banks_list(banks)

    def _aggregate_nuggets_for_topic(
        self,
        topic_id: str,
        title_query: str,
        extraction_results: List[PrefNuggetData],
    ) -> NuggetBank:
        """Aggregate extracted questions into a single NuggetBank, deduplicating."""
        bank = NuggetBank(query_id=topic_id, title_query=title_query)
        seen_questions: Dict[str, NuggetQuestion] = {}

        for result in extraction_results:
            for question_text in result.differentiating_questions:
                normalized = question_text.strip()
                if not normalized:
                    continue

                if normalized in seen_questions:
                    # Could merge references here if NuggetQuestion supported it
                    pass
                else:
                    nugget = NuggetQuestion(
                        query_id=topic_id,
                        question=normalized,
                        question_id=f"{topic_id}-pn{len(seen_questions)}",
                    )
                    seen_questions[normalized] = nugget

        bank.add_nuggets(list(seen_questions.values()))
        bank.index_nuggets()
        return bank


    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanks] = None,
        grade_threshold: int = 3,
        on_missing_evals: str = "fix_aggregate",
        filebase: str = "prefnugget",
        **kwargs
    ) -> tuple[Leaderboard, Optional[Qrels]]:
        """
        Grade each response against all nuggets for its topic.

        Uses shared rubric utilities for grading and aggregation.
        """
        if nugget_banks is None:
            raise ValueError("PrefNuggetJudge requires nugget_banks. Run create_nuggets first or provide --nugget-banks.")

        self.expected_topic_ids = [t.request_id for t in rag_topics]

        # Prepare grading data using shared utility
        print("PrefNuggetJudge: Preparing grade data...")
        grade_data, nuggets_per_topic = prepare_nugget_grade_data(rag_responses, nugget_banks)

        # Run LLM grading using shared utility
        print("PrefNuggetJudge: Grading responses...")
        grade_data = run_dspy_batch_generic(
            grade_data,
            GradeNuggetAnswer,
            GradeNuggetAnswer.convert_prompt_output,
            llm_config,
        )
        print("PrefNuggetJudge: Finished grading")

        # Aggregate grades using shared utility
        aggregates = compute_nugget_aggregates(grade_data, nuggets_per_topic, grade_threshold)

        # Update Report.evaldata
        for response in rag_responses:
            response_key = f"{response.metadata.run_id}:{response.metadata.topic_id}"
            if response_key in aggregates:
                agg = aggregates[response_key]
                response.evaldata = {
                    "nugget_grades": agg.nugget_grades,
                    "coverage_score": agg.coverage_score,
                    "avg_grade": agg.avg_grade,
                    "max_grade": agg.max_grade,
                    "covered_count": agg.covered_count,
                    "total_nuggets": agg.total_nuggets,
                    "graded_nuggets": agg.graded_nuggets,
                }

        # Build leaderboard
        leaderboard = self._build_leaderboard(aggregates, on_missing_evals)
        leaderboard.verify(warn=True, expected_topic_ids=self.expected_topic_ids, on_missing=on_missing_evals)

        # Build qrels from grade data
        qrels = build_qrels(records=grade_data, spec=PREFNUGGET_QRELS) if grade_data else None
        if qrels is not None:
            qrels.verify(warn=True, expected_topic_ids=self.expected_topic_ids)

        return (leaderboard, qrels)

    def _build_leaderboard(self, aggregates: Dict[str, Any], on_missing_evals: str) -> Leaderboard:
        """Build leaderboard from aggregated response grades."""
        b = LeaderboardBuilder(PREFNUGGET_SPEC)

        for response_key, agg in aggregates.items():
            run_id, topic_id = response_key.split(":", 1)
            b.add(
                run_id=run_id,
                topic_id=topic_id,
                values={
                    "NUGGET_COVERAGE": agg.coverage_score,
                    "AVG_GRADE": agg.avg_grade,
                    "MAX_GRADE": agg.max_grade,
                    "COVERED_COUNT": float(agg.covered_count),
                }
            )

        leaderboard = b.build(expected_topic_ids=self.expected_topic_ids, on_missing=on_missing_evals)
        leaderboard.verify(expected_topic_ids=self.expected_topic_ids, warn=False, on_missing=on_missing_evals)
        return leaderboard


if __name__ == "__main__":
    auto_judge_to_click_command(PrefNuggetJudge(), "prefnugget_judge")()