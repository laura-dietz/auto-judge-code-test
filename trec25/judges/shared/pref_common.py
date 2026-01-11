"""
Shared utilities for preference-based AutoJudge implementations.

Provides:
- PrefJudgeData: Data model for pairwise comparisons
- PrefJudgment: DSPy signature for preference judgments
- Pair formation and sampling utilities
- Borda count aggregation
- DSPy batch runner wrapper
"""

import asyncio
import re
from itertools import groupby
from math import gcd
from textwrap import dedent
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Type

import dspy
from pydantic import BaseModel

from trec_auto_judge import (
    MinimaLlmConfig,
    OpenAIMinimaLlm,
    Report,
    Request,
)
from trec_auto_judge.llm.minima_llm_dspy import run_dspy_batch


# =============================================================================
# Data Models
# =============================================================================


class PrefJudgeData(BaseModel):
    """Data model for pairwise preference comparisons."""

    run_id: str
    run_id2: str
    query_id: str
    query_title: str
    query_problem: str = ""
    query_background: str = ""
    passage_1: str
    passage_2: str
    better_passage: Optional[int] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

    def _swap(self, better_passage: Optional[int]) -> Optional[int]:
        """Swap passage preference for reversing 1<->2."""
        if better_passage is None:
            return None
        if better_passage == 1:
            return 2
        if better_passage == 2:
            return 1
        return better_passage

    def flip(self) -> "PrefJudgeData":
        """Create reversed comparison (passage_1 <-> passage_2)."""
        return PrefJudgeData(
            run_id=self.run_id2,
            run_id2=self.run_id,
            query_id=self.query_id,
            query_title=self.query_title,
            query_problem=self.query_problem,
            query_background=self.query_background,
            passage_1=self.passage_2,
            passage_2=self.passage_1,
            better_passage=self._swap(self.better_passage),
            confidence=self.confidence,
            reasoning=self.reasoning,
        )


# =============================================================================
# DSPy Signature
# =============================================================================

def _parse_better_ties(s: str) -> int:
    """Extract passage 0-2 from string."""
    m = re.search(r"\b([0-2])\b", s)
    if not m:
        return -1  # Default to -1 if no valid preference is found
    return int(m.group(1))


def _parse_better(s: str) -> int:
    """Extract passage 1-2 from string."""
    m = re.search(r"\b([1-2])\b", s)
    if not m:
        return 0  # Default to 0 if no valid preference is found
    return int(m.group(1))


class PrefJudgment(dspy.Signature):
    """
    Pairwise preference judgments.
    Mostly following Prompt of Arabzadeh & Clarke, except asking for query instead of question.
    Source: https://github.com/Narabzad/llm-relevance-judgement-comparison/blob/main/Pref/judge.py
    """

    dedent(
        """
You are a highly experienced and accurate assessor for TREC.

Select the passage that answers the query better. Just answer 1 or 2, without any explanation or extra verbiage.
If both passages are similar, select the simplest and clearest.
        """
    )

    query_title: str = dspy.InputField(desc="Query title")
    query_background: str = dspy.InputField(desc="Background context for the query")
    query_problem: str = dspy.InputField(desc="Problem statement to be addressed")

    passage_1: str = dspy.InputField(desc="passage 1")
    passage_2: str = dspy.InputField(desc="passage 2")

    better_passage: Literal["1", "2"] = dspy.OutputField(
        desc="which is the better passage?"
    )

    @classmethod
    def convert_prompt_output(
        cls, prediction: dspy.Prediction, data: PrefJudgeData
    ) -> None:
        """Convert DSPy Prediction output to PrefJudgeData."""
        data.better_passage = _parse_better(prediction.better_passage)
        data.confidence = (
            prediction.confidence or 0.0 if hasattr(prediction, "confidence") else 0.0
        )
        data.reasoning = (
            prediction.reasoning or "" if hasattr(prediction, "reasoning") else ""
        )



class PrefTiesJudgment(dspy.Signature):

    dedent(
        """
You are a highly experienced and accurate assessor for TREC.

Select the passage that answers the query better. Just answer 1 or 2, without any explanation or extra verbiage.
If both passages are similar, answer with 0.
        """
    )

    query_title: str = dspy.InputField(desc="Query title")
    query_background: str = dspy.InputField(desc="Background context for the query")
    query_problem: str = dspy.InputField(desc="Problem statement to be addressed")

    passage_1: str = dspy.InputField(desc="passage 1")
    passage_2: str = dspy.InputField(desc="passage 2")

    better_passage: Literal["1", "2", "0"] = dspy.OutputField(
        desc="which is the better passage?"
    )

    @classmethod
    def convert_prompt_output(
        cls, prediction: dspy.Prediction, data: PrefJudgeData
    ) -> None:
        """Convert DSPy Prediction output to PrefJudgeData."""
        data.better_passage = _parse_better_ties(prediction.better_passage)
        data.confidence = (
            prediction.confidence or 0.0 if hasattr(prediction, "confidence") else 0.0
        )
        data.reasoning = (
            prediction.reasoning or "" if hasattr(prediction, "reasoning") else ""
        )



# =============================================================================
# Pair Formation and Sampling
# =============================================================================


def select_comparison_samples(
    responses: List[Report],
    idx: int,
    num_pivot: int,
    num_others: int,
) -> List[Report]:
    """
    Select responses to compare against for pairwise preference judging.

    Returns a list containing:
    - Pivot responses: always responses[0:num_pivot]
    - Strided samples: from non-pivot responses, rotated around idx

    Args:
        responses: All responses for a topic
        idx: Index of current response being processed
        num_pivot: Number of pivot responses (compared against all)
        num_others: Max number of non-pivot comparisons to sample
    """
    pivots = responses[0:num_pivot]
    non_pivots = responses[num_pivot:]

    if not non_pivots:
        return list(pivots)

    if idx < num_pivot:
        # This is a pivot, it will be automatically selected for all others.
        # We only need to return other pivots.
        # Well actually only need to consider preceding pivots, because we consider pairs both ways in flip
        return pivots[:idx]
    else:
        if num_others <= 0:
            # Only compare to pivots
            return list(pivots)
        else:
            adj_idx = idx - num_pivot
            rotated = non_pivots[adj_idx + 1 :] + non_pivots[:adj_idx]

            # Stride = len/num_others to get ~num_others evenly-spaced samples
            # max(1, ...) ensures we never skip zero elements
            stride = max(1, len(rotated) // num_others) if rotated else 1

            # Phase offset ensures different responses sample different positions
            # when stride and len(rotated) share a common factor
            phase = idx % gcd(stride, len(rotated)) if rotated else 0

            return list(pivots) + rotated[phase::stride][:num_others]


def prepare_prompts(
    rag_topic_dict: Dict[str, Request],
    rag_response_by_topic: Dict[str, List[Report]],
    num_pivot: int,
    num_others: int,
    no_dupes: bool = True,
) -> List[PrefJudgeData]:
    """Create pairwise comparison prompts for all responses."""
    prompts: List[PrefJudgeData] = []
    for topic_id, responses in rag_response_by_topic.items():
        # Sort responses by run_id for deterministic pair ordering
        responses = sorted(responses, key=lambda r: r.metadata.run_id)
        if num_pivot:
            print("pivots: ", [r.metadata.run_id for r in responses[0:num_pivot]])
        if topic_id not in rag_topic_dict:
            available = sorted(rag_topic_dict.keys())[:10]
            available_str = ", ".join(repr(k) for k in available)
            if len(rag_topic_dict) > 10:
                available_str += f", ... ({len(rag_topic_dict)} total)"
            raise KeyError(
                f"Topic ID {topic_id!r} from responses not found in --rag-topics.\n"
                f"  Available topic IDs: {available_str}\n"
                f"  Check that --rag-responses and --rag-topics use matching topic IDs."
            )
        request = rag_topic_dict[topic_id]
        seen: Set[Tuple[str, str]] = set()
        for idx, response in enumerate(responses):
            run_id = response.metadata.run_id
            text = response.get_report_text()

            # Select comparison samples (pivots + strided non-pivots)
            for response_other in select_comparison_samples(
                responses, idx, num_pivot, num_others
            ):
                run_id_other = response_other.metadata.run_id
                if run_id_other == run_id:  # skip self
                    continue
                # Skip if we've already seen this pair (in either direction)
                if no_dupes and (run_id, run_id_other) in seen:
                    continue
                seen.add((run_id, run_id_other))
                seen.add((run_id_other, run_id))
                prompts.append(
                    PrefJudgeData(
                        run_id=run_id,
                        query_id=topic_id,
                        passage_1=text,
                        run_id2=run_id_other,
                        passage_2=response_other.get_report_text(),
                        query_title=request.title or "",
                        query_problem=request.problem_statement or "",
                        query_background=request.background or "",
                    )
                )
    return prompts


# =============================================================================
# Borda Count Aggregation
# =============================================================================


class PrefAggregateResult(BaseModel):
    """Aggregated preference results for a single (run_id, topic_id) pair."""

    run_id: str
    topic_id: str
    borda_score: int  # #wins - #losses
    win_frac: float  # normalized to [-1, 1]
    better_than: List[str]  # run_ids this response beat
    worse_than: List[str]  # run_ids this response lost to


def compute_pref_aggregates(
    grade_data: List[PrefJudgeData],
) -> Dict[str, PrefAggregateResult]:
    """
    Compute Borda count aggregates from pairwise preference data.

    Args:
        grade_data: List of pairwise comparisons with better_passage filled in

    Returns:
        Dict mapping "run_id:topic_id" -> PrefAggregateResult
    """
    data_by_key = {
        k: list(g)
        for k, g in groupby(
            sorted(grade_data, key=lambda data: f"{data.run_id}:{data.query_id}"),
            key=lambda data: f"{data.run_id}:{data.query_id}",
        )
    }

    aggregates: Dict[str, PrefAggregateResult] = {}
    for key, pref_data_list in data_by_key.items():
        if not pref_data_list:
            continue

        first = pref_data_list[0]
        borda_score = sum(
            1 if data.better_passage == 1 else -1 for data in pref_data_list
        )
        win_frac = float(borda_score) / float(len(pref_data_list))

        aggregates[key] = PrefAggregateResult(
            run_id=first.run_id,
            topic_id=first.query_id,
            borda_score=borda_score,
            win_frac=win_frac,
            better_than=[
                data.run_id2 for data in pref_data_list if data.better_passage == 1
            ],
            worse_than=[
                data.run_id2 for data in pref_data_list if data.better_passage == 2
            ],
        )

    return aggregates


# =============================================================================
# DSPy Batch Runner
# =============================================================================


def run_pref_judgment_batch(
    grade_data: List[PrefJudgeData],
    llm_config: MinimaLlmConfig,
    signature: Type[dspy.Signature] = PrefJudgment,
    converter: Callable[[dspy.Prediction, PrefJudgeData], None] = None,
) -> List[PrefJudgeData]:
    """
    Run DSPy batch for preference judgments.

    Args:
        grade_data: List of pairwise comparisons to judge
        llm_config: LLM configuration
        signature: DSPy signature class (default: PrefJudgment)
        converter: Output converter function (default: signature.convert_prompt_output)

    Returns:
        Updated grade_data with predictions filled in
    """
    if not grade_data:
        return grade_data

    if converter is None:
        converter = signature.convert_prompt_output

    return asyncio.run(
        run_dspy_batch(
            signature,
            grade_data,
            converter,
            backend=OpenAIMinimaLlm(llm_config),
        )
    )