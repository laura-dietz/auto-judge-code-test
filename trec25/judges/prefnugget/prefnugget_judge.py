#!/usr/bin/env python3
"""
PrefNuggetJudge: Extract differentiating nuggets from preference comparisons.

First runs pairwise comparisons (via pref_common), then extracts NuggetQuestion
objects explaining WHY the better response won.

This judge is primarily a nugget creator - judge() returns (None, None).
"""
import collections
from itertools import groupby
import sys
from textwrap import dedent
from typing import Any, Dict, List, Literal, Set, TypeVar

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
    PrefJudgment,
    PrefTiesJudgment,
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
    __doc__ = dedent(
        """
        For a query as title, problem statement, and user background, you are given Winner and Loser RAG responses.
        Generate brief, atomic questions that target query-essential information which the Winner answers well
        and the Loser omits or mishandles.

        Only include differences that change the answer to the query (correctness, completeness,
        usefulness). Prefer short questions such as "Capital of USA?" or "Process of steel cooking?".
        Avoid generic quality questions.
        """
    )

    query_title: str = dspy.InputField(desc="Query title")
    query_background: str = dspy.InputField(desc="Background context for the query")
    winner_passage: str = dspy.InputField(desc="The passage that won the comparison")
    loser_passage: str = dspy.InputField(desc="The passage that lost the comparison")

    differentiating_questions: list[str] = dspy.OutputField(
        desc='JSON array, e.g. ["Capital of USA?", "Process to cook steel?"]'
        # desc='JSON array with double quotes, e.g. ["USA\'s capital?", "Process to cook steel?"]'
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why these questions differentiate the passages"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score from 0.0 to 1.0 indicating how certain you are"
    )



class IterativeExtractDifferentiatingNuggets(dspy.Signature):
    __doc__ = dedent(
      """
      Compare Winner vs Loser RAG responses for a query. Focus on relevance, correctness, completeness.
      
      From given_exam_questions, identify or generate which ones the Winner handles well but the Loser 
      omits or mishandles.  New differentiating_questions must be brief, 
      atomic questions about information the Winner handels much better.

      Avoid generic quality questions. 
      Make questions self-contained (e.g., "Capital of France?" not "The capital?").
      """        
        # """
        # For a query with title and background, you are given Winner and Loser RAG responses.

        # Examine the given_exam_questions. Which of these questions are answered
        # better by the Winner than the Loser? 

        # You can also generate new differentiating_questions that capture better 
        # explain why the Winner is better. These should be brief, atomic questions
        # targeting query-essential information which the Winner answers well and the
        # Loser omits or mishandles.

        # Focus on differences that affect rlevance, correctness, completeness, or usefulness.
        # Prefer short questions like "Capital of USA?" or "Process of steel cooking?".
        # Avoid generic quality questions. Resolve all determiner and implicit references.
        # """
    )

    query_title: str = dspy.InputField(desc="Query title")
    query_background: str = dspy.InputField(desc="Background context for the query")
    winner_passage: str = dspy.InputField(desc="The passage that won the comparison")
    loser_passage: str = dspy.InputField(desc="The passage that lost the comparison")
    given_exam_questions: list[str] = dspy.InputField(desc="Given exam questions")

    differentiating_questions: Optional[List[str]] = dspy.OutputField(
        desc='Generated questions as a JSON array, e.g. ["Capital of USA?", "Process to cook steel?"]'
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of the analysis"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score from 0.0 to 1.0 indicating how certain you are"
    )



class IterativeDExtractDifferentiatingNuggets(dspy.Signature):
    __doc__ = dedent(
        """
        For a query with title and background, you are given Winner and Loser RAG responses.

        First, examine the given_exam_questions. Which of these questions are answered
        better by the Winner than the Loser? Output those as addressed_questions.

        Only if NONE of the given_exam_questions explain why the Winner is better,
        generate new differentiating_questions. These should be brief, atomic questions
        targeting query-essential information which the Winner answers well and the
        Loser omits or mishandles.

        Focus on differences that affect correctness, completeness, or usefulness.
        Prefer short questions like "Capital of USA?" or "Process of steel cooking?".
        Avoid generic quality questions.
        """
    )

    query_title: str = dspy.InputField(desc="Query title")
    query_background: str = dspy.InputField(desc="Background context for the query")
    winner_passage: str = dspy.InputField(desc="The passage that won the comparison")
    loser_passage: str = dspy.InputField(desc="The passage that lost the comparison")
    given_exam_questions: list[str] = dspy.InputField(desc="Given exam questions")

    addressed_questions: Optional[List[str]] = dspy.OutputField(
        desc='The questions better addressed in the Winner as a JSON array, e.g. ["Capital of USA?", "Process to cook steel?"]'
    )
    differentiating_questions: Optional[List[str]] = dspy.OutputField(
        desc='Generated questions as a JSON array, e.g. ["Capital of USA?", "Process to cook steel?"]'
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of the analysis"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score from 0.0 to 1.0 indicating how certain you are"
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



class IterativePrefNuggetData(BaseModel):
    """Data model for extracting differentiating nuggets from comparison pairs."""

    # Input fields (for DSPy signature)
    query_id: str
    query_title: str
    query_background: str = ""
    winner_run_id: str
    loser_run_id: str
    winner_passage: str
    loser_passage: str
    given_exam_questions: List[str] = []

    # Output fields (populated by LLM)
    addressed_questions: List[str] = []
    differentiating_questions: List[str] = []


# =============================================================================
# PrefNuggetJudge Implementation
# =============================================================================

T = TypeVar('T')


class QuestionTracker:
    """Track unique questions and their occurrence counts per topic."""

    def __init__(self):
        # counts[query_id][question] = count
        self._counts: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))

    def add(self, query_id: str, question: str, count: int = 1) -> None:
        """Add a question, incrementing its count."""
        self._counts[query_id][question] += count

    def questions(self, query_id: str) -> List[str]:
        """Get list of unique questions for a topic."""
        return list(self._counts[query_id].keys())

    def count(self, query_id: str, question: str) -> int:
        """Get count for a specific question."""
        return self._counts[query_id].get(question, 0)

    def counts_dict(self, query_id: str) -> Dict[str, int]:
        """Get all counts for a topic."""
        return dict(self._counts[query_id])

    def num_questions(self, query_id: str) -> int:
        """Get number of unique questions for a topic."""
        return len(self._counts[query_id])

    def items(self):
        """Iterate over (query_id, questions_dict) pairs."""
        return self._counts.items()

    def top_questions(self, query_id: str, n: int) -> List[str]:
        """Get top n questions by count, sorted descending."""
        return sorted(
            self._counts[query_id].keys(),
            key=lambda q: self._counts[query_id][q],
            reverse=True
        )[:n]

def _print_tracker(tracker: QuestionTracker) -> str:
    lines = []
    for query_id, counts in sorted(tracker.items())[:5]:
        # Sort by count descending, then alphabetically for determinism
        sorted_qs = sorted(counts.keys(), key=lambda q: (-counts[q], q))[:5]
        formatted = [f"  - {q} ({counts[q]})" for q in sorted_qs]
        lines.append(f"{query_id}: {len(counts)} questions:\n" + "\n".join(formatted))
    return "\n".join(lines)




def _chunk_by_query_both(
    lst: List[IterativePrefNuggetData],
    borda_scores: Dict[str, int],
    nugget_gen_order: Literal["winner", "both"],
    num_per_query: int = 2, 
    max_pairs_considered: int = -1
) -> List[List[IterativePrefNuggetData]]:
    """Split list into chunks with at most `num_per_query` items per query_id.

    Pairs with higher borda_scores (sum) are prioritized first.

    Args:
        lst: List of data items with query_id attribute
        borda_scores: Mapping of "run_id:topic_id" -> borda_score
        nugget_gen_order: Sorting strategy
        num_per_query: Maximum items per query_id in each chunk
        max_pairs_considered (k): only look at the top-k pairs

    Returns:
        List of batches, each respecting the per-query limit
    """
    if not lst:
        return []

    # First split by topic (convert groupby iterator to dict)
    sorted_per_topic: Dict[str, List[IterativePrefNuggetData]] = {
        k: list(g)
        for k, g in groupby(sorted(lst, key=lambda d: d.query_id), key=lambda d: d.query_id)
    }

    for query_id in sorted_per_topic:
        topic_lst = sorted_per_topic[query_id]

        # Sort by quality within each topic
        if nugget_gen_order == 'both':
            topic_lst = sorted(
                topic_lst,
                key=lambda x: (
                    borda_scores.get(f"{x.winner_run_id}:{x.query_id}", 0)
                    + 0.99 * borda_scores.get(f"{x.loser_run_id}:{x.query_id}", 0)
                ),
                reverse=True,
            )
        elif nugget_gen_order == 'winner':
            topic_lst = sorted(
                topic_lst,
                key=lambda x: borda_scores.get(f"{x.winner_run_id}:{x.query_id}", 0),
                reverse=True,
            )

        # Limit to top-k pairs per topic (if max_pairs_considered > 0)
        if max_pairs_considered > 0:
            sorted_per_topic[query_id] = topic_lst[:max_pairs_considered]
        else:
            sorted_per_topic[query_id] = topic_lst
    
    # Third build chunks by popping `n` per topic
    
    chunks=[]    
    while any(x for x in sorted_per_topic.values()):
        chunk: List[IterativePrefNuggetData] = []
        
        for query_id in sorted_per_topic:
            for rounds in range(num_per_query):
                lst = sorted_per_topic[query_id]
                if lst:
                    elem = lst.pop(0)
                    sorted_per_topic[query_id]=lst
                    chunk.append(elem)
            
        chunks.append(chunk)
        
    return chunks


def _chunk_by_query_first(
    lst: List[IterativePrefNuggetData],
    borda_scores: Dict[str, int],
    max_size: int = -1,
    num_per_query: int = 2, 
) -> List[List[IterativePrefNuggetData]]:
    """(Obsolete) Split list into chunks with at most `num_per_query` items per query_id.

    Items with higher borda_score (for their winner) are prioritized first.

    Args:
        lst: List of data items with query_id attribute
        borda_scores: Mapping of "run_id:topic_id" -> borda_score
        max_size: Maximum batch size (-1 means unlimited)
        num_per_query: Maximum items per query_id in each batch

    Returns:
        List of batches, each respecting the per-query limit
    """
    if not lst:
        return []

    # Sort by borda_score descending (best winners first)
    sorted_lst = sorted(
        lst,
        key=lambda x: borda_scores.get(f"{x.winner_run_id}:{x.query_id}", 0),
        reverse=True
    )

    chunks: List[List[IterativePrefNuggetData]] = []
    remaining = sorted_lst

    while remaining:
        chunk: List[IterativePrefNuggetData] = []
        query_counts: Dict[str, int] = collections.defaultdict(int)
        still_remaining: List[IterativePrefNuggetData] = []

        for item in remaining:
            # Check if we can add this item
            at_max_size = max_size > 0 and len(chunk) >= max_size
            at_query_limit = query_counts[item.query_id] >= num_per_query

            if at_max_size:
                # Batch is full, save rest for next chunk
                still_remaining.append(item)
            elif at_query_limit:
                # This query has enough in current batch, defer to next
                still_remaining.append(item)
            else:
                # Add to current batch
                chunk.append(item)
                query_counts[item.query_id] += 1

        if chunk:
            chunks.append(chunk)
        remaining = still_remaining

    return chunks


def _chunk_by_query(
    lst: List[IterativePrefNuggetData],
    borda_scores: Dict[str, int],
    nugget_gen_order: Literal["first", "both", "winner"],
    max_size: int = -1,
    num_per_query: int = 2, 
    max_pairs_considered: int = -1
    ):
    if nugget_gen_order == "first":
        return _chunk_by_query_first(lst, borda_scores=borda_scores, max_size=max_size, num_per_query=num_per_query)
    else:
        return _chunk_by_query_both(lst
                                    , borda_scores=borda_scores
                                    , nugget_gen_order=nugget_gen_order
                                    , num_per_query=num_per_query
                                    , max_pairs_considered=max_pairs_considered
                                    )

def _interleave_by_query_id(
    data: List[IterativePrefNuggetData],
    borda_scores: Dict[str, int],
) -> List[IterativePrefNuggetData]:
    """
    Reorder so elements with same query_id are maximally spread out (round-robin).

    Within each topic, pairs are sorted by winner's borda_score (descending) so that
    comparisons involving the best-performing responses are processed first.

    Args:
        data: List of extraction data items
        borda_scores: Mapping of "run_id:topic_id" -> borda_score
    """
    by_query: Dict[str, List[IterativePrefNuggetData]] = collections.defaultdict(list)
    for item in data:
        by_query[item.query_id].append(item)

    # Sort by query_id for deterministic ordering
    sorted_query_ids = sorted(by_query.keys())

    # Within each topic, sort by winner's borda_score (descending), then by run_ids for determinism
    for qid in sorted_query_ids:
        by_query[qid].sort(
            key=lambda x: (
                -borda_scores.get(f"{x.winner_run_id}:{x.query_id}", 0),  # Best winners first
                x.winner_run_id,
                x.loser_run_id,
            )
        )

    # Round-robin interleave (deterministic order)
    result: List[IterativePrefNuggetData] = []
    iterators = [iter(by_query[qid]) for qid in sorted_query_ids]
    while iterators:
        next_round = []
        for it in iterators:
            try:
                result.append(next(it))
                next_round.append(it)
            except StopIteration:
                pass
        iterators = next_round
    return result


class PrefNuggetJudge(AutoJudge):
    """
    AutoJudge that extracts differentiating nuggets from PrefJudge comparisons.

    Requires responses to have evaldata with 'better_than' lists from PrefJudge.
    Produces NuggetBanks containing NuggetQuestion objects.
    """

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def __init__(self):
        pass

    def filter_non_topic_responses(self, rag_responses: Sequence[Report], topic_ids:Set[str])->Sequence[Report]:
        broken:bool = False
        broken_run_ids = []
        for r in rag_responses:
            if r.metadata.run_id not in broken_run_ids:
                if r.metadata.topic_id not in topic_ids:
                    print(f"Warning, report of run {r.metadata.run_id} is about topic {r.metadata.request_id}, which is not in topic_ids {format_preview(list(topic_ids), limit=10)}" , file=sys.stderr)
                    broken=True
                    broken_run_ids.append(r.metadata.run_id)
        
        if broken:
            return list(filter(lambda r: r.metadata.request_id in topic_ids, rag_responses))
        else:
            return rag_responses
    

    def create_nuggets(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        pref_judge:Literal['must_decide','ties_allowed'],
        iterative_nuggets:bool,
        max_nuggets_per_topic: int,
        stop_collecting_at_nuggets_per_topic: int,
        gen_batch_num_per_query:int,
        max_pairs_considered:int,
        nugget_gen_order: Literal["first","both", "winner"], # "first is deprecated"
        max_questions_per_pair: int = 5,
        num_pivot: int = 0,
        num_others: int = 8,
        no_dupes:bool = True,
        nugget_banks: Optional[NuggetBanks] = None,
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
        
        if not iterative_nuggets:
            raise RuntimeError("This nugget creation only produces iterative_nuggets")
        # Build lookup structures
        rag_topic_dict: Dict[str, Request] = {t.request_id: t for t in rag_topics}
        num_topics = len(rag_topic_dict)
        rag_responses = self.filter_non_topic_responses(rag_responses, rag_topic_dict.keys())
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

        pref_signature = PrefTiesJudgment if pref_judge == "ties_allowed" else PrefJudgment
        grade_data = run_pref_judgment_batch(grade_data, llm_config, signature=pref_signature)
        print(f"PrefNuggetJudge: Completed {len(grade_data)} pairwise comparisons")

        # Include pairs in reverse for position bias handling
        grade_data = grade_data + [data.flip() for data in grade_data]
        # Drop ties (only keep pairs with a clear winner)
        grade_data = [d for d in grade_data if d.better_passage in [1, 2]]

        # Compute aggregates (better_than/worse_than lists)
        aggregates = compute_pref_aggregates(grade_data)  # Note this function does not handle ties

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
                            IterativePrefNuggetData(
                                query_id=topic_id,
                                query_title=request.title or "",
                                query_background=request.background or "",
                                winner_run_id=winner_run_id,
                                loser_run_id=loser_run_id,
                                winner_passage=winner_response.get_report_text(),
                                loser_passage=loser_response.get_report_text(),
                                given_exam_questions=[]  # needs to get filled in later
                            )
                        )

        if not extraction_data:
            print("PrefNuggetJudge: No winner/loser pairs found after comparison")
            return None
        # Debug: print extraction pairs for reproducibility debugging
        for i, ed in enumerate(extraction_data[:20]):  # First 20
            print(f"  [{i}] {ed.query_id}: {ed.winner_run_id} > {ed.loser_run_id}")


        print(
            f"PrefNuggetJudge: Extracting nuggets from {len(extraction_data)} comparison pairs..."
        )

        # Output converter (JSON parsing handled by TolerantChatAdapter)
        def convert_output(
            prediction: dspy.Prediction, data: IterativePrefNuggetData
        ) -> None:
            differentiating_questions = getattr(prediction, "differentiating_questions", [])
            # Normalize: strip whitespace, filter empty
            data.differentiating_questions = [
                q.strip() for q in (differentiating_questions or [])
                if q and q.strip()
            ][:max_questions_per_pair]

            # we actually may not need those, they are just a ruse
            addressed_questions = getattr(prediction, "addressed_questions", [])
            data.addressed_questions = [
                q.strip() for q in (addressed_questions or [])
                if q and q.strip()
            ][:max_questions_per_pair]


        # =----------------------------------------------=
        # prepare small batches to generate questions, then test.


        # Build borda_scores lookup for prioritizing best-performing responses
        borda_scores: Dict[str, int] = {key: agg.borda_score for key, agg in aggregates.items()}

        # Spread out elements with same query_id (round-robin interleave)
        # Within each topic, pairs are sorted by winner's borda_score (best performers first)
        extraction_data_chunks = _chunk_by_query(
            extraction_data,
            borda_scores=borda_scores,
            nugget_gen_order=nugget_gen_order,
            num_per_query=gen_batch_num_per_query,
            max_pairs_considered=max_pairs_considered,
        )
        tracker = QuestionTracker()
        topics_done:Set[str] = set()

        extraction_result_data = list()
        for chunk_idx, extraction_chunk in enumerate(extraction_data_chunks):
            # Skip prompts for topics that already have enough questions
            extraction_chunk = [d for d in extraction_chunk if d.query_id not in topics_done]

            if not extraction_chunk:
                continue  # All topics in this chunk are done

            # set questions so far
            for data in extraction_chunk:
                topic_id = data.query_id
                data.given_exam_questions = tracker.questions(topic_id)

            # Run LLM extraction
            extraction_chunk = run_dspy_batch_generic(
                extraction_chunk,
                IterativeExtractDifferentiatingNuggets,
                convert_output,
                llm_config,
            )

            for data in filter(lambda d: d.query_id not in topics_done, extraction_chunk):
                query_id = data.query_id

                # Add new/duplicate questions (count increments for duplicates too)
                for q in data.differentiating_questions:
                    tracker.add(query_id, q)
                # Tabulate how often each question is addressed
                for q in data.addressed_questions:
                    tracker.add(query_id, q)

                # Stop collecting if we have enough unique questions
                if tracker.num_questions(query_id) > stop_collecting_at_nuggets_per_topic:
                    topics_done.add(query_id)
            extraction_result_data.extend(extraction_chunk)

            print(f"-- PrefNuggetJudge: Finished extracting nuggets pass {chunk_idx}. Questions:\n{_print_tracker(tracker)}")

        extraction_data = extraction_result_data
        print("PrefNuggetJudge: Finished extracting nuggets")
        print(f"Question counts: {dict(tracker.items())}")

        # Keep only top max_nuggets_per_topic questions per topic
        final_questions: Dict[str, List[str]] = {
            query_id: tracker.top_questions(query_id, max_nuggets_per_topic)
            for query_id, _ in tracker.items()
        }

        # =----------------------------------------------=

        # Build NuggetBanks from final_questions (already deduplicated and filtered by count)
        banks: List[NuggetBank] = []
        total_nuggets = 0
        for topic_id, questions in final_questions.items():
            request = rag_topic_dict.get(topic_id)
            bank = NuggetBank(
                query_id=topic_id,
                title_query=request.title if request else topic_id
            )
            nuggets = [
                NuggetQuestion(
                    query_id=topic_id,
                    question=q,
                    question_id=f"{topic_id}-pn{i}",
                )
                for i, q in enumerate(questions)
            ]
            bank.add_nuggets(nuggets)
            bank.index_nuggets()
            banks.append(bank)
            total_nuggets += len(nuggets)

        print(
            f"PrefNuggetJudge: Created {total_nuggets} nuggets across {len(banks)} topics"
        )

        return NuggetBanks.from_banks_list(banks)


##########################################
##########################################

    def create_nuggets_non_iterative(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        iterative_nuggets:bool,
        max_questions_per_pair: int = 5,
        num_pivot: int = 0,
        num_others: int = 8,
        no_dupes:bool = True,
        nugget_banks: Optional[NuggetBanks] = None,
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
        
        if iterative_nuggets:
            raise RuntimeError("This nugget creation does not produce iterative_nuggets")
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
        # Debug: print extraction pairs for reproducibility debugging
        for i, ed in enumerate(extraction_data[:20]):  # First 20
            print(f"  [{i}] {ed.query_id}: {ed.winner_run_id} > {ed.loser_run_id}")

        # Output converter (JSON parsing handled by TolerantChatAdapter)
        def convert_output(
            prediction: dspy.Prediction, data: PrefNuggetData
        ) -> None:
            questions = getattr(prediction, "differentiating_questions", [])
            data.differentiating_questions = list(questions)[:max_questions_per_pair] if questions else []

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