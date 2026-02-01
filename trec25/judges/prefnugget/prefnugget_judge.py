#!/usr/bin/env python3
"""
PrefNuggetJudge: Extract differentiating nuggets from preference comparisons.

First runs pairwise comparisons (via pref_common), then extracts NuggetQuestion
objects explaining WHY the better response won.

This judge is primarily a nugget creator - judge() returns (None, None).
"""
import collections
from itertools import groupby
from random import shuffle, Random
import sys
from textwrap import dedent
from typing import Any, Dict, List, Literal, Set

import dspy
from pydantic import BaseModel

from trec_auto_judge import MinimaLlmConfig

from trec_auto_judge import *
from trec_auto_judge.nugget_data import NuggetBanks, NuggetBanksProtocol


# Import shared utilities
from trec_auto_judge.llm.minima_llm_dspy import run_dspy_batch_generic
from trec25.judges.shared.pref_common import (
    PrefAggregateResult,
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
    prepare_nugget_grade_data_for_documents,
    compute_nugget_aggregates,
    compute_nugget_aggregates_for_documents,
    build_nugget_banks,
)


# =============================================================================
# Leaderboard & Qrels Specs (judge-specific)
# =============================================================================

PREFNUGGET_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("NUGGET_COVERAGE"),
    MeasureSpec("AVG_GRADE"),
    MeasureSpec("MAX_GRADE"),
    MeasureSpec("COVERED_COUNT"),
))


PREFNUGGET_QRELS: QrelsSpec[NuggetGradeData] = QrelsSpec[NuggetGradeData](
    topic_id=lambda r: r.query_id,
    doc_id=lambda r: doc_id_md5(r.passage),
    grade=lambda r: float(r.grade),
    on_duplicate="keep_max"
)

PREFNUGGET_CITE_QRELS: QrelsSpec[NuggetGradeData] = QrelsSpec[NuggetGradeData](
    topic_id=lambda r: r.query_id,
    doc_id=lambda r: r.doc_id,
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
      
      From given_exam_questions, identify or generate questions the Winner addresses much better than the Loser.
      Reuse questions where possible. New differentiating_questions must be brief, 
      atomic questions about information the Winner handels much better.

      Avoid generic quality questions. 
      Make questions self-contained (e.g., "Capital of France?" not "The capital?").
      """        
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


# =============================================================================
# Data Model (for nugget extraction - specific to PrefNuggetJudge)
# =============================================================================


class PrefNuggetData(BaseModel):
    """Data model for extracting differentiating nuggets from comparison pairs.

    Used by both iterative and non-iterative extraction paths.
    For iterative extraction, given_exam_questions is set before each batch.
    """

    # Input fields (for DSPy signature)
    query_id: str
    query_title: str
    query_background: str = ""
    winner_run_id: str
    loser_run_id: str
    winner_passage: str
    loser_passage: str
    given_exam_questions: Optional[List[str]] = None  # Set only for iterative extraction

    # Output fields (populated by LLM)
    differentiating_questions: List[str] = []


# =============================================================================
# PrefNuggetJudge Implementation
# =============================================================================


class QuestionTracker:
    """Track unique questions and their occurrence counts per topic."""

    def __init__(self):
        # counts[query_id][question] = count
        self._counts: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        self._topics_done: Set[str] = set()  # Topics that have collected enough questions

    def add(self, query_id: str, question: str, count: int = 1) -> None:
        """Add a question, incrementing its count."""
        self._counts[query_id][question] += count

    def add_all(self, query_id: str, questions: List[str], count: int = 1) -> None:
        """Add multiple questions, incrementing each count."""
        for q in questions:
            self._counts[query_id][q] += count

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

    def is_done(self, query_id: str) -> bool:
        """Check if topic has collected enough questions."""
        return query_id in self._topics_done

    def mark_done(self, query_id: str) -> None:
        """Mark topic as having collected enough questions."""
        self._topics_done.add(query_id)

    def check_and_mark_done(self, query_id: str, stop_at_count: int) -> bool:
        """Mark done if threshold exceeded. Returns True if now done."""
        if self.num_questions(query_id) > stop_at_count:
            self._topics_done.add(query_id)
            return True
        return False

    def check_all_done(self, stop_at_count: int) -> None:
        """Check all topics and mark done if threshold exceeded."""
        for query_id in self._counts.keys():
            if self.num_questions(query_id) > stop_at_count:
                self._topics_done.add(query_id)

def _print_tracker(tracker: QuestionTracker) -> str:
    lines = []
    for query_id, counts in sorted(tracker.items())[:5]:
        # Sort by count descending, then alphabetically for determinism
        sorted_qs = sorted(counts.keys(), key=lambda q: (-counts[q], q))[:5]
        formatted = [f"  - {q} ({counts[q]})" for q in sorted_qs]
        lines.append(f"{query_id}: {len(counts)} questions:\n" + "\n".join(formatted))
    return "\n".join(lines)




def _chunk_by_query_both(
    lst: List[PrefNuggetData],
    borda_scores: Dict[str, int],
    nugget_gen_order: Literal["both", "winner","as_provided"],
    num_per_query: int = 2, 
    max_pairs_considered: int = -1
) -> List[List[PrefNuggetData]]:
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
    sorted_per_topic: Dict[str, List[PrefNuggetData]] = {
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
        elif nugget_gen_order == 'as_provided':
            topic_lst = topic_lst

        # Limit to top-k pairs per topic (if max_pairs_considered > 0)
        if max_pairs_considered > 0:
            sorted_per_topic[query_id] = topic_lst[:max_pairs_considered]
        else:
            sorted_per_topic[query_id] = topic_lst
    
    # Third build chunks by popping `n` per topic
    
    chunks=[]    
    while any(x for x in sorted_per_topic.values()):
        chunk: List[PrefNuggetData] = []
        
        for query_id in sorted_per_topic:
            for rounds in range(num_per_query):
                lst = sorted_per_topic[query_id]
                if lst:
                    elem = lst.pop(0)
                    sorted_per_topic[query_id]=lst
                    chunk.append(elem)
            
        chunks.append(chunk)
        
    return chunks


def _chunk_by_query(
    lst: List[PrefNuggetData],
    borda_scores: Dict[str, int],
    nugget_gen_order: Literal["both", "winner","as_provided"],
    num_per_query: int = 2,
    max_pairs_considered: int = -1
    ):
    return _chunk_by_query_both(
        lst,
        borda_scores=borda_scores,
        nugget_gen_order=nugget_gen_order,
        num_per_query=num_per_query,
        max_pairs_considered=max_pairs_considered,
    )

# =============================================================================
# Shared Utilities
# =============================================================================

def build_response_lookups(
    rag_responses: Sequence[Report],
    rag_topics: Sequence[Request],
) -> tuple[Dict[str, Request], Dict[str, List[Report]]]:
    """Build lookup structures for responses and topics.

    Returns:
        rag_topic_dict: topic_id -> Request
        rag_response_by_topic: topic_id -> List[Report]
    """
    rag_topic_dict: Dict[str, Request] = {t.request_id: t for t in rag_topics}
    rag_response_by_topic: Dict[str, List[Report]] = {
        topic: list(responses)
        for topic, responses in groupby(
            sorted(rag_responses, key=lambda r: r.metadata.topic_id),
            key=lambda r: r.metadata.topic_id,
        )
    }
    return rag_topic_dict, rag_response_by_topic


def run_preference_phase(
    rag_topic_dict: Dict[str, Request],
    rag_response_by_topic: Dict[str, List[Report]],
    llm_config: MinimaLlmConfig,
    num_pivot: int,
    num_others: int,
    no_dupes: bool,
    pref_judge: Literal['must_decide', 'ties_allowed'] = 'must_decide',
) -> Optional[tuple[List[PrefNuggetData], Dict[str, PrefAggregateResult]]]:
    """Run pairwise preference comparisons and compute aggregates.

    Returns:
        Tuple of (grade_data, aggregates) or None if no comparison pairs generated.
        grade_data: List of comparison results (flipped for position bias, ties dropped)
        aggregates: Dict of computed aggregates with better_than/worse_than lists
    """
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
    aggregates = compute_pref_aggregates(grade_data)

    return grade_data, aggregates


def extract_random_pairs(
    # aggregates: Dict[str, PrefAggregateResult],
    rag_responses: Sequence[Report],
    rag_topic_dict: Dict[str, Request],
) -> List[PrefNuggetData]:
    """Use random pairs to (null hypothesis that winner/loser pairs are not helping).

    Args:
        aggregates: Dict of computed aggregates with better_than/worse_than lists
        rag_responses: RAG responses (used to build responses_by_key)
        rag_topic_dict: topic_id -> Request mapping

    Returns:
        List of PrefNuggetData objects ready for LLM extraction
    """
    responses_by_topic: Dict[str,List[Report]] = collections.defaultdict(list)
    for r in rag_responses:
        responses_by_topic[r.metadata.topic_id].append(r)
    
    extraction_data: List[PrefNuggetData] = []

    for topic_id, responses in responses_by_topic.items():
        rng = Random(topic_id)
        rng.shuffle(responses) # make this pseudo-random
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                winner_response = responses[i]
                loser_response = responses[j]
        
                if not winner_response:
                    continue

                winner_run_id = winner_response.metadata.run_id
                # winner_key = f"{winner_run_id}:{topic_id}"
                # winner_response = responses_by_key.get(winner_key)

                loser_run_id = loser_response.metadata.run_id
                # looser_key = f"{loser_run_id}:{topic_id}"

                request = rag_topic_dict.get(topic_id)
                if not request:
                    continue

                # This response beat these runs
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

    return extraction_data

def extract_winner_loser_pairs(
    aggregates: Dict[str, PrefAggregateResult],
    rag_responses: Sequence[Report],
    rag_topic_dict: Dict[str, Request],
) -> List[PrefNuggetData]:
    """Extract winner/loser pairs from preference aggregates.

    Args:
        aggregates: Dict of computed aggregates with better_than/worse_than lists
        rag_responses: RAG responses (used to build responses_by_key)
        rag_topic_dict: topic_id -> Request mapping

    Returns:
        List of PrefNuggetData objects ready for LLM extraction
    """
    responses_by_key: Dict[str, Report] = {
        f"{r.metadata.run_id}:{r.metadata.topic_id}": r for r in rag_responses
    }
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

    return extraction_data


def _flatten_extraction_results(
    extraction_data: List[PrefNuggetData],
    rag_topic_dict: Dict[str, Request],
) -> Dict[str, tuple[str, List[str]]]:
    """Convert extraction results to format expected by build_nugget_banks.

    Returns:
        Dict mapping topic_id -> (title, list of question strings)
    """
    # Group questions by topic
    questions_by_topic: Dict[str, List[str]] = {}
    for data in extraction_data:
        questions_by_topic.setdefault(data.query_id, []).extend(data.differentiating_questions)

    # Add titles
    return {
        topic_id: (rag_topic_dict[topic_id].title or topic_id, questions)
        for topic_id, questions in questions_by_topic.items()
        if topic_id in rag_topic_dict
    }


class PrefNuggetJudge(AutoJudge):
    """
    AutoJudge that extracts differentiating nuggets from PrefJudge comparisons.

    Requires responses to have evaldata with 'better_than' lists from PrefJudge.
    Produces NuggetBanks containing NuggetQuestion objects.
    """

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def __init__(self):
        pass

    def create_qrels(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> Optional[Qrels]:
        """PrefNuggetJudge does not produce qrels."""
        return None

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
        nugget_gen_order: Literal["both", "winner"],
        max_questions_per_pair: int = 5,
        num_pivot: int = 0,
        num_others: int = 8,
        no_dupes:bool = True,
        random_pairs:bool = False,
        # nugget_banks: Optional[NuggetBanks] = None,
        **kwargs,
    ) -> Optional[NuggetBanksProtocol]:
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
        rag_topic_dict, rag_response_by_topic = build_response_lookups(rag_responses, rag_topics)
        rag_responses = self.filter_non_topic_responses(rag_responses, rag_topic_dict.keys())

        extraction_data: List[PrefNuggetData]
        borda_scores: Dict[str, int]
        if not random_pairs:
            # Step 1: Run pairwise preference comparisons
            result = run_preference_phase(
                rag_topic_dict=rag_topic_dict,
                rag_response_by_topic=rag_response_by_topic,
                llm_config=llm_config,
                num_pivot=num_pivot,
                num_others=num_others,
                no_dupes=no_dupes,
                pref_judge=pref_judge,
            )
            if result is None:
                return None
            _, aggregates = result

            # Step 2: Extract comparison pairs from aggregates
            extraction_data = extract_winner_loser_pairs(aggregates, rag_responses=rag_responses, rag_topic_dict=rag_topic_dict)
            # Build borda_scores lookup for prioritizing best-performing responses
            borda_scores = {key: agg.borda_score for key, agg in aggregates.items()}
        else:
            extraction_data = extract_random_pairs(rag_responses=rag_responses, rag_topic_dict=rag_topic_dict)
            borda_scores = dict()
            
        # Initialize given_exam_questions for iterative extraction (filled in per-chunk later)
        for data in extraction_data:
            data.given_exam_questions = []

        if not extraction_data:
            print("PrefNuggetJudge: No winner/loser pairs found after comparison")
            return None
        # Debug: print extraction pairs for reproducibility debugging
        for i, ed in enumerate(extraction_data[:20]):  # First 20
            print(f"  [{i}] {ed.query_id}: {ed.winner_run_id} > {ed.loser_run_id}")

        print(f"PrefNuggetJudge: Extracting nuggets from {len(extraction_data)} comparison pairs...")

        # Output converter (JSON parsing handled by TolerantChatAdapter)
        def convert_output(
            prediction: dspy.Prediction, data: PrefNuggetData
        ) -> None:
            differentiating_questions = getattr(prediction, "differentiating_questions", [])
            # Normalize: strip whitespace, filter empty
            data.differentiating_questions = [
                q.strip() for q in (differentiating_questions or [])
                if q and q.strip()
            ][:max_questions_per_pair]

        if iterative_nuggets:

            # Spread out elements with same query_id (round-robin interleave)
            # Within each topic, pairs are sorted by winner's borda_score (best performers first)
            extraction_data_chunks = _chunk_by_query(
                extraction_data,
                borda_scores=borda_scores,
                nugget_gen_order="as_provided" if random_pairs else nugget_gen_order,
                num_per_query=gen_batch_num_per_query,
                max_pairs_considered=max_pairs_considered,
            )
            tracker = QuestionTracker()

            extraction_result_data = list()
            
            
            ### This is special for `iterative_nuggets = True`
            for chunk_idx, extraction_chunk in enumerate(extraction_data_chunks):
                # Skip prompts for topics that already have enough questions
                extraction_chunk = [d for d in extraction_chunk if not tracker.is_done(d.query_id)]

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

                for data in extraction_chunk:
                    # Add questions (count increments for duplicates - tracks reuse)
                    tracker.add_all(data.query_id, data.differentiating_questions)

                # Mark topics as done if they've collected enough questions
                tracker.check_all_done(stop_at_count=stop_collecting_at_nuggets_per_topic)

                extraction_result_data.extend(extraction_chunk)

                print(f"-- PrefNuggetJudge: Finished extracting nuggets pass {chunk_idx}. Questions:\n{_print_tracker(tracker)}")

            print("PrefNuggetJudge: Finished extracting nuggets")
            print(f"Question counts: {dict(tracker.items())}")

        else:
            # Non-iterative: single batch extraction
            extraction_result_data = run_dspy_batch_generic(
                extraction_data,
                ExtractDifferentiatingNuggets,
                convert_output,
                llm_config,
            )
            print("PrefNuggetJudge: Finished extracting nuggets")
        
        
        # Build NuggetBanks from results (max_per_topic limits final count)
        questions_by_topic = _flatten_extraction_results(extraction_result_data, rag_topic_dict)
        return build_nugget_banks(questions_by_topic, max_per_topic=max_nuggets_per_topic)

    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        grade_threshold: int = 4,
        on_missing_evals: str = "fix_aggregate",
        grade_text: Literal["response", "document", "document_paragraphs"] = "response",
        filebase: str = "prefnugget",
        **kwargs
    ) -> Leaderboard:
        """
        Grade each response against all nuggets for its topic.

        Uses shared rubric utilities for grading and aggregation.
        """
        if nugget_banks is None:
            raise ValueError("PrefNuggetJudge requires nugget_banks. Run create_nuggets first or provide --nugget-banks.")

        self.expected_topic_ids = [t.request_id for t in rag_topics]

        # Prepare grading data using shared utility
        print("PrefNuggetJudge: Preparing grade data...")
        # ToDo if flag then use `prepare_nugget_grade_data_for_documents` instead
        grade_data, nuggets_per_topic = prepare_nugget_grade_data(rag_responses, nugget_banks)  if grade_text == "response" else     prepare_nugget_grade_data_for_documents(rag_responses, nugget_banks, use_paragraphs = grade_text == "document_paragraphs")

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
        # For document/paragraph grading, take max grade per nugget across all docs/paragraphs first
        if grade_text == "response":
            aggregates = compute_nugget_aggregates(grade_data, nuggets_per_topic, grade_threshold)
        else:
            aggregates = compute_nugget_aggregates_for_documents(grade_data, nuggets_per_topic, grade_threshold)

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

        # # Build qrels from grade data
        # if grade_text == "response":
        #     qrels = build_qrels(records=grade_data, spec=PREFNUGGET_QRELS) if grade_data else None
        # else:
        #     qrels = build_qrels(records=grade_data, spec=PREFNUGGET_CITE_QRELS) if grade_data else None
        # if qrels is not None:
        #     qrels.verify(warn=True, expected_topic_ids=self.expected_topic_ids)

        return leaderboard

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