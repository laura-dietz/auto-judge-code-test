#!/usr/bin/env python3
"""
Prefernece-based AutoJudge that:
1. Compares two passages to decide which one is better
2. Applies a reverse comparison to avoid position bias
3. Computes the transitive closure + Borda count to obtain ranking
This judge does not use nuggets.
"""
from itertools import groupby
from typing import Dict, List, Optional, Sequence, Type

from trec_auto_judge import (
    AutoJudge,
    LeaderboardSpec,
    MeasureSpec,
    Leaderboard,
    LeaderboardBuilder,
    MinimaLlmConfig,
    NuggetBanks,
    NuggetBanksProtocol,
    Qrels,
    Report,
    Request,
    auto_judge_to_click_command,
    mean_of_floats,
    mean_of_ints,
)

# Import shared preference utilities
from trec25.judges.shared.pref_common import (
    PrefAggregateResult,
    PrefJudgeData,
    compute_pref_aggregates,
    prepare_prompts,
    run_pref_judgment_batch,
)


# =============================================================================
# Leaderboard & Response Export (judge-specific)
# =============================================================================


PREF_SPEC = LeaderboardSpec(
    measures=(
        MeasureSpec("BORDA_COUNT", aggregate=mean_of_ints, cast=float, default=0.0),
        MeasureSpec("WIN_FRAC", aggregate=mean_of_floats, cast=float, default=0.0),
    )
)


def build_pref_leaderboard(
    aggregates: Dict[str, PrefAggregateResult],
) -> LeaderboardBuilder:
    """Build LeaderboardBuilder from preference aggregates."""
    b = LeaderboardBuilder(PREF_SPEC)
    for agg in aggregates.values():
        b.add(
            run_id=agg.run_id,
            topic_id=agg.topic_id,
            values={"BORDA_COUNT": agg.borda_score, "WIN_FRAC": agg.win_frac},
        )
    return b


def update_response_evaldata(
    rag_response_by_topic: Dict[str, List[Report]],
    aggregates: Dict[str, PrefAggregateResult],
) -> None:
    """Update Report.evaldata with preference aggregates."""
    for topic_id, responses in rag_response_by_topic.items():
        for response in responses:
            response_key = f"{response.metadata.run_id}:{response.metadata.topic_id}"
            if response_key in aggregates:
                agg = aggregates[response_key]
                response.evaldata = {
                    "BORDA_COUNT": agg.borda_score,
                    "WIN_FRAC": agg.win_frac,
                    "better_than": agg.better_than,
                    "worse_than": agg.worse_than,
                }


def read_results(
    rag_response_by_topic: Dict[str, List[Report]],
    grade_data: List[PrefJudgeData],
) -> LeaderboardBuilder:
    """
    Aggregate pairwise preferences into Borda count scores.

    Updates Report.evaldata and returns LeaderboardBuilder.
    """
    aggregates = compute_pref_aggregates(grade_data)
    update_response_evaldata(rag_response_by_topic, aggregates)
    return build_pref_leaderboard(aggregates)


class PrefJudge(AutoJudge):
    """
    Preference-based judge that:
    1. Determines which of two responses is better
    2. Decodes preferences into a ranking via Borda Count
    3. Computes Borda Count as an evaluation score
    """

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks  # Does not matter

    def __init__(self):
        # self.on_missing_evals: OnMissing = "fix_aggregate"
        pass

    def create_nuggets(self, **args) -> Optional[NuggetBanksProtocol]:
        return None   # We are not using nuggets

    def create_qrels(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> Optional[Qrels]:
        """PrefJudge does not produce qrels."""
        return None
    
    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        num_others: int,
        num_pivot: int,
        on_missing_evals: str,
        no_dupes: bool = False,
        **kwargs
    ) -> Leaderboard:
        num_runs = len({r.metadata.run_id for r in rag_responses})
        expected_topic_ids = [t.request_id for t in rag_topics]

        for resp in rag_responses:
            if resp.metadata.topic_id is None:
                print(f"Invalid reponse: {resp.metadata.run_id}")

        # Hash topics
        rag_topic_dict: Dict[str, Request] = {r.request_id: r for r in rag_topics}
        rag_response_by_topic: Dict[str, List[Report]] = {
            topic: list(responses)
            for topic, responses in groupby(
                sorted(rag_responses, key=lambda r: r.metadata.topic_id),
                key=lambda r: r.metadata.topic_id
            )
        }

        print(f"rag_response_by_topic: {len(rag_response_by_topic)} entries, keys: {rag_response_by_topic.keys()}")

        grade_data = prepare_prompts(rag_topic_dict=rag_topic_dict
                                     , rag_response_by_topic=rag_response_by_topic
                                     , num_pivot=num_pivot
                                     , num_others=num_others
                                     , no_dupes=no_dupes)

        # Run LLM grading
        print("PrefJudge: Grading responses...")
        grade_data = run_pref_judgment_batch(grade_data, llm_config)
        print("PrefJudge: Finished grading")

        # Include pairs also in reverse (p1 <-> p2)
        grade_data = grade_data + [data.flip() for data in grade_data]

        # this changed reports
        b = read_results(rag_response_by_topic=rag_response_by_topic
                                        , grade_data=grade_data)

        leaderboard = b.build(expected_topic_ids=expected_topic_ids, on_missing = on_missing_evals)
        leaderboard.verify(expected_topic_ids=expected_topic_ids, warn=False, on_missing = on_missing_evals)
        return leaderboard


if __name__ == '__main__':
    auto_judge_to_click_command(PrefJudge(), "pref_judge")()