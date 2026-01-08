#!/usr/bin/env python3
"""
Prefernece-based AutoJudge that:
1. Compares two passages to decide which one is better
2. Applies a reverse comparison to avoid position bias
3. Computes the transitive closure + Borda count to obtain ranking
This judge does not use nuggets.
"""
from math import gcd
from textwrap import dedent
from trec_auto_judge import *

import dspy
import asyncio
import re
from typing import *
from pydantic import BaseModel
from itertools import groupby

from trec_auto_judge.llm.minima_llm_dspy import run_dspy_batch
from trec_auto_judge import MinimaLlmConfig, OpenAIMinimaLlm
from trec_auto_judge.leaderboard.leaderboard import OnMissing


# =============================================================================
# Data Models (combined input/output)
# =============================================================================

class PrefJudgeData(BaseModel):
    run_id:str
    run_id2:str
    query_id:str
    query_title:str
    query_problem:str=""
    query_background:str=""
    passage_1:str
    passage_2:str
    better_passage:Optional[int] = None
    confidence:Optional[float] = None
    reasoning:Optional[str] = None
    

        
    def _swap(self, better_passage:Optional[int])-> Optional[int]:
        '''for reversing passage 1<->2 '''
        if better_passage is None:
            return None
        if better_passage == 1:
            return 2
        if better_passage == 2:
            return 1
        else:
            return better_passage
                
    def flip(self):
        return PrefJudgeData (run_id = self.run_id2
                         , run_id2 = self.run_id
                         , query_id = self.query_id
                         , query_title = self.query_title
                         , query_problem = self.query_problem
                         , query_background = self.query_background
                         , passage_1 = self.passage_2
                         , passage_2 = self.passage_1
                         , better_passage = self._swap(self.better_passage)
                         , confidence = self.confidence
                         , reasoning = self.reasoning
                         )



# =============================================================================
# DSPy Signatures and Output Conversion
# =============================================================================

def _parse_better(s: str) -> int:
    """Extract passage 1-2 from string."""
    m = re.search(r'\b([1-2])\b', s)
    if not m:
        return 0  # Default to 0 if no valid preference is found
    return int(m.group(1))




class PrefJudgment(dspy.Signature):
    '''
    Pairwise preference judgments.
    Mostly following Prompt of Arabzadeh & Clarke, except asking for query instead of question.
    Source: https://github.com/Narabzad/llm-relevance-judgement-comparison/blob/main/Pref/judge.py
    '''
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


    passage_1:str = dspy.InputField(des="passage 1")
    passage_2:str = dspy.InputField(des="passage 2")
    
    better_passage: Literal[ "1", "2"] = dspy.OutputField(
        desc="which is the better passage?"
    )

    @classmethod
    def convert_prompt_output(cls, prediction: dspy.Prediction, data: PrefJudgeData) -> None:
        '''Convert Prompt output to PrefJudgeData'''
        data.better_passage = _parse_better(prediction.better_passage)
        data.confidence = prediction.confidence or 0.0  if hasattr(prediction, "confidence") else 0.0
        data.reasoning = prediction.reasoning or "" if hasattr(prediction, "reasoning") else ""
        
        return



# =============================================================================
# Leaderboard & Qrels Specs
# =============================================================================

PREF_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("BORDA_COUNT", aggregate=mean_of_floats, cast=float, default=0.0),
))



# =============================================================================
# Sampling Functions
# =============================================================================

def check_sampling_coprimality(
    num_responses: int,
    num_pivot: int,
    num_others: int,
) -> None:
    """
    Warn if sampling parameters may cause symmetric pair generation.

    For non-pivot responses, the rotated list excludes self, so its length
    is (num_responses - num_pivot - 1). If stride and this length share a
    common factor, the sampling pattern may generate both (A,B) and (B,A).
    """
    non_pivot_count = num_responses - num_pivot - 1
    if non_pivot_count <= 0:
        return

    stride = max(1, non_pivot_count // num_others)

    if gcd(stride, non_pivot_count) != 1:
        print(
            f"Warning: stride {stride} and non-pivot count {non_pivot_count} "
            f"are not coprime (gcd={gcd(stride, non_pivot_count)}). "
            f"This may cause duplicate pairings."
        )


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

    # Rotate non_pivots around current idx (skip self if in non_pivots)
    if idx < num_pivot:
        rotated = non_pivots
    else:
        adj_idx = idx - num_pivot
        rotated = non_pivots[adj_idx+1:] + non_pivots[:adj_idx]

    stride = max(1, len(rotated) // num_others) if rotated else 1

    return list(pivots) + rotated[::stride][:num_others]



# =============================================================================
# PrefJudge Implementation
# =============================================================================


class PrefJudge(AutoJudge):
    """
    Preference-based judge that:
    1. Determines which of two responses is better
    2. Decodes preferences into a ranking via Borda Count
    3. Computes Borda Count as an evaluation score
    """

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks  # Does not matter
    
    def __init__(self):
        self.expected_topic_ids:Sequence[str] = []
        self.on_missing_evals: OnMissing = "fix_aggregate"

    def create_nuggets(self, **args) -> Optional[NuggetBanks]:
        return None   # We are not using nuggets
    
    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        num_others: int,
        num_pivot: int,
        **kwargs
    ) -> tuple[Leaderboard, Optional[Qrels]]:

        check_sampling_coprimality(len(rag_responses), num_pivot, num_others)

        for resp in rag_responses:
            if resp.metadata.topic_id is None:
                print(f"Invalid reponse: {resp.metadata.run_id}")
        
        rag_topic_dict:Dict[str,Request] = {r.request_id:r  for r in rag_topics}
        # first group rag_responses by topic.
        # rag_response_by_topic:Dict[str,List[Report]] = {topic: responses 
        #                                                     for topic, responses 
        #                                                     in  groupby(sorted(rag_responses, key=lambda r: r.metadata.topic_id)
        #                                                                 , key=lambda r: r.metadata.topic_id)
        #                                                 }

        rag_response_by_topic: Dict[str, List[Report]] = {
            topic: list(responses)
            for topic, responses in groupby(
                sorted(rag_responses, key=lambda r: r.metadata.topic_id),
                key=lambda r: r.metadata.topic_id
            )
        }
        
        
        print(f"rag_response_by_topic: {len(rag_response_by_topic)} entries, keys: {rag_response_by_topic.keys()}")
        
        grade_data: List[PrefJudgeData] = []
        self.expected_topic_ids=[t.request_id for t in rag_topics]

        for topic_id, responses in rag_response_by_topic.items():
            request = rag_topic_dict[topic_id]
            for idx, response in enumerate(responses):
                run_id = response.metadata.run_id
                text = response.get_report_text()

                # Select comparison samples (pivots + strided non-pivots)
                for response_other in select_comparison_samples(responses, idx, num_pivot, num_others):
                    run_id_other = response_other.metadata.run_id
                    if run_id_other != run_id:  # skip self
                        data = PrefJudgeData(
                                run_id=run_id,
                                query_id=topic_id,
                                passage_1=text,
                                run_id2=run_id_other,
                                passage_2=response_other.get_report_text(),
                                #
                                query_title = request.title or "",
                                query_problem = request.problem_statement or "",
                                query_background = request.background or "",
                        )
                        grade_data.append(data)




        # Run LLM grading
        print(f"PrefJudge: Grading responses...")
        if grade_data:
            grade_data = asyncio.run(run_dspy_batch(
                PrefJudgment,
                grade_data,
                PrefJudgment.convert_prompt_output,
                backend=OpenAIMinimaLlm(llm_config)
            ))
        print(f"PrefJudge: Finished grading")



        b = LeaderboardBuilder(PREF_SPEC)

        # todo add passage_1 <-> passage_2  flipped
            
        # Update Report.evaldata
        
        # Include pairs also in reverse (p1 <-> p2)
        grade_data = grade_data + [data.flip() for data in grade_data]
        
        data_by_key =  {k: list(g) for k, g in groupby(sorted(grade_data, key=lambda data: f"{data.run_id}:{data.query_id}")
                              , key=lambda data: f"{data.run_id}:{data.query_id}")}
        
        
        for topic_id, responses in rag_response_by_topic.items():
            for response in responses:
                response_key = f"{response.metadata.run_id}:{response.metadata.topic_id}"
                if response_key in data_by_key:
                    pref_data_list  = data_by_key[response_key]
                    borda_score = sum(1 if data.better_passage ==1 else -1 for data in pref_data_list) # wins - losses
                    
                    b.add(
                        run_id=response.metadata.run_id,
                        topic_id=topic_id,
                        values={
                            "BORDA_COUNT": borda_score
                        }
                    )
        leaderboard = b.build(expected_topic_ids=self.expected_topic_ids, on_missing = self.on_missing_evals)
        leaderboard.verify(expected_topic_ids=self.expected_topic_ids, warn=False, on_missing = self.on_missing_evals)
        return (leaderboard, None)


if __name__ == '__main__':
    auto_judge_to_click_command(PrefJudge(), "pref_judge")()