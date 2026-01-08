#!/usr/bin/env python3
"""
Prefernece-based AutoJudge that:
1. Compares two passages to decide which one is better
2. Applies a reverse comparison to avoid position bias
3. Computes the transitive closure + Borda count to obtain ranking
This judge does not use nuggets.
"""
import random
from textwrap import dedent
from trec_auto_judge import *

import dspy
import asyncio
import re
import json
from typing import *
from pydantic import BaseModel
from itertools import groupby

from trec_auto_judge.llm.minima_llm_dspy import run_dspy_batch
from trec_auto_judge import MinimaLlmConfig, OpenAIMinimaLlm
from trec_auto_judge.leaderboard.leaderboard import OnMissing




# =============================================================================
# DSPy Signatures
# =============================================================================


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

#     confidence : Optional[float] = dspy.OutputField(
#         required=False,          # header may be omitted
#         default=None,
#         transform=_num_or_none   # runs *before* ChatAdapter hands to float()
#     )
#     reasoning: Optional[str] = dspy.OutputField(desc="reasoning", default=None, required=False)

   
    @classmethod
    def convert_output(cls,prediction: dspy.Prediction, alignment: BaseModel)->None:
        
        def parse_1_to_2(s: str) -> int:
            m = re.search(r'\b([1-2])\b', s)
            if not m:
                raise ValueError(f"No integer 1–2 found in response: {s!r}")
            return int(m.group(1))

        alignment.confidence = (prediction.confidence) or 0.0
        alignment.reasoning = prediction.reasoning
        alignment.better_passage = int(parse_1_to_2(prediction.better_passage))

        # alignment.is_match = (alignment.answerability>=2)
        # alignment.match_score = float(alignment.answerability)

        return


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
    

# =============================================================================
# Leaderboard & Qrels Specs
# =============================================================================

PREF_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("BORDA_COUNT", aggregate=mean_of_floats, cast=float, default=0.0),
))



# =============================================================================
# Conversion Functions
# =============================================================================

def _parse_better(s: str) -> int:
    """Extract passage 1-2 from string."""
    m = re.search(r'\b([1-2])\b', s)
    if not m:
        return 0  # Default to 0 if no valid preference is found
    return int(m.group(1))




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
        **kwargs
    ) -> tuple[Leaderboard, Optional[Qrels]]:
        
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
        response_nugget_map: Dict[str, List[PrefJudgeData]] = {}  # run_id:topic_id -> data list
        self.expected_topic_ids=[t.request_id for t in rag_topics]
        
        for topic_id, responses in rag_response_by_topic.items():
            request = rag_topic_dict[topic_id]
            for idx, response in enumerate(responses):
                metadata = response.metadata
                run_id = metadata.run_id
                text = response.get_report_text()

                response_key = f"{run_id}:{topic_id}"
                response_nugget_map[response_key] = []

                # Create pref data for random other passages
                # stride = max(1, len(responses) // num_others)      
                rotated = responses[idx+1:] + responses[:idx]  # skip self, wrap around
                stride = max(1, len(rotated) // num_others)
                for response_other in responses[0:2] + rotated[::stride][:num_others]:
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
                        response_nugget_map[response_key].append(data)



        # Convert output handler
        def convert_grade_output(prediction: dspy.Prediction, data: PrefJudgeData) -> None:
            data.better_passage = _parse_better(prediction.better_passage)
            data.reasoning = getattr(prediction, 'reasoning', None)
            data.confidence = getattr(prediction, 'confidence', None)


        # Run LLM grading
        print(f"Rubric: Grading responses...")
        if grade_data:
            grade_data = asyncio.run(run_dspy_batch(
                PrefJudgment,
                grade_data,
                convert_grade_output,
                backend=OpenAIMinimaLlm(llm_config)
            ))
        print(f"PrefJudge: Finished grading")



        b = LeaderboardBuilder(PREF_SPEC)

        # todo add passage_1 <-> passage_2  flipped
            
        # Update Report.evaldata
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


# def main(rag_responses: list[dict], rag_topics: List[Request], output: Path):
#     """
#     A naive rag response assessor that just orders each response by its length.
#     """
#     sample_k = 2
#     topic_dict = {request.request_id: request for request in rag_topics}

#     def prepare_prompts()->List[PrefJudgAnnotation]:
#         alignment_input_list = list()
        
        
#         responses_per_topic = defaultdict(list)
#         for rag_response in rag_responses:
#             metadata = rag_response["metadata"]
#             topic_id = metadata["narrative_id"]

#             responses_per_topic[topic_id].append(rag_response)
        
#         for responses in responses_per_topic.values():

#             metadata = rag_response["metadata"]
#             run_id = metadata["run_id"]
#             topic_id = metadata["narrative_id"]
                        
#             topic = topic_dict[topic_id]
#             if topic is None:
#                 raise RuntimeError("Could not identify request object for topic {topic_id}")
            
#             if (topic.title is None) or (topic.background is None) or  (topic.problem_statement is None):
#                 raise RuntimeError(f"Missing fields in report request: title {topic.title}, background:{topic.background}, problem_statement: {topic.problem_statement}.")
            
#             passage2list = np.random.choice(np.array(responses), size=sample_k+1, replace=False)
#             passage2list=[p for p in passage2list if p["metadata"]["run_id"] is not run_id][0:2] # do not draw this passage
            
            
#             text_1 = " ".join([i["text"] for i in rag_response["answer"]])
            
#             for response_2 in passage2list:
                
#                 text_2 = " ".join([i["text"] for i in rag_response["answer"]])
#                 prompt_objs = PrefJudgAnnotation(query_id = topic_id
#                                                     , run_id = run_id
#                                                     , run_id2 = response_2["metadata"]["run_id"]
#                                                     ,passage_1 = text_1
#                                                     ,passage_2 = text_2
#                                                     ,metadata= metadata
#                                                     ,title_query = topic.title
#                                                     ,background = topic.background
#                                                     ,problem_statement =  topic.problem_statement
#                                                     )
#                 alignment_input_list.append(prompt_objs)
#                 prompt_objs_rev = PrefJudgAnnotation(query_id = topic_id
#                                                     , run_id = response_2["metadata"]["run_id"]
#                                                     , run_id2 = run_id
#                                                     ,passage_1 = text_2
#                                                     ,passage_2 = text_1
#                                                     ,metadata= metadata
#                                                     ,title_query = topic.title
#                                                     ,background = topic.background
#                                                     ,problem_statement =  topic.problem_statement
#                                                     )
#                 alignment_input_list.append(prompt_objs_rev)
#         return alignment_input_list

#     def topo_sort(topic:str, prompt_output:List[PrefJudgment]):
#         is_directly_above=defaultdict(set) # lists a set of run_ids that are above the key.
#         is_directly_below=defaultdict(set) # lists a set of run_ids that are above the key.
        
#         # build directed graph
#         for prompt in prompt_output:
#             if prompt.run_id != prompt.run_id2: # Todo we should not be adding self-references
#                 if prompt.better_passage == 1:
#                     is_directly_above[prompt.run_id2].add(prompt.run_id)
#                     is_directly_below[prompt.run_id].add(prompt.run_id2)
#                 elif prompt.better_passage == 2:
#                     is_directly_above[prompt.run_id].add(prompt.run_id2)
#                     is_directly_below[prompt.run_id2].add(prompt.run_id)


#         # transitive closure
#         is_indirectly_above = defaultdict(set)
#         count = 100
#         for k in list(is_directly_below.keys()):
#             count -=1
#             vs = is_directly_below[k]
#             below = set(vs)
#             if count>0: print(f"topic: {topic} key: {k} below{below}")
            
#             # we will walk down all nodes that are below v,
#             # keep adding those children to the below list,
#             # and make sure to track that k is above them all
            
#             while len(below)>0: # will be false if empty
#                 v = below.pop()
#                 if count>0: print(f"     pop {v}")
                
#                 is_indirectly_above[v].add(k)
#                 # below.update(is_directly_below[v])  # endless loop
#                 for vv in is_directly_below[v]:
#                     if k in is_indirectly_above[vv]:
#                         pass
#                     else:
#                         below.add(vv)
#                 if count>0: print(f"     below={below}")

#         # borda count
#         borda_count = [(k, len(vs)) for k, vs in is_indirectly_above.items()]
#         borda_count = sorted(borda_count, key=lambda t: t[1])
#         return borda_count
            

#     # TOD: we are missing at least one run_id (probably the lowest one)
#     def write_results(prompt_output:List[PrefJudgAnnotation]):
#         ret = []
#         avg_grades = defaultdict(list)

#         output_by_topic = defaultdict(list)
#         for out in prompt_output:
#             output_by_topic[out.query_id].append(out)
            
#         for topic_id,res_list in output_by_topic.items():

#             borda_count = topo_sort(topic=topic_id, prompt_output=res_list)
#             num_runs = len(borda_count)

#             for (run_id, num_above) in borda_count:
#                 system_score = float(num_runs-num_above)
#                 ret.append(f"{run_id} PREF {topic_id} {system_score}")
#                 avg_grades[run_id].append(system_score)
            
#         ret.append(f"{run_id} PREF all {mean([float(g) for g in avg_grades[run_id]])}")

#         output.parent.mkdir(exist_ok=True, parents=True)
#         output.write_text("\n".join(ret))

#     prompt_input = prepare_prompts()
#     print("Debug in", "\n".join(str(p) for p in prompt_input[0:4]))
    
#     prompt_output =  evaluator_run(prompt=PrefJudgment, output_converter=PrefJudgment.convert_output,alignment_input_list=prompt_input)
#     print("Debug out", "\n".join(str(p) for p in prompt_input[0:4]))

#     write_results(prompt_output=prompt_output)

# if __name__ == '__main__':
#     main()


if __name__ == '__main__':
    auto_judge_to_click_command(PrefJudge(), "pref_judge")()