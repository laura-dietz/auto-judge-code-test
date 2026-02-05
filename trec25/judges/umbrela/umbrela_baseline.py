#!/usr/bin/env python3
import dspy
from textwrap import dedent
import re
import asyncio
from typing import Iterable, Literal, Optional, Sequence, Type

from pydantic import BaseModel

from autojudge_base import (
    AutoJudge,
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    LeaderboardVerification,
    MeasureSpec,
    NuggetBanksProtocol,
    Qrels,
    QrelsSpec,
    Report,
    Request,
    auto_judge_to_click_command,
    build_qrels,
    doc_id_md5,
)
from autojudge_base.nugget_data import NuggetBanks
from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm
from minima_llm.dspy_adapter import run_dspy_batch


class UmbrelaAnnotation(BaseModel):
    run_id:str
    query_id:str
    # title_query:str
    source_document:str
    # problem_statement:str=""
    # background:str=""
    query:str=""
    confidence:Optional[float] = None
    reasoning:Optional[str] = None
    answerability:Optional[str] = None
    is_match:Optional[bool] = None
    match_score:Optional[float] = None




def _num_or_none(raw: str | None) -> Optional[float]:
    """
    * If `raw` contains a number (first numeric token) ➜ float(token)
    * else ➜ None
    """
    if raw is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", raw)
    return float(m.group()) if m else None



class Umbrela(dspy.Signature):
    __doc__ = dedent(
        """
        Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:
        0 = represent that the passage has nothing to do with the query,
        1 = represents that the passage seems related to the query but does not answer it,
        2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and
        3 = represents that the passage is dedicated to the query and contains the exact answer.

        Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely,
        category 2 if passage presents something very important related to the entire topic but also has some extra
        information and category 3 if the passage only and entirely refers to the topic.
        If none of the above satisfies give it category 0.

        Split this problem into steps: Consider the underlying intent of the search. Measure how well the content matches a likely intent of the query (M).
        Measure how trustworthy the passage is (T).
        Consider the aspects above and the relative importance of each, and decide on a final score (O).
        Final score must be an integer value only. Do not provide any code in result.
        Provide each score without providing any reasoning.
        """
    )

    query:str = dspy.InputField(desc="query")
    # title_query:str = dspy.InputField(desc="topic")
    # problem_statement:str =  dspy.InputField(desc="problem")
    # background:str =  dspy.InputField(desc="background")
    source_document:str = dspy.InputField(desc="passage")
    
    answerability: Literal["0", "1", "2", "3"] = dspy.OutputField(
        desc="final score O"
    )
    trust_worthy: Literal["0", "1", "2", "3"] = dspy.OutputField(
        desc="trustworthy score T"
    )
    intent: Literal["0", "1", "2", "3"] = dspy.OutputField(
        desc="intent score M"
    )

    confidence : Optional[float] = dspy.OutputField(
        required=False,          # header may be omitted
        default=None,
        transform=_num_or_none   # runs *before* ChatAdapter hands to float()
    )
    reasoning: Optional[str] = dspy.OutputField(desc="reasoning", default=None, required=False)

   
    @classmethod
    def convert_output(cls,prediction: dspy.Prediction, alignment: BaseModel)->None:
        # print("Umbrela convert_output prediction", prediction)
        
        
        def parse_0_to_3(s: str) -> int:
            m = re.search(r'\b([0-3])\b', s)
            if not m:
                raise ValueError(f"No integer 0–3 found in response: {s!r}")
            return int(m.group(1))

        alignment.confidence = (prediction.confidence) or 0.0
        alignment.reasoning = prediction.reasoning
        alignment.answerability = int(parse_0_to_3(prediction.answerability))

        alignment.is_match = (alignment.answerability>=2)
        alignment.match_score = float(alignment.answerability)

        return





UMBRELA_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("GRADE"),
    MeasureSpec("IS_MATCH"),  # 1.0/0.0 for boolean
))


UMBRELA_QRELS = QrelsSpec["UmbrelaAnnotation"](
    topic_id=lambda r: r.query_id,
    doc_id=lambda r: doc_id_md5(r.source_document),
    grade=lambda r: r.match_score,
    on_duplicate="error"
)


def umbrela_to_qrels(
    prompt_output: Iterable["UmbrelaAnnotation"]
) -> Qrels:
    qrels = build_qrels(records=prompt_output, spec=UMBRELA_QRELS)    
    return qrels


class UmbrelaJudge(AutoJudge):
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def create_nuggets(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence["Request"],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> Optional[NuggetBanksProtocol]:
        return None

    def create_qrels(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> Optional[Qrels]:
        """Run Umbrela grading and produce qrels."""
        topic_dict = {request.request_id: request for request in rag_topics}

        # Prepare input prompts
        alignment_input_list = []
        for rag_response in rag_responses:
            metadata = rag_response.metadata
            run_id = metadata.run_id
            topic_id = metadata.topic_id
            text = rag_response.get_report_text()

            topic = topic_dict[topic_id]
            if topic is None:
                raise RuntimeError(f"Could not identify request object for topic {topic_id}")

            query = (f"{topic.title} {topic.problem_statement} {topic.background}").strip()
            if not query:
                raise RuntimeError(f"Missing fields in report request: title {topic.title}, background:{topic.background}, problem_statement: {topic.problem_statement}.")

            prompt_obj = UmbrelaAnnotation(
                query_id=topic_id,
                run_id=run_id,
                source_document=text,
                query=query
            )
            alignment_input_list.append(prompt_obj)

        print("Debug in", "\n".join(str(p) for p in alignment_input_list[0:1]))

        # Execute with MinimaLLM backend using helper
        prompt_output = asyncio.run(run_dspy_batch(
            Umbrela,
            alignment_input_list,
            Umbrela.convert_output,
            backend=OpenAIMinimaLlm(llm_config)
        ))
        print("Debug out", "\n".join(str(p) for p in prompt_output[0:1]))

        return umbrela_to_qrels(prompt_output)

    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        qrels: Optional[Qrels],
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> Leaderboard:
        """
        Build leaderboard from qrels grades.
        """
        if qrels is None:
            raise ValueError("UmbrelaJudge requires qrels. Run create_qrels first.")

        # Build lookup: (topic_id, doc_id) -> grade
        grade_lookup = {(row.topic_id, row.doc_id): row.grade for row in qrels.rows}

        b = LeaderboardBuilder(UMBRELA_SPEC)
        for response in rag_responses:
            topic_id = response.metadata.topic_id
            run_id = response.metadata.run_id
            doc_id = doc_id_md5(response.get_report_text())

            grade = grade_lookup.get((topic_id, doc_id), 0.0)
            is_match = 1.0 if grade >= 2.0 else 0.0

            b.add(
                run_id=run_id,
                topic_id=topic_id,
                values={"GRADE": grade, "IS_MATCH": is_match},
            )

        leaderboard = b.build()
        LeaderboardVerification(leaderboard).complete_measures(include_all_row=False).same_topics_per_run()
        return leaderboard


if __name__ == '__main__':
    auto_judge_to_click_command(UmbrelaJudge(), "umbrela_baseline")()