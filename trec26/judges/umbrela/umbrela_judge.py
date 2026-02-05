#!/usr/bin/env python3
"""
UmbrelaJudge: Minimal UMBRELA implementation.

Grades RAG responses using the UMBRELA prompting framework.
"""
from typing import Sequence, Optional, Type, Literal
from pydantic import BaseModel
import dspy
from textwrap import dedent
import re
import asyncio

from trec_auto_judge import (
    AutoJudge,
    Report,
    Request,
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    MeasureSpec,
    Qrels,
    QrelsSpec,
    MinimaLlmConfig,
    OpenAIMinimaLlm,
    build_qrels,
    doc_id_md5,
)
from trec_auto_judge.leaderboard.leaderboard import mean_of_floats, mean_of_bools
from trec_auto_judge.nugget_data import NuggetBanks, NuggetBanksProtocol
from trec_auto_judge.llm.minima_llm_dspy import run_dspy_batch



class UmbrelaGrade(BaseModel):
    run_id: str
    topic_id: str
    passage: str
    query: str


    # LLM outputs (filled after grading)
    grade: Optional[int] = None  # 0-3
    reasoning: Optional[str] = None # Optional --  UMBRELA does not have reasoning


# UMBRELA DSPy Signature

def _parse_0_to_3(s: str) -> int:
    """Extract first 0-3 integer from string."""
    m = re.search(r'\b([0-3])\b', s)
    if not m:
        raise ValueError(f"No 0-3 score found in: {s!r}")
    return int(m.group(1))


class UmbrelaPrompt(dspy.Signature):
    """UMBRELA prompting framework for passage grading."""

    __doc__ = dedent("""
        Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:
        0 = represent that the passage has nothing to do with the query, 
        1 = represents that the passage seems related to the query but does not answer it, 
        2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and 
        3 = represents that the passage is dedicated to the query and contains the exact answer.

        Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.

        Query: {query}
        Passage: {passage}

        Split this problem into steps:
        Consider the underlying intent of the search.
        Measure how well the content matches a likely intent of the query (M).
        Measure how trustworthy the passage is (T).
        Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.
        Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning.
    """)

    query: str = dspy.InputField(desc="The user's query")
    passage: str = dspy.InputField(desc="The passage to grade")

    final_score: Literal["0", "1", "2", "3"] = dspy.OutputField(
        desc="Final grade O (0-3)"
    )
    reasoning: Optional[str] = dspy.OutputField( # Optional --  UMBRELA does not have reasoning
        desc="Brief explanation",
        required=False,
        default=None
    )

    @classmethod
    def convert_output(cls, prediction: dspy.Prediction, alignment: BaseModel) -> None:
        """Convert DSPy prediction to UmbrelaGrade."""
        alignment.grade = _parse_0_to_3(prediction.final_score)
        alignment.reasoning = prediction.reasoning


# Leaderboard & Qrels Specs

UMBRELA_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("AVG_GRADE", aggregate=mean_of_floats, cast=float, default=0.0),
    MeasureSpec("IS_RELEVANT", aggregate=mean_of_bools, cast=bool, default=False),
))

UMBRELA_QRELS = QrelsSpec[UmbrelaGrade](
    topic_id=lambda r: r.topic_id,
    doc_id=lambda r: doc_id_md5(r.passage),  # Hash passage as doc_id
    grade=lambda r: float(r.grade),
    on_duplicate="keep_max"
)



# Implement the Judge

class UmbrelaJudge(AutoJudge):
    """UMBRELA-based judge: grades passages, then aggregates to leaderboard."""

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks
    print(f"nugget_banks_type:{nugget_banks_type}")

    def __init__(self, settings: Optional[dict] = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        # Topic format: auto-detect or explicit
        self.topic_format = settings.get("topic_format", "auto") if settings else "auto"

    def extract_query(self, topic) -> str:
        """
        Extract query text from topic based on format.
        Supports: RAGTIME (title+problem+background), DRAGUN (body), RAG (title), auto-detect
        """
        # Explicit format override
        if self.topic_format == "dragun":
            return getattr(topic, "body", topic.title)
        elif self.topic_format == "rag":
            return topic.title
        elif self.topic_format == "ragtime":
            parts = [topic.title, getattr(topic, "problem_statement", ""), getattr(topic, "background", "")]
            return " ".join(p for p in parts if p).strip()

        # Auto-detect format
        if hasattr(topic, 'body') and topic.body:  # DRAGUN
            return topic.body
        elif hasattr(topic, 'problem_statement'):  # RAGTIME
            parts = [topic.title, getattr(topic, "problem_statement", ""), getattr(topic, "background", "")]
            return " ".join(p for p in parts if p).strip()
        else:  # RAG (simple title-only)
            return topic.title

    def create_qrels(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> Optional[Qrels]:
        """Grade each passage using UMBRELA."""

        # Build topic lookup
        topic_dict = {req.request_id: req for req in rag_topics}

        # Prepare grading inputs
        grades = []
        for response in rag_responses:
            topic = topic_dict.get(response.metadata.topic_id)
            if not topic:
                continue

            query = self.extract_query(topic)

            grades.append(UmbrelaGrade(
                run_id=response.metadata.run_id,
                topic_id=response.metadata.topic_id,
                passage=response.get_report_text(),
                query=query,
            ))


        print(f"Grading {len(grades)} passages with UMBRELA...")

        # Run UMBRELA grading via DSPy
        graded = asyncio.run(run_dspy_batch(
            UmbrelaPrompt,
            grades,
            UmbrelaPrompt.convert_output,
            backend=OpenAIMinimaLlm(llm_config)
        ))

        # Convert to qrels
        qrels = build_qrels(records=graded, spec=UMBRELA_QRELS)
        print(f"Created qrels with {len(qrels.rows)} entries")

        return qrels

    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        relevance_threshold: int = 2,  # Grade >= 2 is "relevant"
        **kwargs
    ) -> Leaderboard:
        """Aggregate qrels grades into leaderboard."""

        if not qrels:
            raise ValueError("UmbrelaJudge requires qrels. Run create_qrels first.")

        # Build grade lookup: (topic_id, doc_id) -> grade
        grade_lookup = {
            (row.topic_id, row.doc_id): row.grade
            for row in qrels.rows
        }

        # Build leaderboard
        builder = LeaderboardBuilder(UMBRELA_SPEC)
        expected_topic_ids = [t.request_id for t in rag_topics]

        for response in rag_responses:
            doc_id = doc_id_md5(response.get_report_text())
            grade = grade_lookup.get(
                (response.metadata.topic_id, doc_id),
                0.0  # Default for missing entries
            )

            builder.add(
                run_id=response.metadata.run_id,
                topic_id=response.metadata.topic_id,
                AVG_GRADE=float(grade),
                IS_RELEVANT=(grade >= relevance_threshold)
            )

        leaderboard = builder.build(
            expected_topic_ids=expected_topic_ids,
            on_missing="fix_aggregate"
        )

        print(f"Built leaderboard with {len(leaderboard.entries)} entries")
        return leaderboard

    def create_nuggets(self, *args, **kwargs):
        """Not used by UMBRELA."""
        return None


# CLI Entry Point

if __name__ == "__main__":
    from trec_auto_judge import auto_judge_to_click_command

    judge = UmbrelaJudge()
    cli = auto_judge_to_click_command(judge, "simple-umbrela")
    cli()
