#!/usr/bin/env python3
"""
DirectPromptJudge: Direct relevance assessment through prompting.

Uses LLM-based prompting for response quality evaluation:
- UMBRELA prompt for RAG and RAGTIME datasets (relevance grading)
- DRAGUN prompt for DRAGUN dataset (trustworthiness assessment)
"""
from typing import Sequence, Optional, Type, Literal
from pydantic import BaseModel
import dspy
from textwrap import dedent
import re
import asyncio
from pathlib import Path
import sys

# Add parent directory to path for debug_logger import
sys.path.insert(0, str(Path(__file__).parent.parent))
from debug_logger import DebugLogger

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


# UMBRELA DSPy Signature

def _parse_0_to_3(s: str) -> int:
    """Extract first 0-3 integer from string."""
    m = re.search(r'\b([0-3])\b', s)
    if not m:
        raise ValueError(f"No 0-3 score found in: {s!r}")
    return int(m.group(1))


def _convert_score_output(prediction: dspy.Prediction, alignment: BaseModel) -> None:
    """Shared convert_output for both UMBRELA and DRAGUN prompts."""
    alignment.grade = _parse_0_to_3(prediction.final_score)


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

    # Use shared conversion function
    convert_output = staticmethod(_convert_score_output)


# DRAGUN DSPy Signature (Trustworthiness Assessment)

class DragunPrompt(dspy.Signature):
    """DRAGUN trustworthiness assessment prompt."""

    __doc__ = dedent("""
        Given a news article and a generated report, evaluate how well the report helps assess the article's trustworthiness on a 0-3 scale:

        0 = No useful trustworthiness information (irrelevant or off-topic)
        1 = Somewhat related but doesn't meaningfully help assess trustworthiness (generic, lacks specifics)
        2 = Provides helpful trustworthiness information but has gaps (addresses 1-2 factors, incomplete)
        3 = Comprehensive trustworthiness assessment (addresses source bias/credibility, motivation, multi-source context, well-attributed)

        Key trustworthiness factors:
        - Source bias, credibility, or motivation
        - Multi-source context or alternative viewpoints
        - Proper attribution and citations
        - Helps readers form independent judgment

        News Article: {query}
        Generated Report: {passage}

        Evaluate the report's trustworthiness assessment quality and provide a final score (0-3). Format: ##final score: score
    """)

    query: str = dspy.InputField(desc="The news article text")
    passage: str = dspy.InputField(desc="The generated report to evaluate")

    final_score: Literal["0", "1", "2", "3"] = dspy.OutputField(
        desc="Final trustworthiness assessment score (0-3)"
    )

    # Use shared conversion function
    convert_output = staticmethod(_convert_score_output)


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

class DirectPromptJudge(AutoJudge):
    """Direct prompting judge: uses LLM prompts for relevance assessment, then aggregates to leaderboard."""

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def __init__(self, settings: Optional[dict] = None, debug_log: Optional[str] = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        # Topic format: auto-detect or explicit
        self.topic_format = settings.get("topic_format", "auto") if settings else "auto"
        # Debug logger
        self.debug_logger = DebugLogger(debug_log)

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
            return ". ".join(p for p in parts if p).strip()

        # Auto-detect format
        if hasattr(topic, 'body') and topic.body:  # DRAGUN
            return topic.body
        elif hasattr(topic, 'problem_statement'):  # RAGTIME
            parts = [topic.title, getattr(topic, "problem_statement", ""), getattr(topic, "background", "")]
            return ". ".join(p for p in parts if p).strip()
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


        # Determine which prompt to use based on explicit topic_format setting
        # (same logic as extract_query: explicit first, then auto-detect)
        if self.topic_format == "dragun":
            # Explicit DRAGUN dataset
            prompt_class = DragunPrompt
            prompt_name = "DRAGUN"
        elif self.topic_format in ["rag", "ragtime"]:
            # Explicit RAG or RAGTIME dataset
            prompt_class = UmbrelaPrompt
            prompt_name = "UMBRELA"
        else:  # topic_format == "auto"
            # Auto-detect only when topic_format is "auto"
            # Check if any topic has 'body' field (DRAGUN indicator)
            has_body = any(hasattr(topic_dict.get(g.topic_id), 'body') and
                          topic_dict.get(g.topic_id).body
                          for g in grades if g.topic_id in topic_dict)
            if has_body:
                prompt_class = DragunPrompt
                prompt_name = "DRAGUN (auto-detected)"
            else:
                prompt_class = UmbrelaPrompt
                prompt_name = "UMBRELA (auto-detected)"

        print(f"Grading {len(grades)} passages with {prompt_name}...")

        # Log inputs if debug enabled
        if self.debug_logger.enabled:
            topic_lookup = topic_dict
            for grade in grades:
                topic = topic_lookup.get(grade.topic_id)
                # Build complete topic data
                topic_data = {
                    "topic_id": topic.request_id if topic else grade.topic_id,
                    "title": topic.title if topic else "(unknown)",
                }
                if hasattr(topic, 'problem_statement') and topic.problem_statement:
                    topic_data["problem_statement"] = topic.problem_statement
                if hasattr(topic, 'background') and topic.background:
                    topic_data["background"] = topic.background
                if hasattr(topic, 'body') and topic.body:
                    topic_data["body"] = topic.body

                # Build complete prompt as it would be sent (using selected prompt class)
                template = prompt_class.__doc__
                complete_prompt = template.format(
                    query=grade.query,
                    passage=grade.passage,
                )

                self.debug_logger.log(
                    f"INPUT [{grade.run_id} / {grade.topic_id}]",
                    {
                        "topic_data": topic_data,
                        "query": grade.query,
                        "response": grade.passage,
                        "prompt": complete_prompt,
                    }
                )

        # Run grading via DSPy (using selected prompt class)
        graded = asyncio.run(run_dspy_batch(
            prompt_class,
            grades,
            prompt_class.convert_output,
            backend=OpenAIMinimaLlm(llm_config)
        ))

        # Log outputs if debug enabled
        for grade in graded:
            self.debug_logger.log(
                f"OUTPUT [{grade.run_id} / {grade.topic_id}]",
                {
                    "output": grade.grade,
                    "grade": grade.grade,
                }
            )

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
            raise ValueError("DirectPromptJudge requires qrels. Run create_qrels first.")

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
    import os
    from trec_auto_judge import auto_judge_to_click_command

    # Check for debug log from environment variable (set by run_judge.py)
    debug_log = os.getenv('DIRECT_PROMPT_DEBUG_LOG')

    judge = DirectPromptJudge(debug_log=debug_log)
    cli = auto_judge_to_click_command(judge, "direct-prompt-judge")
    cli()
