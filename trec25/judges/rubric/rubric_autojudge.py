#!/usr/bin/env python3
"""
Rubric-based AutoJudge that:
1. Generates nugget questions from query (create_nuggets)
2. Grades how well each response answers each nugget (judge)
3. Derives evaluation score from nugget coverage
"""
from textwrap import dedent
from trec_auto_judge import *
from trec_auto_judge.nugget_data import (
    NuggetBank, NuggetBanks, NuggetQuestion
)

import dspy
import asyncio
import re
import json
from typing import *
from pydantic import BaseModel

from trec_auto_judge.llm.minima_llm_dspy import run_dspy_batch
from trec_auto_judge import MinimaLlmConfig, OpenAIMinimaLlm
from trec_auto_judge.leaderboard.leaderboard import OnMissing


# =============================================================================
# DSPy Signatures
# =============================================================================

class GenerateNuggetQuestionsMinimal(dspy.Signature):
    """Break a query into concise questions that must be answered."""

    query_title: str = dspy.InputField(desc="Query title")
    query_background: str = dspy.InputField(desc="Background context for the query")
    query_problem: str = dspy.InputField(desc="Problem statement to be addressed")

    questions: list[str] = dspy.OutputField(
        desc="List of concise questions that must be answered to address the query"
    )

class GenerateNuggetQuestionsWeb(dspy.Signature):
    dedent(
    """Break the query into concise questions that must be answered. 
     Generate 10 concise insightful questions that reveal whether information relevant for the query was provided, showcasing a deep understanding of the subject matter. Avoid basic or introductory-level inquiries. Keep the questions short."""
    )

    query_title: str = dspy.InputField(desc="Query title")
    query_background: str = dspy.InputField(desc="Background context for the query")
    query_problem: str = dspy.InputField(desc="Problem statement to be addressed")

    questions: list[str] = dspy.OutputField(
        desc="List of concise questions that must be answered to address the query"
    )



class GradeNuggetAnswer(dspy.Signature):
    """
    Grade how well a passage answers a specific question.

    Can the question be answered based on the available context? Choose one:
    - 5: The answer is highly relevant, complete, and accurate.
    - 4: The answer is mostly relevant and complete but may have minor gaps or inaccuracies.
    - 3: The answer is partially relevant and complete, with noticeable gaps or inaccuracies.
    - 2: The answer has limited relevance and completeness, with significant gaps or inaccuracies.
    - 1: The answer is minimally relevant or complete, with substantial shortcomings.
    - 0: The answer is not relevant or complete at all.
    """

    question: str = dspy.InputField(desc="The question to be answered")
    passage: str = dspy.InputField(desc="The passage that may contain the answer")

    grade: Literal["0", "1", "2", "3", "4", "5"] = dspy.OutputField(
        desc="Grade from 0-5 indicating how well the passage answers the question"
    )
    reasoning: Optional[str] = dspy.OutputField(
        desc="Brief explanation of the grade", default=None, required=False
    )


# =============================================================================
# Data Models (combined input/output)
# =============================================================================

class NuggetGenerationData(BaseModel):
    """Combined input/output for nugget question generation."""
    # Input fields
    query_id: str
    query_title: str
    query_background: str
    query_problem: str
    # Output fields (populated by LLM)
    questions: List[str] = []


class NuggetGradeData(BaseModel):
    """Combined input/output for grading a nugget against a passage."""
    # Input fields
    run_id: str
    query_id: str
    nugget_id: str
    question: str
    passage: str
    # Output fields (populated by LLM)
    grade: int = 0
    reasoning: Optional[str] = None
    confidence: Optional[float] = None


# =============================================================================
# Talmudir Export Models
# =============================================================================

class TalmudirCommentary(BaseModel):
    """Single commentary entry for Talmudir export."""
    id: str
    comment: str
    grade: str  # Numeric grade as string ("0"-"5")


class TalmudirSample(BaseModel):
    """Talmudir export format for a single sample."""
    sample_id: str
    query: str
    answer: str
    snippets: Dict[str, List[Dict[str, str]]] = {}
    commentary: List[TalmudirCommentary] = []
    mockAnswers: Dict[str, str] = {}
    automated_metrics: Dict[str, float] = {}


def write_talmudir_export(
    rag_responses: Sequence["Report"],
    rag_topics: Sequence["Request"],
    grade_data: List[NuggetGradeData],
    response_grades: Dict[str, Dict[str, Any]],
    filebase: str,
    grade_threshold: int = 4,
) -> None:
    """
    Export judged results to Talmudir format.

    Args:
        rag_responses: RAG responses that were judged
        rag_topics: Topics/queries
        grade_data: Per-nugget grading results
        response_grades: Aggregated grades per response (response_key -> evaldata)
        filebase: Output file base path (writes to {filebase}.talmudir.jsonl)
        grade_threshold: Only include commentary for nuggets with grade >= threshold
    """
    from pathlib import Path

    # Build request lookup by topic_id
    request_by_topic: Dict[str, "Request"] = {
        t.request_id: t for t in rag_topics
    }

    # Build grade_data lookup by response_key
    grades_by_response: Dict[str, List[NuggetGradeData]] = {}
    for data in grade_data:
        response_key = f"{data.run_id}:{data.query_id}"
        if response_key not in grades_by_response:
            grades_by_response[response_key] = []
        grades_by_response[response_key].append(data)

    # Build Talmudir samples
    samples: List[TalmudirSample] = []

    for response in rag_responses:
        topic_id = response.metadata.topic_id
        run_id = response.metadata.run_id
        response_key = f"{run_id}:{topic_id}"

        # Get request for query text
        request = request_by_topic.get(topic_id)
        if request is None:
            continue

        # Build query string
        query_parts = []
        if request.title:
            query_parts.append(request.title)
        if request.problem_statement:
            query_parts.append(request.problem_statement)
        query = " ".join(query_parts)

        # Build commentary (only for grade >= threshold)
        commentary: List[TalmudirCommentary] = []
        response_grade_data = grades_by_response.get(response_key, [])
        for gd in response_grade_data:
            if gd.grade >= grade_threshold:
                comment_parts = [gd.question]
                if gd.reasoning:
                    comment_parts.append(gd.reasoning)
                commentary.append(TalmudirCommentary(
                    id=gd.nugget_id,
                    comment=" ".join(comment_parts),
                    grade=str(gd.grade),
                ))

        # Get automated metrics from response_grades
        evaldata = response_grades.get(response_key, {})
        automated_metrics = {
            "NUGGET_COVERAGE": evaldata.get("coverage_score", 0.0),
            "AVG_GRADE": evaldata.get("avg_grade", 0.0),
            "MAX_GRADE": float(evaldata.get("max_grade", 0)),
            "COVERED_COUNT": float(evaldata.get("covered_count", 0)),
        }

        sample = TalmudirSample(
            sample_id=f"{topic_id}-{run_id}",
            query=query,
            answer=response.get_report_text(),
            snippets={},
            commentary=commentary,
            mockAnswers={},
            automated_metrics=automated_metrics,
        )
        samples.append(sample)

    # Write to file
    output_path = Path(f"{filebase}.talmudir.jsonl")
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(sample.model_dump_json() + "\n")

    print(f"Talmudir: Exported {len(samples)} samples to {output_path}")


# =============================================================================
# Leaderboard & Qrels Specs
# =============================================================================

RUBRIC_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("NUGGET_COVERAGE", aggregate=mean_of_floats, cast=float, default=0.0),
    MeasureSpec("AVG_GRADE", aggregate=mean_of_floats, cast=float, default=0.0),
    MeasureSpec("MAX_GRADE", aggregate=mean_of_floats, cast=float, default=0.0),
    MeasureSpec("COVERED_COUNT", aggregate=mean_of_floats, cast=float, default=0.0),
))


RUBRIC_QRELS = QrelsSpec["NuggetGradeData"](
    topic_id=lambda r: r.query_id,
    doc_id=lambda r: doc_id_md5(r.passage),
    grade=lambda r: float(r.grade),
    on_duplicate="keep_max"
)


# =============================================================================
# Conversion Functions
# =============================================================================

def _parse_grade(s: str) -> int:
    """Extract grade 0-5 from string."""
    m = re.search(r'\b([0-5])\b', s)
    if not m:
        return 0  # Default to 0 if no valid grade found
    return int(m.group(1))


# =============================================================================
# RubricJudge Implementation
# =============================================================================

class RubricJudge(AutoJudge):
    """
    Rubric-based judge that:
    1. Generates nugget questions from topics
    2. Grades responses against each nugget
    3. Computes coverage score based on grade threshold
    """

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks
    
    def __init__(self):
        self.expected_topic_ids:Sequence[str] = []
        self.on_missing_evals: OnMissing = "fix_aggregate"
        
    def create_nuggets(
        self,
        prompt:str,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanks] = None,
        **kwargs
    ) -> Optional[NuggetBanks]:
        """Generate nugget questions for each topic using LLM."""

        # Prepare generation data
        gen_data = [
            NuggetGenerationData(
                query_id=topic.request_id,
                query_title=topic.title or "",
                query_background=topic.background or "",
                query_problem=topic.problem_statement or ""
            )
            for topic in rag_topics
        ]

        # Convert output handler
        def convert_gen_output(prediction: dspy.Prediction, data: NuggetGenerationData) -> None:
            questions = prediction.questions if hasattr(prediction, 'questions') else []
            # DSPy may return list as JSON string - parse it
            if isinstance(questions, str):
                try:
                    parsed = json.loads(questions)
                    if isinstance(parsed, list):
                        questions = [str(q).strip() for q in parsed if q]
                    else:
                        # Fallback: split by newlines
                        questions = [q.strip() for q in questions.split('\n') if q.strip()]
                except json.JSONDecodeError:
                    # Fallback: split by newlines
                    questions = [q.strip() for q in questions.split('\n') if q.strip()]
            data.questions = questions

        # Run LLM generation
        print(f"Rubric: Generating questions...")
        
        if prompt == "minimal":
            prompt_sig = GenerateNuggetQuestionsMinimal
        elif prompt == "web":
            prompt_sig = GenerateNuggetQuestionsWeb
        else:
            raise RuntimeError(f"Prompt not defined: {prompt}")    
        
        gen_data = asyncio.run(run_dspy_batch(
            prompt_sig,
            gen_data,
            convert_gen_output,
            backend=OpenAIMinimaLlm(llm_config)
            
        ))
        print(f"Rubric: Finished gnerating questions")

        # Build NuggetBanks from generated questions
        banks = []
        for data in gen_data:
            bank = NuggetBank(
                query_id=data.query_id,
                title_query=data.query_title
            )

            nuggets = [
                NuggetQuestion(
                    query_id=data.query_id,
                    question=question_text,
                    question_id=f"{data.query_id}-q{i}"
                )
                for i, question_text in enumerate(data.questions)
            ]

            bank.add_nuggets(nuggets)
            bank.index_nuggets()
            banks.append(bank)

        return NuggetBanks.from_banks_list(banks)


    
    
    
    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanks] = None,
        grade_threshold: int = 3,
        filebase: str = "rubric",
        **kwargs
    ) -> tuple[Leaderboard, Optional[Qrels]]:
        """
        Grade each response against all nuggets for its topic.

        Stores per-nugget grades in Report.evaldata with format:
        {
            "nugget_grades": {
                "<nugget_id>": {"grade": int, "reasoning": str},
                ...
            },
            "coverage_score": float,
            "avg_grade": float,
            "covered_count": int,
            "total_nuggets": int
        }
        """
        if nugget_banks is None:
            raise ValueError("RubricJudge requires nugget_banks. Run create_nuggets first or provide --nugget-banks.")

        # Prepare grading data (one per response-nugget pair)
        grade_data: List[NuggetGradeData] = []
        response_nugget_map: Dict[str, List[NuggetGradeData]] = {}  # run_id:topic_id -> data list
        self.expected_topic_ids=[t.request_id for t in rag_topics]  # Todo not necessary to be a member variable, unless passed in during construction
        
        for response in rag_responses:
            metadata = response.metadata
            run_id = metadata.run_id
            topic_id = metadata.topic_id
            text = response.get_report_text()

            bank = nugget_banks.banks.get(topic_id)
            if bank is None:
                print(f"Warning: No nugget bank for topic {topic_id}, skipping")
                continue

            response_key = f"{run_id}:{topic_id}"
            response_nugget_map[response_key] = []

            # Create grade data for each nugget question
            for nugget in bank.nuggets_as_list():
                if isinstance(nugget, NuggetQuestion):
                    data = NuggetGradeData(
                        run_id=run_id,
                        query_id=topic_id,
                        nugget_id=nugget.question_id or nugget.question,
                        question=nugget.question,
                        passage=text
                    )
                    grade_data.append(data)
                    response_nugget_map[response_key].append(data)

        # Convert output handler
        def convert_grade_output(prediction: dspy.Prediction, data: NuggetGradeData) -> None:
            data.grade = _parse_grade(prediction.grade)
            data.reasoning = getattr(prediction, 'reasoning', None)
            data.confidence = getattr(prediction, 'confidence', None)


        # Run LLM grading
        print(f"Rubric: Grading responses...")
        if grade_data:
            grade_data = asyncio.run(run_dspy_batch(
                GradeNuggetAnswer,
                grade_data,
                convert_grade_output,
                backend=OpenAIMinimaLlm(llm_config)
            ))
        print(f"Rubric: Finished grading")


        # Aggregate grades per response and store in evaldata
        response_grades: Dict[str, Dict[str, Any]] = {}  # response_key -> evaldata

        for data in grade_data:
            response_key = f"{data.run_id}:{data.query_id}"
            if response_key not in response_grades:
                response_grades[response_key] = {
                    "nugget_grades": {},
                    "grades_list": []
                }

            response_grades[response_key]["nugget_grades"][data.nugget_id] = {
                "grade": data.grade,
                "reasoning": data.reasoning
            }
            response_grades[response_key]["grades_list"].append(data.grade)

        # Compute coverage scores
        for response_key, evaldata in response_grades.items():
            grades = evaldata["grades_list"]
            if grades:
                covered = sum(1 for g in grades if g >= grade_threshold)
                evaldata["coverage_score"] = covered / len(grades)
                evaldata["avg_grade"] = sum(grades) / len(grades)
                evaldata["max_grade"] = max(grades)
                evaldata["covered_count"] = covered
                evaldata["total_nuggets"] = len(grades)
            else:
                evaldata["coverage_score"] = 0.0
                evaldata["avg_grade"] = 0.0
                evaldata["covered_count"] = 0
                evaldata["max_grade"] = 0
                evaldata["total_nuggets"] = 0
            del evaldata["grades_list"]  # Remove temporary field

        # Update Report.evaldata
        for response in rag_responses:
            response_key = f"{response.metadata.run_id}:{response.metadata.topic_id}"
            if response_key in response_grades:
                response.evaldata = response_grades[response_key]


        # Build leaderboard
        leaderboard = self._build_leaderboard(response_grades)
        leaderboard.verify(warn=True, expected_topic_ids=self.expected_topic_ids, on_missing=self.on_missing_evals)

        # Build qrels from grade data
        qrels = build_qrels(records=grade_data, spec=RUBRIC_QRELS) if grade_data else None
        qrels.verify(warn=True, expected_topic_ids=self.expected_topic_ids)

        # Export to Talmudir format
        write_talmudir_export(
            rag_responses=rag_responses,
            rag_topics=rag_topics,
            grade_data=grade_data,
            response_grades=response_grades,
            filebase=filebase,
            grade_threshold=grade_threshold,  # Only include commentary for high-quality answers
        )

        return (leaderboard, qrels)

    def _build_leaderboard(self, response_grades: Dict[str, Dict[str, Any]]) -> Leaderboard:
        """Build leaderboard from aggregated response grades."""
        b = LeaderboardBuilder(RUBRIC_SPEC)

        for response_key, evaldata in response_grades.items():
            run_id, topic_id = response_key.split(":", 1)
            b.add(
                run_id=run_id,
                topic_id=topic_id,
                values={
                    "NUGGET_COVERAGE": evaldata["coverage_score"],
                    "AVG_GRADE": evaldata["avg_grade"],
                    "MAX_GRADE": evaldata["max_grade"],
                    "COVERED_COUNT": float(evaldata["covered_count"]),
                }
            )

        leaderboard = b.build(expected_topic_ids=self.expected_topic_ids, on_missing = self.on_missing_evals)
        leaderboard.verify(expected_topic_ids=self.expected_topic_ids, warn=False, on_missing = self.on_missing_evals)
        return leaderboard


if __name__ == '__main__':
    auto_judge_to_click_command(RubricJudge(), "rubric_autojudge")()