#!/usr/bin/env python3
"""
MinimalJudge: A simple example AutoJudge implementation.

This judge demonstrates the AutoJudge protocol without using any LLM calls.
It creates nuggets, qrels, and leaderboards using deterministic logic based
on response text length and keyword matching.

Use this as a starting template for building your own judge.
"""

from typing import Iterable, Optional, Sequence, Type

from trec_auto_judge import (
    AutoJudge,
    MinimaLlmConfig,
    Report,
    Request,
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    MeasureSpec,
    mean_of_floats,
    mean_of_bools,
    Qrels,
    QrelsSpec,
    build_qrels,
    doc_id_md5,
    auto_judge_to_click_command,
)
from trec_auto_judge.nugget_data import (
    NuggetBanks,
    NuggetBank,
    NuggetQuestion,
    NuggetBanksProtocol,
)


# =============================================================================
# Leaderboard Specification
# =============================================================================
# Define what measures the judge produces and how to aggregate them.

MINIMAL_SPEC = LeaderboardSpec(measures=(
    MeasureSpec(
        name="SCORE",
        aggregate=mean_of_floats,  # Average across topics
        cast=float,                # Ensure values are floats
        default=0.0,               # Default for missing entries
    ),
    MeasureSpec(
        name="HAS_KEYWORDS",
        aggregate=mean_of_bools,   # Fraction of topics with keywords
        cast=bool,
        default=False,
    ),
))


# =============================================================================
# Qrels Specification
# =============================================================================
# Define how to extract (topic_id, doc_id, grade) from grading records.

class GradeRecord:
    """Simple record for qrels building."""
    def __init__(self, topic_id: str, text: str, grade: int):
        self.topic_id = topic_id
        self.text = text
        self.grade = grade


MINIMAL_QRELS_SPEC = QrelsSpec[GradeRecord](
    topic_id=lambda r: r.topic_id,
    doc_id=lambda r: doc_id_md5(r.text),  # Hash response text as doc_id
    grade=lambda r: r.grade,
    on_duplicate="keep_max",  # Keep highest grade if duplicates
)


# =============================================================================
# MinimalJudge Implementation
# =============================================================================

class MinimalJudge(AutoJudge):
    """
    A minimal AutoJudge that demonstrates the protocol.

    This judge:
    - Creates nuggets: Generates questions from topic titles
    - Creates qrels: Grades responses based on text length
    - Judges: Scores responses based on keyword coverage
    """

    # Declare the nugget format this judge uses
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    # create_nuggets(): Generate nugget questions for each topic
    # -------------------------------------------------------------------------
    def create_nuggets(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        # Judge-specific settings from workflow.yml
        questions_per_topic: int = 3,
        **kwargs,
    ) -> Optional[NuggetBanksProtocol]:
        """
        Create nugget questions for each topic.

        In a real judge, this would use an LLM to generate meaningful questions.
        Here we create simple template questions for demonstration.
        """
        banks = []

        for topic in rag_topics:
            # Create a NuggetBank for this topic
            bank = NuggetBank(
                query_id=topic.request_id,
                title_query=topic.title or topic.request_id,
            )

            # Generate questions (in a real judge, use LLM)
            questions = []
            for i in range(questions_per_topic):
                question = NuggetQuestion.from_lazy(
                    query_id=topic.request_id,
                    question=f"Q{i+1}: What information about '{topic.title}' is provided?",
                    gold_answers=[f"Answer about {topic.title}"],  # Optional gold answers
                )
                questions.append(question)

            # Add questions to the bank
            bank.add_nuggets(questions)
            banks.append(bank)

        # Combine into multi-topic container
        nugget_banks = NuggetBanks.from_banks_list(banks)

        print(f"MinimalJudge: Created nuggets for {len(banks)} topics")
        return nugget_banks

    # -------------------------------------------------------------------------
    # create_qrels(): Generate relevance judgments
    # -------------------------------------------------------------------------
    def create_qrels(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        # Judge-specific settings from workflow.yml
        grade_range: tuple = (0, 3),
        length_threshold: int = 100,
        **kwargs,
    ) -> Optional[Qrels]:
        """
        Create relevance judgments (qrels) for each response.

        In a real judge, this would use an LLM to assess relevance.
        Here we use a simple length-based heuristic.
        """
        grade_records = []

        for response in rag_responses:
            topic_id = response.metadata.topic_id
            text = response.get_report_text()

            # Simple grading heuristic (replace with LLM in real judge)
            text_length = len(text)
            if text_length > length_threshold * 3:
                grade = grade_range[1]  # Excellent
            elif text_length > length_threshold * 2:
                grade = 2  # Good
            elif text_length > length_threshold:
                grade = 1  # Fair
            else:
                grade = grade_range[0]  # Poor

            grade_records.append(GradeRecord(topic_id, text, grade))

        # Build qrels from records
        qrels = build_qrels(records=grade_records, spec=MINIMAL_QRELS_SPEC)

        print(f"MinimalJudge: Created qrels for {len(grade_records)} responses")
        return qrels

    # -------------------------------------------------------------------------
    # judge(): Score responses and produce leaderboard
    # -------------------------------------------------------------------------
    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        # Judge-specific settings from workflow.yml
        keyword_bonus: float = 0.2,
        on_missing_evals: str = "fix_aggregate",
        **kwargs,
    ) -> Leaderboard:
        """
        Judge RAG responses and produce a leaderboard.

        This method scores each response and builds a leaderboard with
        per-topic rows and aggregate "all" rows.
        """
        # Get expected topic IDs for verification
        expected_topic_ids = [t.request_id for t in rag_topics]

        # Build topic title lookup for keyword matching
        topic_titles = {t.request_id: (t.title or "").lower() for t in rag_topics}

        # Create leaderboard builder
        builder = LeaderboardBuilder(MINIMAL_SPEC)

        for response in rag_responses:
            run_id = response.metadata.run_id
            topic_id = response.metadata.topic_id
            text = response.get_report_text().lower()

            # Calculate SCORE (0.0 to 1.0)
            # Base score from text length (normalize to 0-1)
            base_score = min(len(text) / 1000.0, 1.0)

            # Check for keywords from topic title
            title_words = topic_titles.get(topic_id, "").split()
            keywords_found = sum(1 for word in title_words if word in text)
            has_keywords = keywords_found > 0

            # Apply keyword bonus
            score = base_score
            if has_keywords:
                score = min(score + keyword_bonus, 1.0)

            # Optionally use nuggets for additional scoring
            if nugget_banks and topic_id in nugget_banks.banks:
                bank = nugget_banks.banks[topic_id]
                # Example: bonus for having nuggets available
                nugget_count = len(bank.nuggets_as_list())
                if nugget_count > 0:
                    score = min(score + 0.05 * nugget_count, 1.0)

            # Optionally use qrels
            if qrels:
                # Example: could adjust score based on qrels grades
                pass

            # Add row to leaderboard
            builder.add(
                run_id=run_id,
                topic_id=topic_id,
                values={
                    "SCORE": score,
                    "HAS_KEYWORDS": has_keywords,
                },
            )

        # Build leaderboard with aggregate rows
        leaderboard = builder.build(
            expected_topic_ids=expected_topic_ids,
            on_missing=on_missing_evals,
        )

        # Verify the leaderboard
        leaderboard.verify(
            expected_topic_ids=expected_topic_ids,
            warn=True,
            on_missing=on_missing_evals,
        )

        print(f"MinimalJudge: Built leaderboard with {len(leaderboard.entries)} entries")
        return leaderboard


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    # Register CLI with subcommands: nuggify, judge, run
    auto_judge_to_click_command(MinimalJudge(), "minimal_judge")()
