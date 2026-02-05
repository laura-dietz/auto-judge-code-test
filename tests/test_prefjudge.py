"""
Verification test suite for PrefJudge implementation.

Runs each verifier separately to provide granular feedback on issues.
PrefJudge is a preference-based judge that does not use nuggets.

Note: These tests require a configured LLM backend. Set up via:
- Direct config: llm-config.yml with base_url and model
- Environment: OPENAI_BASE_URL and OPENAI_MODEL
"""

import pytest
from pathlib import Path
from typing import List, Optional

from autojudge_base import Request, Report, Leaderboard, Qrels
from autojudge_base.leaderboard.leaderboard import (
    LeaderboardVerification,
    LeaderboardVerificationError,
)
from autojudge_base.report import ReportMetaData, Rag24ReportSentence
from minima_llm import MinimaLlmConfig


# =============================================================================
# Fixtures
# =============================================================================

PREFJUDGE_DIR = Path(__file__).parent.parent / "trec25" / "judges" / "prefjudge"


@pytest.fixture
def llm_config() -> MinimaLlmConfig:
    """Load LLM config from judge's config file or environment."""
    config_path = PREFJUDGE_DIR / "llm-config.yml"
    try:
        return MinimaLlmConfig.from_yaml(config_path)
    except (FileNotFoundError, ValueError):
        # Fall back to environment variables
        return MinimaLlmConfig.from_env()


@pytest.fixture
def sample_topics() -> List[Request]:
    """Create sample topics for testing."""
    return [
        Request(
            request_id="topic-001",
            title="Climate change impacts",
            problem_statement="What are the main impacts of climate change?",
            background="Understanding climate change effects on ecosystems.",
        ),
        Request(
            request_id="topic-002",
            title="Renewable energy",
            problem_statement="What are the benefits of renewable energy?",
            background="Exploring sustainable energy alternatives.",
        ),
    ]


@pytest.fixture
def sample_responses(sample_topics) -> List[Report]:
    """Create sample RAG responses for testing."""
    responses = []
    for topic in sample_topics:
        for run_id in ["run-A", "run-B", "run-C"]:  # 3 runs for pairwise comparison
            metadata = ReportMetaData(
                run_id=run_id,
                narrative_id=topic.request_id,
                narrative=topic.problem_statement,
                team_id="test-team",
                type="automatic",
            )
            answer = [
                Rag24ReportSentence(
                    text=f"Sample response for {topic.title} from {run_id}.",
                    citations=[0, 1],
                ),
                Rag24ReportSentence(
                    text="This discusses the topic in detail with relevant information.",
                    citations=[1, 2],
                ),
            ]
            report = Report(
                metadata=metadata,
                answer=answer,
                references=["doc-0", "doc-1", "doc-2"],
            )
            responses.append(report)
    return responses


# =============================================================================
# Test Driver Class
# =============================================================================


class AutoJudgeTestDriver:
    """
    Test driver that runs an AutoJudge and stores results for verification.

    Usage:
        driver = AutoJudgeTestDriver(judge, topics, responses, llm_config)
        driver.run_judge()

        # Then use driver.leaderboard, driver.qrels in tests

    For judges that require additional settings (from workflow), pass them as kwargs:
        driver = AutoJudgeTestDriver(..., num_pivot=1, num_others=2)
    """

    def __init__(
        self,
        auto_judge,
        rag_topics: List[Request],
        rag_responses: List[Report],
        llm_config: MinimaLlmConfig,
        **kwargs,
    ):
        self.auto_judge = auto_judge
        self.rag_topics = rag_topics
        self.rag_responses = rag_responses
        self.llm_config = llm_config
        self.settings = kwargs  # Additional settings from workflow

        # Results populated by run methods
        self.leaderboard: Optional[Leaderboard] = None
        self.qrels: Optional[Qrels] = None

    def run_judge(self):
        """Run judge and store results."""
        # Qrels now come from create_qrels(), not judge()
        self.qrels = self.auto_judge.create_qrels(
            rag_responses=self.rag_responses,
            rag_topics=self.rag_topics,
            llm_config=self.llm_config,
            nugget_banks=None,  # PrefJudge doesn't use nuggets
            **self.settings,
        )

        self.leaderboard = self.auto_judge.judge(
            rag_responses=self.rag_responses,
            rag_topics=self.rag_topics,
            llm_config=self.llm_config,
            nugget_banks=None,  # PrefJudge doesn't use nuggets,
            qrels=self.qrels,
            **self.settings,
        )

        return self.leaderboard, self.qrels


# =============================================================================
# PrefJudge Tests
# =============================================================================


class TestPrefJudgeVerification:
    """Verification tests for PrefJudge implementation."""

    @pytest.fixture
    def pref_judge(self):
        """Create PrefJudge instance."""
        dspy = pytest.importorskip("dspy")
        from trec25.judges.prefjudge.pref_judge import PrefJudge

        return PrefJudge()

    @pytest.fixture
    def pref_settings(self):
        """Settings required by PrefJudge (from workflow)."""
        return {
            "num_pivot": 1,  # 1 pivot response compared against all
            "num_others": 2,  # Compare against 2 other non-pivot responses
            "on_missing_evals": "fix_aggregate",
        }

    @pytest.fixture
    def driver(self, pref_judge, sample_topics, sample_responses, llm_config, pref_settings):
        """Create test driver for PrefJudge."""
        return AutoJudgeTestDriver(
            auto_judge=pref_judge,
            rag_topics=sample_topics,
            rag_responses=sample_responses,
            llm_config=llm_config,
            **pref_settings,
        )

    # -------------------------------------------------------------------------
    # create_nuggets tests (PrefJudge doesn't use nuggets)
    # -------------------------------------------------------------------------

    def test_create_nuggets_returns_none(self, pref_judge):
        """Verify create_nuggets returns None (PrefJudge doesn't use nuggets)."""
        result = pref_judge.create_nuggets()
        assert result is None

    # -------------------------------------------------------------------------
    # judge verification tests
    # -------------------------------------------------------------------------

    @pytest.fixture
    def judge_results(self, driver):
        """Run judge and return driver with results."""
        driver.run_judge()
        return driver

    def test_judge_returns_leaderboard(self, judge_results):
        """Verify judge returns a Leaderboard object."""
        assert judge_results.leaderboard is not None
        assert isinstance(judge_results.leaderboard, Leaderboard)

    def test_judge_leaderboard_has_borda_count(self, judge_results):
        """Verify leaderboard has BORDA_COUNT measure."""
        assert "BORDA_COUNT" in judge_results.leaderboard.measures

    def test_judge_leaderboard_has_win_frac(self, judge_results):
        """Verify leaderboard has WIN_FRAC measure."""
        assert "WIN_FRAC" in judge_results.leaderboard.measures

    def test_judge_leaderboard_complete_measures(self, judge_results):
        """Verify every leaderboard entry has all measures."""
        LeaderboardVerification(
            judge_results.leaderboard, on_missing="error", warn=False
        ).complete_measures()

    def test_judge_leaderboard_complete_measures_excluding_all_row(self, judge_results):
        """Verify per-topic entries have all measures (excluding 'all' row)."""
        LeaderboardVerification(
            judge_results.leaderboard, on_missing="error"
        ).complete_measures(include_all_row=False)

    def test_judge_leaderboard_same_topics_per_run(self, judge_results):
        """Verify all runs have the same set of topics."""
        LeaderboardVerification(
            judge_results.leaderboard, on_missing="error", warn=False
        ).same_topics_per_run()

    def test_judge_leaderboard_complete_topics(self, judge_results, sample_topics):
        """Verify every expected topic has leaderboard entries."""
        topic_ids = [t.request_id for t in sample_topics]
        LeaderboardVerification(
            judge_results.leaderboard,
            on_missing="error",
            expected_topic_ids=topic_ids,
            warn=False,
        ).complete_topics()

    def test_judge_leaderboard_no_extra_topics(self, judge_results, sample_topics):
        """Verify no leaderboard entries for unexpected topics."""
        topic_ids = [t.request_id for t in sample_topics]
        LeaderboardVerification(
            judge_results.leaderboard,
            on_missing="error",
            expected_topic_ids=topic_ids,
            warn=False,
        ).no_extra_topics()

    def test_judge_leaderboard_all_verification(self, judge_results, sample_topics):
        """Run all leaderboard verification checks."""
        topic_ids = [t.request_id for t in sample_topics]
        LeaderboardVerification(
            judge_results.leaderboard,
            on_missing="error",
            expected_topic_ids=topic_ids,
            warn=False,
        ).all()

    # -------------------------------------------------------------------------
    # qrels tests (PrefJudge doesn't return qrels)
    # -------------------------------------------------------------------------

    def test_judge_qrels_is_none(self, judge_results):
        """Verify PrefJudge returns None for qrels."""
        assert judge_results.qrels is None

    # -------------------------------------------------------------------------
    # Response evaldata tests
    # -------------------------------------------------------------------------

    def test_responses_have_evaldata(self, judge_results, driver):
        """Verify responses are annotated with evaldata after judging."""
        for response in driver.rag_responses:
            assert response.evaldata is not None, (
                f"Response {response.metadata.run_id}:{response.metadata.topic_id} "
                "missing evaldata"
            )

    def test_responses_evaldata_has_borda_count(self, judge_results, driver):
        """Verify response evaldata contains BORDA_COUNT."""
        for response in driver.rag_responses:
            assert "BORDA_COUNT" in response.evaldata, (
                f"Response {response.metadata.run_id}:{response.metadata.topic_id} "
                "missing BORDA_COUNT in evaldata"
            )

    def test_responses_evaldata_has_win_frac(self, judge_results, driver):
        """Verify response evaldata contains WIN_FRAC."""
        for response in driver.rag_responses:
            assert "WIN_FRAC" in response.evaldata, (
                f"Response {response.metadata.run_id}:{response.metadata.topic_id} "
                "missing WIN_FRAC in evaldata"
            )

    def test_responses_evaldata_has_comparison_lists(self, judge_results, driver):
        """Verify response evaldata contains better_than and worse_than lists."""
        for response in driver.rag_responses:
            assert "better_than" in response.evaldata, (
                f"Response {response.metadata.run_id}:{response.metadata.topic_id} "
                "missing better_than in evaldata"
            )
            assert "worse_than" in response.evaldata, (
                f"Response {response.metadata.run_id}:{response.metadata.topic_id} "
                "missing worse_than in evaldata"
            )


# =============================================================================
# Verification Failure Tests (ensure verifiers catch problems)
# =============================================================================


class TestVerificationCatchesProblems:
    """Tests that verify the verifiers actually catch problems."""

    def test_leaderboard_verification_catches_missing_topic(self, sample_topics):
        """Verify LeaderboardVerification catches missing topics."""
        from trec_auto_judge import (
            LeaderboardSpec,
            LeaderboardBuilder,
            MeasureSpec,
        )

        spec = LeaderboardSpec(
            measures=(MeasureSpec("SCORE"),)
        )
        builder = LeaderboardBuilder(spec)

        # Only add entry for first topic
        builder.add(run_id="run-A", topic_id="topic-001", values={"SCORE": 0.5})
        leaderboard = builder.build()

        topic_ids = [t.request_id for t in sample_topics]

        with pytest.raises(LeaderboardVerificationError):
            LeaderboardVerification(
                leaderboard,
                on_missing="error",
                expected_topic_ids=topic_ids,
                warn=False,
            ).complete_topics()