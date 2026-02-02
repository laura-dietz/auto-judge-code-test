from typing import Iterable, Protocol, Sequence, Optional, Type

from .report import Report, load_report
from .request import Request, load_requests_from_irds, load_requests_from_file
from .leaderboard import Leaderboard, LeaderboardEntry, MeasureSpec, LeaderboardSpec, LeaderboardBuilder, LeaderboardVerification, LeaderboardVerificationError
from .qrels.qrels import QrelsSpec, QrelRow, Qrels, build_qrels, QrelsVerification, QrelsVerificationError, write_qrel_file, doc_id_md5
from .llm.minima_llm import MinimaLlmConfig, OpenAIMinimaLlm
from .nugget_data import NuggetBanks, NuggetBanksProtocol
from .utils import format_preview
__version__ = '0.0.1'


# === The interface for AutoJudges to implement ====
#
# Three separate protocols allow mixing implementations:
#   - LeaderboardJudgeProtocol: produces leaderboard scores
#   - QrelsCreatorProtocol: creates relevance judgments
#   - NuggetCreatorProtocol: creates nugget banks
#
# A single class can implement all three (common case), or different
# classes can be used for each phase via workflow.yml configuration.


class LeaderboardJudgeProtocol(Protocol):
    """Protocol for leaderboard generation."""

    def judge(
        self,
        rag_responses: Iterable["Report"],
        rag_topics: Sequence["Request"],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        **kwargs
    ) -> "Leaderboard":
        """
        Judge RAG responses against topics and produce a leaderboard.

        Args:
            rag_responses: RAG system outputs to judge
            rag_topics: Evaluation topics/queries
            llm_config: LLM configuration
            nugget_banks: Optional nuggets for judgment
            qrels: Optional qrels from create_qrels() phase
            **kwargs: Additional settings

        Returns:
            Leaderboard with rankings/scores for runs
        """
        ...


class QrelsCreatorProtocol(Protocol):
    """Protocol for qrels creation."""

    def create_qrels(
        self,
        rag_responses: Iterable["Report"],
        rag_topics: Sequence["Request"],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> Optional["Qrels"]:
        """
        Create relevance judgments (qrels) for RAG responses.

        Args:
            rag_responses: RAG system outputs to judge
            rag_topics: Evaluation topics/queries
            llm_config: LLM configuration for qrels generation
            nugget_banks: Optional nuggets to use for judgment
            **kwargs: Additional settings (e.g., grade_range=[0, 3])

        Returns:
            Qrels with relevance judgments, or None if not supported
        """
        ...


class NuggetCreatorProtocol(Protocol):
    """Protocol for nugget creation."""

    nugget_banks_type: Type[NuggetBanksProtocol]
    """The NuggetBanks container type this creator produces."""

    def create_nuggets(
        self,
        rag_responses: Iterable["Report"],
        rag_topics: Sequence["Request"],
        llm_config: MinimaLlmConfig,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> Optional[NuggetBanksProtocol]:
        """
        Create or refine nugget banks based on RAG responses.

        Args:
            rag_responses: RAG system outputs to analyze for nugget creation/refinement
            rag_topics: Evaluation topics/queries
            llm_config: LLM configuration for nugget generation
            nugget_banks: Optional existing nuggets to refine/extend

        Returns:
            NuggetBanks container, or None if not supported
        """
        ...


class AutoJudge(LeaderboardJudgeProtocol, QrelsCreatorProtocol, NuggetCreatorProtocol, Protocol):
    """
    Combined protocol for judges that implement all three phases.

    This is a convenience protocol for the common case where a single class
    handles nugget creation, qrels creation, and leaderboard generation.

    For modular configurations, use the individual protocols:
    - LeaderboardJudgeProtocol
    - QrelsCreatorProtocol
    - NuggetCreatorProtocol
    """
    pass


# === The click interface to the trec-auto-judge command line ====

from ._commands._meta_evaluate import meta_evaluate
from ._commands._meta_evaluate_v2 import meta_evaluate as meta_evaluate_v2
from ._commands._leaderboard import leaderboard
from ._commands._eval_result import eval_result
from ._commands._export_corpus import export_corpus
from ._commands._list_models import list_models
from ._commands._run import run_workflow
from click import group
from .click_plus import option_rag_responses, option_rag_topics, option_ir_dataset, auto_judge_to_click_command

@group()
def main():
    pass


main.command("meta-evaluate")(meta_evaluate)
main.command("meta-evaluate-v2")(meta_evaluate_v2)
main.command("leaderboard")(leaderboard)
main.add_command(eval_result)
main.command()(export_corpus)
main.add_command(list_models)
main.add_command(run_workflow)


if __name__ == '__main__':
    main()