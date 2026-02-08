"""
trec_auto_judge - TREC-specific evaluation tools for AutoJudge systems.

This package provides TREC-specific evaluation functionality built on autojudge_base.

For core AutoJudge functionality, import directly from:
- autojudge_base: Report, Request, Leaderboard, AutoJudge, etc.
- minima_llm: MinimaLlmConfig, OpenAIMinimaLlm, run_dspy_batch

This package provides:
- evaluation.py: TrecLeaderboardEvaluation for correlation analysis
- eval_results/: EvalResult containers and I/O
- pyircore.py: PyTerrier/IR integration
- _commands/: CLI commands (meta-evaluate, leaderboard, etc.)
"""

__version__ = '0.0.1'

# Re-export from autojudge_base for backwards compatibility
from autojudge_base import (
    # Core data models
    Report,
    load_report,
    Request,
    load_requests_from_irds,
    load_requests_from_file,
    # Output containers
    Leaderboard,
    LeaderboardEntry,
    MeasureSpec,
    LeaderboardSpec,
    LeaderboardBuilder,
    LeaderboardVerification,
    LeaderboardVerificationError,
    Qrels,
    QrelsSpec,
    QrelRow,
    build_qrels,
    QrelsVerification,
    QrelsVerificationError,
    write_qrel_file,
    doc_id_md5,
    # Nugget data
    NuggetBanks,
    NuggetBanksProtocol,
    # Protocols
    LeaderboardJudgeProtocol,
    QrelsCreatorProtocol,
    NuggetCreatorProtocol,
    AutoJudge,
    # CLI utilities
    option_rag_responses,
    option_rag_topics,
    option_ir_dataset,
    auto_judge_to_click_command,
    # Utilities
    format_preview,
)

# Re-export from minima_llm for backwards compatibility
from minima_llm import MinimaLlmConfig, OpenAIMinimaLlm

# TREC-specific exports
from .evaluation import TrecLeaderboardEvaluation

# === The click interface to the trec-auto-judge command line ====

from ._commands._meta_evaluate_deprecated import meta_evaluate as meta_evaluate_deprecated
from ._commands._meta_evaluate import meta_evaluate
from ._commands._leaderboard import leaderboard
from ._commands._eval_result import eval_result
from ._commands._export_corpus import export_corpus
from ._commands._list_models import list_models
from ._commands._qrel_evaluate import qrel_evaluate
from ._commands._run import run_workflow
from click import group
from autojudge_base.click_plus import option_rag_responses, option_rag_topics, option_ir_dataset, auto_judge_to_click_command

@group()
def main():
    pass


main.command("meta-evaluate-deprecated")(meta_evaluate_deprecated)
main.command("meta-evaluate")(meta_evaluate)
main.command("leaderboard")(leaderboard)
main.add_command(eval_result)
main.command()(export_corpus)
main.add_command(list_models)
main.add_command(qrel_evaluate, "qrel-evaluate")
main.add_command(run_workflow)


if __name__ == '__main__':
    main()
