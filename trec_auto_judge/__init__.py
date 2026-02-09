"""
trec_auto_judge - TREC-specific judge implementations for AutoJudge systems.

This package provides TREC-specific judge functionality built on autojudge_base.

For core AutoJudge functionality, import directly from:
- autojudge_base: Report, Request, Leaderboard, AutoJudge, etc.
- minima_llm: MinimaLlmConfig, OpenAIMinimaLlm, run_dspy_batch

Evaluation commands (meta-evaluate, qrel-evaluate, leaderboard, eval-result) are
available via both ``trec-auto-judge`` and ``auto-judge-evaluate`` CLIs.

This package provides:
- _commands/: CLI commands (run, export-corpus, list-models)
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

# === The click interface to the trec-auto-judge command line ====

from ._commands._export_corpus import export_corpus
from ._commands._list_models import list_models
from ._commands._run import run_workflow
from click import group
from autojudge_base.click_plus import option_rag_responses, option_rag_topics, option_ir_dataset, auto_judge_to_click_command

@group()
def main():
    pass


main.command()(export_corpus)
main.add_command(list_models)
main.add_command(run_workflow)

# Re-export autojudge_evaluate CLI commands for backwards compatibility
from autojudge_evaluate._commands._meta_evaluate import meta_evaluate
from autojudge_evaluate._commands._leaderboard import leaderboard
from autojudge_evaluate._commands._eval_result import eval_result
from autojudge_evaluate._commands._qrel_evaluate import qrel_evaluate

main.command("meta-evaluate")(meta_evaluate)
main.command("leaderboard")(leaderboard)
main.add_command(eval_result)
main.add_command(qrel_evaluate, "qrel-evaluate")


if __name__ == '__main__':
    main()
