import click
from pathlib import Path
import pandas as pd
from ..evaluation import TrecLeaderboardEvaluation
from ..click_plus import (
    detect_header_interactive,
    LEADERBOARD_FORMATS,
    LEADERBOARD_FORMAT_HELP,
)
from typing import List, Optional
from tira.io_utils import to_prototext


def persist_output(df: pd.DataFrame, output: Path) -> None:
    if output.name.endswith(".jsonl"):
        df.to_json(output, lines=True, orient="records")
    elif output.name.endswith(".prototext"):
        ret = {k: v for k, v in df.iloc[0].to_dict().items()}
        ret = to_prototext([ret])
        output.write_text(ret)
    else:
        raise ValueError(f"Can not handle output file format {output}")

@click.option(
    "--truth-leaderboard",
    type=Path,
    required=False,
    help="The ground truth leaderboard file.",
)
@click.option(
    "--truth-measure",
    type=str,
    required=False,
    help="The measure from the ground truth leaderboard to evaluate against.",
)
@click.option(
    "--eval-measure",
    type=str,
    required=False,
    help="The measure from the auto-judge leaderboard to evaluate.",
)
@click.option(
    "--truth-format",
    type=click.Choice(LEADERBOARD_FORMATS),
    required=True,
    help="Format of the ground truth leaderboard file:\n" + LEADERBOARD_FORMAT_HELP,
)
@click.option(
    "--eval-format",
    type=click.Choice(LEADERBOARD_FORMATS),
    required=True,
    help="Format of the input leaderboard file(s):\n" + LEADERBOARD_FORMAT_HELP,
)
@click.option(
    "--truth-header/--no-truth-header",
    default=False,
    help="Truth leaderboard has header row to skip.",
)
@click.option(
    "--eval-header/--no-eval-header",
    default=False,
    help="Eval leaderboard(s) have header row to skip.",
)
@click.option(
    "--on-missing",
    type=click.Choice(["error", "warn", "skip", "default"]),
    default="error",
    help="How to handle run_id mismatches between truth and eval leaderboards: \n"
         "error: raise an error \n"
         "warn: print warning, use common systems only \n"
         "skip: silently use common systems only \n"
         "default: use 0.0 for missing values",
)
@click.option(
    "--input",
    type=Path,
    required=True,
    multiple=True,
    help="The to-be-evaluated leaderboard(s).",
)
@click.option(
    "--output",
    type=Path,
    required=False,
    help="The file where the evaluation should be persisted.",
)
@click.option(
    "--aggregate",
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help="Should only aggregates scores be reported.",
)
def evaluate(
    truth_leaderboard: Optional[Path],
    truth_measure: Optional[str],
    eval_measure: Optional[str],
    truth_format: str,
    truth_header: bool,
    eval_format: str,
    eval_header: bool,
    on_missing: str,
    input: List[Path],
    output: Path,
    aggregate: bool,
) -> int:
    """Evaluate the input leaderboards against the ground-truth leaderboards."""
    # Detect headers interactively if not explicitly specified
    truth_has_header = detect_header_interactive(
        truth_leaderboard, truth_format, truth_header, "truth"
    )

    # For eval files, check the first one and apply to all
    eval_has_header = eval_header
    if input and not eval_header:
        eval_has_header = detect_header_interactive(
            input[0], eval_format, eval_header, "eval"
        )

    te = TrecLeaderboardEvaluation(
        truth_leaderboard,
        truth_measure=truth_measure,
        eval_measure=eval_measure,
        truth_format=truth_format,
        truth_has_header=truth_has_header,
        eval_format=eval_format,
        eval_has_header=eval_has_header,
        on_missing=on_missing,
    )

    df = []

    for c in input:
        result = te.evaluate(c)

        for i in result:
            tmp = {"Judge": c.name.replace(".txt", ""), "Metric": i}
            for k, v in result[i].items():
                tmp[k] = v
            df.append(tmp)

    df = pd.DataFrame(df)

    if aggregate:
        df_aggr = {"Judges": len(df)}
        for k in df.columns:
            if k in ("Judge", "Metric"):
                continue
            df_aggr[k] = df[k].mean()
        df = pd.DataFrame([df_aggr])

    print(df.to_string(index=False))

    if output:
        persist_output(df, output)

    return 0