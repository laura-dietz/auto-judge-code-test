import click
from pathlib import Path
import pandas as pd
import sys
from ..evaluation import TrecLeaderboardEvaluation
from typing import List, Optional
from tira.io_utils import to_prototext


def detect_header_interactive(path: Path, format: str, has_header: bool, label: str) -> bool:
    """
    Check if file has header and prompt user if detected but not specified.

    Args:
        path: Path to leaderboard file
        format: Format string (trec_eval, tot, ir_measures, ranking)
        has_header: User-specified header flag
        label: Label for prompt (e.g., "truth" or "eval")

    Returns:
        has_header value to use
    """
    if has_header:
        return True  # Already specified by user

    if not path or not path.is_file():
        return False

    # Read first line and check if value column is numeric
    try:
        first_line = path.read_text().split("\n")[0].strip()
    except Exception:
        return False

    if not first_line:
        return False

    parts = first_line.split()
    if not parts:
        return False

    # Value is always last column for all formats
    value_col = parts[-1]

    try:
        float(value_col)
        return False  # Looks like data
    except ValueError:
        # Looks like header - prompt user
        print(f"First line of {label} leaderboard looks like header:", file=sys.stderr)
        print(f"  '{first_line}'", file=sys.stderr)
        response = input(f"Skip this header line? [Y/n]: ")
        return response.strip().lower() != 'n'


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
    type=click.Choice(["trec_eval", "ir_measures", "tot", "ranking"]),
    required=True,
    help="Format of the ground truth leaderboard file:\n"
         "  trec_eval: measure topic value (3 cols, run from filename)\n"
         "  ir_measures: run topic measure value (4 cols)\n"
         "  tot: run measure topic value (4 cols)\n"
         "  ranking: topic Q0 doc_id rank score run (6 cols)",
)
@click.option(
    "--eval-format",
    type=click.Choice(["trec_eval", "ir_measures", "tot", "ranking"]),
    required=True,
    help="Format of the input leaderboard file(s):\n"
         "  trec_eval: measure topic value (3 cols, run from filename)\n"
         "  ir_measures: run topic measure value (4 cols)\n"
         "  tot: run measure topic value (4 cols)\n"
         "  ranking: topic Q0 doc_id rank score run (6 cols)",
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