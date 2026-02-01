import click
import glob
from pathlib import Path
import pandas as pd
from ..evaluation import LeaderboardEvaluator, CORRELATION_METHODS
from ..click_plus import (
    detect_header_interactive,
    LEADERBOARD_FORMATS,
    LEADERBOARD_FORMAT_HELP,
)
from ..leaderboard import Leaderboard
from typing import List, Set
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
    required=True,
    help="The ground truth leaderboard file or directory.",
)
@click.option(
    "--truth-measure",
    type=str,
    multiple=True,
    help="Measure(s) from truth leaderboard to use. Repeatable. If omitted, uses all.",
)
@click.option(
    "--eval-measure",
    type=str,
    multiple=True,
    help="Measure(s) from eval leaderboard to use. Repeatable. If omitted, uses all.",
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
    "--truth-drop-aggregate/--no-truth-drop-aggregate",
    default=False,
    help="Drop pre-existing aggregate rows from truth and recompute from per-topic data.",
)
@click.option(
    "--eval-drop-aggregate/--no-eval-drop-aggregate",
    default=False,
    help="Drop pre-existing aggregate rows from eval and recompute from per-topic data.",
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
    "--input", "-i",
    type=str,
    multiple=True,
    help="Input leaderboard file(s) or glob pattern (e.g., --input '*.txt').",
)
@click.option(
    "--output",
    type=Path,
    required=False,
    help="Output file path. Format determined by extension: .jsonl for JSON Lines, .prototext for Prototext.",
)
@click.option(
    "--aggregate",
    type=bool,
    required=False,
    default=False,
    is_flag=True,
    help="Should only aggregates scores be reported.",
)
@click.option(
    "--correlation",
    type=click.Choice(CORRELATION_METHODS),
    multiple=True,
    help="Correlation method(s) to compute. Repeatable. If omitted, computes all.",
)
@click.option(
    "--topic-id",
    type=str,
    multiple=True,
    help="Topic ID(s) to use for evaluation. Repeatable. If omitted, uses truth's topics.",
)
@click.option(
    "--topic-ids-from-eval",
    is_flag=True,
    default=False,
    help="Derive topic IDs from the union of topics in eval leaderboards (ignores --topic-id).",
)
@click.argument("input_files", nargs=-1, type=str)
def meta_evaluate(
    truth_leaderboard: Path,
    truth_measure: tuple,
    eval_measure: tuple,
    truth_format: str,
    truth_header: bool,
    eval_format: str,
    eval_header: bool,
    truth_drop_aggregate: bool,
    eval_drop_aggregate: bool,
    on_missing: str,
    input: tuple,
    output: Path,
    aggregate: bool,
    correlation: tuple,
    topic_id: tuple,
    topic_ids_from_eval: bool,
    input_files: tuple,
) -> int:
    """Compute correlation between predicted leaderboards and ground-truth leaderboard."""
    # Combine --input options and positional arguments, expand globs
    all_inputs: List[Path] = []
    for pattern in list(input) + list(input_files):
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            all_inputs.extend(Path(m) for m in matches)
        else:
            # No glob match - treat as literal path
            all_inputs.append(Path(pattern))

    if not all_inputs:
        raise click.ClickException("No input files specified. Use --input or positional arguments.")

    # Detect headers interactively if not explicitly specified
    truth_has_header = detect_header_interactive(
        truth_leaderboard, truth_format, truth_header, "truth"
    )

    # For eval files, check the first one and apply to all
    eval_has_header = eval_header
    if all_inputs and not eval_header:
        eval_has_header = detect_header_interactive(
            all_inputs[0], eval_format, eval_header, "eval"
        )

    # Convert tuples to lists/sets (empty tuple means "all" / None)
    truth_measures = list(truth_measure) if truth_measure else None
    eval_measures = list(eval_measure) if eval_measure else None
    correlation_methods = list(correlation) if correlation else None

    # Determine topic IDs to use
    topic_ids_set: Set[str] | None = None
    if topic_ids_from_eval:
        # Derive from union of eval leaderboard topics
        topic_ids_set = set()
        for eval_path in all_inputs:
            lb = Leaderboard.load(
                eval_path,
                format=eval_format,
                has_header=eval_has_header,
                on_missing="skip",
                drop_aggregate=eval_drop_aggregate,
            )
            # topic_ids property excludes "all" aggregate topic
            topic_ids_set.update(lb.topic_ids)
        click.echo(f"Derived {len(topic_ids_set)} topic IDs from eval leaderboards", err=True)
    elif topic_id:
        topic_ids_set = set(topic_id)

    te = LeaderboardEvaluator(
        truth_leaderboard,
        truth_measures=truth_measures,
        eval_measures=eval_measures,
        truth_format=truth_format,
        truth_has_header=truth_has_header,
        truth_drop_aggregate=truth_drop_aggregate,
        eval_format=eval_format,
        eval_has_header=eval_has_header,
        eval_drop_aggregate=eval_drop_aggregate,
        on_missing=on_missing,
        correlation_methods=correlation_methods,
        topic_ids=topic_ids_set,
    )

    df = []

    for c in all_inputs:
        result = te.evaluate(c)

        for (truth_m, eval_m), metrics in result.items():
            tmp = {
                "Judge": c.name.replace(".txt", ""),
                "TruthMeasure": truth_m,
                "EvalMeasure": eval_m,
            }
            for k, v in metrics.items():
                tmp[k] = v
            df.append(tmp)

    df = pd.DataFrame(df)

    if aggregate:
        df_aggr = {"Judges": len(df)}
        for k in df.columns:
            if k in ("Judge", "TruthMeasure", "EvalMeasure"):
                continue
            df_aggr[k] = df[k].mean()
        df = pd.DataFrame([df_aggr])

    print(df.to_string(index=False))

    if output:
        persist_output(df, output)

    return 0