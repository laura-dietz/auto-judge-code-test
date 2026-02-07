"""
Correlation analysis tables from meta-evaluate JSONL output.

Analyzes how well different AutoJudge configurations correlate with ground truth
across datasets (ragtime, rag, dragun).

Usage:
    python -m trec_auto_judge.analysis.correlation_table \\
        --dataset ragtime:ragtime.jsonl \\
        --dataset rag:rag.jsonl \\
        --dataset dragun:dragun.jsonl
"""

import click
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence


# Meta columns (not metrics)
META_COLS = ["Judge", "Dataset", "TruthMeasure", "EvalMeasure"]


def get_metric_columns(df: pd.DataFrame) -> list[str]:
    """Extract metric columns from DataFrame (everything except meta columns)."""
    return [c for c in df.columns if c not in META_COLS]


def load_dataset(path: Path, label: str) -> pd.DataFrame:
    """Load correlation results from JSONL file."""
    df = pd.read_json(path, lines=True)
    df["Dataset"] = label
    return df


def load_datasets(dataset_specs: Sequence[str]) -> pd.DataFrame:
    """Load multiple datasets from 'label:path' specs.

    Args:
        dataset_specs: List of 'label:path' strings, or just 'path' (uses stem as label)
    """
    dfs = []
    for spec in dataset_specs:
        if ":" in spec:
            label, path_str = spec.split(":", 1)
        else:
            path_str = spec
            label = Path(path_str).stem
        dfs.append(load_dataset(Path(path_str), label))
    return pd.concat(dfs, ignore_index=True)


def aggregate_by_judge(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Aggregate correlation metrics across datasets per judge.

    Groups by (Judge, TruthMeasure, EvalMeasure), averages metrics.
    """
    if metrics is None:
        metrics = get_metric_columns(df)

    # Filter to metrics that exist
    available = [m for m in metrics if m in df.columns]

    group_cols = ["Judge", "TruthMeasure", "EvalMeasure"]
    return df.groupby(group_cols)[available].mean().reset_index()


def bold_max(df: pd.DataFrame, metric_cols: list[str], fmt: str = "github") -> pd.DataFrame:
    """Return a copy with string-formatted values, max per column bolded.

    For github/pipe markdown: **0.8182**
    For latex: \\textbf{0.8182}
    """
    out = df.copy()
    for col in metric_cols:
        if col not in out.columns:
            continue
        max_val = out[col].max()
        if fmt in ("latex", "latex_raw", "latex_booktabs"):
            out[col] = out[col].apply(
                lambda v: f"\\textbf{{{v:.4f}}}" if v == max_val else f"{v:.4f}"
            )
        else:
            out[col] = out[col].apply(
                lambda v: f"**{v:.4f}**" if v == max_val else f"{v:.4f}"
            )
    return out


def format_table(
    df: pd.DataFrame,
    fmt: str = "github",
    float_format: str = ".4f",
    highlight_max: bool = False,
    metric_cols: Optional[list[str]] = None,
) -> str:
    """Format DataFrame as table string using pandas to_markdown."""
    if highlight_max and metric_cols:
        df = bold_max(df, metric_cols, fmt)
        # Already string-formatted, don't apply floatfmt
        return df.to_markdown(index=False, tablefmt=fmt)
    return df.to_markdown(index=False, tablefmt=fmt, floatfmt=float_format)


def correlation_consistency(
    df: pd.DataFrame,
    metric_cols: list[str],
    fmt: str = "github",
) -> str:
    """One table per (Dataset, TruthMeasure, EvalMeasure).

    Rows: Judge
    Columns: correlation metrics
    Highest value per column bolded.
    """
    sections = []
    group_keys = [c for c in ["Dataset", "TruthMeasure", "EvalMeasure"] if c in df.columns]

    for group_vals, group_df in df.groupby(group_keys, sort=True):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        header_parts = [f"{k}={v}" for k, v in zip(group_keys, group_vals)]
        sections.append(f"#### {', '.join(header_parts)}\n")

        # Select Judge + metric columns only
        table_cols = ["Judge"] + [c for c in metric_cols if c in group_df.columns]
        out_df = group_df[table_cols].copy()

        sections.append(format_table(out_df, fmt, highlight_max=True, metric_cols=metric_cols))
        sections.append("")

    return "\n".join(sections)


@click.command()
@click.option(
    "--dataset", "-d",
    "datasets",
    type=str,
    multiple=True,
    help="Dataset as 'label:path' or just 'path'. Repeatable.",
)
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--format", "-f",
    type=click.Choice(["github", "latex", "tsv", "plain", "html", "pipe"]),
    default="github",
    help="Table format.",
)
@click.option(
    "--aggregate/--no-aggregate",
    default=False,
    help="Average metrics across datasets (per judge) before tabulating.",
)
@click.option(
    "--correlation",
    type=str,
    multiple=True,
    help="Correlation metric columns to include (e.g., kendall, spearman@5). Repeatable. Default: all.",
)
@click.option(
    "--truth-measure", "-t",
    type=str,
    multiple=True,
    help="Filter to specific TruthMeasure(s). Repeatable.",
)
@click.option(
    "--eval-measure", "-e",
    "eval_measures",
    type=str,
    multiple=True,
    help="Filter to specific EvalMeasure(s) (e.g., AVG_GRADE). Repeatable.",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output file. If omitted, prints to stdout.",
)
def main(
    datasets: tuple,
    files: tuple,
    format: str,
    aggregate: bool,
    correlation: tuple,
    truth_measure: tuple,
    eval_measures: tuple,
    output: Optional[Path],
):
    """Generate correlation consistency tables from meta-evaluate JSONL output.

    Produces one table per (Dataset, TruthMeasure, EvalMeasure) combination,
    with judges as rows, correlation metrics as columns, and the best value
    per column highlighted in bold.

    Examples:
        # Single file
        correlation_table results.jsonl

        # Multiple labeled datasets
        correlation_table -d ragtime:rt.jsonl -d rag:r.jsonl

        # Filter by measures
        correlation_table file.jsonl -t nugget_coverage -e AVG_GRADE

        # Select specific correlation metrics
        correlation_table file.jsonl --correlation kendall --correlation spearman

        # Aggregate across datasets, then tabulate
        correlation_table -d rt:rt.jsonl -d rag:rag.jsonl --aggregate
    """
    # Combine --dataset specs and positional files
    all_specs = list(datasets) + list(files)
    if not all_specs:
        click.echo("No input files specified.", err=True)
        return 1

    # Load data
    df = load_datasets(all_specs)

    # Filter rows by TruthMeasure if specified
    if truth_measure:
        df = df[df["TruthMeasure"].isin(truth_measure)]

    # Filter rows by EvalMeasure if specified
    if eval_measures:
        df = df[df["EvalMeasure"].isin(eval_measures)]

    # Select correlation metric columns
    if correlation:
        metric_cols = list(correlation)
    else:
        metric_cols = get_metric_columns(df)

    # Optionally aggregate across datasets first
    if aggregate:
        df = aggregate_by_judge(df, metrics=metric_cols)

    # Produce correlation_consistency tables
    result = correlation_consistency(df, metric_cols, format)

    # Output
    if output:
        output.write_text(result)
        click.echo(f"Wrote to {output}", err=True)
    else:
        click.echo(result)

    return 0


if __name__ == "__main__":
    main()
