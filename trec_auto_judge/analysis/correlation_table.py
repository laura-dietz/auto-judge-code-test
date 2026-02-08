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
import yaml
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence


# Meta columns (not metrics) - category columns get added dynamically
BASE_META_COLS = ["Judge", "Dataset", "TruthMeasure", "EvalMeasure"]


def get_meta_columns(df: pd.DataFrame) -> list[str]:
    """Get all non-metric columns present in df."""
    metric_types = (int, float)
    meta = []
    for c in df.columns:
        if c in BASE_META_COLS:
            meta.append(c)
        elif df[c].dtype == object and c not in BASE_META_COLS:
            # String columns added by judges YAML (categories)
            meta.append(c)
    return meta


def get_metric_columns(df: pd.DataFrame) -> list[str]:
    """Extract metric columns from DataFrame (everything except meta columns)."""
    meta = get_meta_columns(df)
    return [c for c in df.columns if c not in meta]


def load_judges_yaml(path: Path) -> dict[str, dict]:
    """Load judges YAML file.

    Returns:
        Dict mapping cryptic judge name -> {name, category1, category2, ...}
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("judges", data)


def apply_judges(df: pd.DataFrame, judges: dict[str, dict], all_judges: bool = False) -> pd.DataFrame:
    """Apply judge mappings: rename judges, add category columns, sort.

    Category columns (everything except 'name') are added and used as sort keys.
    Unless all_judges=True, drops judges not defined in the YAML.
    """
    df = df.copy()

    # Filter to only defined judges (unless --all-judges)
    if not all_judges:
        df = df[df["Judge"].isin(judges.keys())]

    # Discover category dimensions from YAML (all keys except 'name')
    all_categories = set()
    for info in judges.values():
        all_categories.update(k for k in info if k != "name")
    category_cols = sorted(all_categories)

    # Add category columns
    for cat in category_cols:
        df[cat] = df["Judge"].map(lambda j, c=cat: judges.get(j, {}).get(c, ""))

    # Sort by YAML definition order (before renaming, so we match on original keys)
    judge_order = {j: i for i, j in enumerate(judges.keys())}
    df["_sort_key"] = df["Judge"].map(lambda j: judge_order.get(j, len(judge_order)))
    df = df.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)

    # Rename judges
    name_map = {j: info["name"] for j, info in judges.items() if "name" in info}
    df["Judge"] = df["Judge"].map(lambda j: name_map.get(j, j))

    return df


def category_comparison(
    df: pd.DataFrame,
    judges: dict[str, dict],
    metric_cols: list[str],
    fmt: str = "github",
    same_threshold: float = 0.01,
) -> str:
    """Compare average correlation by category dimension.

    For each category dimension (e.g., 'graded'), groups judges by their
    category value (e.g., 'docs' vs 'response') and shows average metrics.
    """
    # Discover categories
    all_categories = set()
    for info in judges.values():
        all_categories.update(k for k in info if k != "name")

    sections = []

    # Group by (Dataset, TruthMeasure, EvalMeasure) then by category
    group_keys = [c for c in ["Dataset", "TruthMeasure", "EvalMeasure"] if c in df.columns]

    for cat in sorted(all_categories):
        if cat not in df.columns:
            continue

        sections.append(f"## Category: {cat}\n")

        for group_vals, group_df in df.groupby(group_keys, sort=True):
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            header_parts = [f"{k}={v}" for k, v in zip(group_keys, group_vals)]
            sections.append(f"#### {', '.join(header_parts)}\n")

            # Average metrics by category value
            avail_metrics = [m for m in metric_cols if m in group_df.columns]
            avg_df = group_df.groupby(cat)[avail_metrics].max().reset_index()
            avg_df = avg_df.rename(columns={cat: cat.title()})

            sections.append(format_table(
                avg_df, fmt, highlight_max=True,
                metric_cols=avail_metrics, same_threshold=same_threshold,
            ))
            sections.append("")

    return "\n".join(sections)


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


def bold_max(
    df: pd.DataFrame,
    metric_cols: list[str],
    fmt: str = "github",
    same_threshold: float = 0.05,
) -> pd.DataFrame:
    """Return a copy with string-formatted values, max and near-max bolded.

    Values within `same_threshold` of the column max are considered
    statistically indistinguishable and also bolded.

    For github/pipe markdown: **0.8182**
    For latex: \\textbf{0.8182}
    """
    out = df.copy()
    for col in metric_cols:
        if col not in out.columns:
            continue
        max_val = out[col].max()
        cutoff = max_val - same_threshold
        if fmt in ("latex", "latex_raw", "latex_booktabs"):
            out[col] = out[col].apply(
                lambda v, c=cutoff: f"\\textbf{{{v:.4f}}}" if v >= c else f"{v:.4f}"
            )
        else:
            out[col] = out[col].apply(
                lambda v, c=cutoff: f"**{v:.4f}**" if v >= c else f"{v:.4f}"
            )
    return out


def format_table(
    df: pd.DataFrame,
    fmt: str = "github",
    float_format: str = ".4f",
    highlight_max: bool = False,
    metric_cols: Optional[list[str]] = None,
    same_threshold: float = 0.02,
) -> str:
    """Format DataFrame as table string using pandas to_markdown."""
    if highlight_max and metric_cols:
        df = bold_max(df, metric_cols, fmt, same_threshold)
        # Already string-formatted, don't apply floatfmt
        return df.to_markdown(index=False, tablefmt=fmt)
    return df.to_markdown(index=False, tablefmt=fmt, floatfmt=float_format)


def correlation_consistency(
    df: pd.DataFrame,
    metric_cols: list[str],
    fmt: str = "github",
    same_threshold: float = 0.01,
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

        # Select row-header cols (categories + Judge) + metric columns
        meta = get_meta_columns(group_df)
        row_cols = [c for c in meta if c not in group_keys]
        table_cols = row_cols + [c for c in metric_cols if c in group_df.columns]
        out_df = group_df[table_cols].copy()

        sections.append(format_table(out_df, fmt, highlight_max=True, metric_cols=metric_cols, same_threshold=same_threshold))
        sections.append("")

    return "\n".join(sections)


def measures_as_columns(
    df: pd.DataFrame,
    metric_cols: list[str],
    fmt: str = "github",
    same_threshold: float = 0.01,
) -> str:
    """One table per (Dataset, correlation metric).

    Rows: Judge
    Columns: TruthMeasure/EvalMeasure combinations
    Highest value per column bolded.
    """
    sections = []

    # Group by Dataset (if present) then iterate correlation metrics
    dataset_keys = ["Dataset"] if "Dataset" in df.columns else []

    for metric in metric_cols:
        if metric not in df.columns:
            continue

        if dataset_keys:
            for dataset, ds_df in df.groupby("Dataset", sort=True):
                sections.append(f"#### Dataset={dataset}, Correlation={metric}\n")
                sections.append(_pivot_measures_table(ds_df, metric, fmt, same_threshold))
                sections.append("")
        else:
            sections.append(f"#### Correlation={metric}\n")
            sections.append(_pivot_measures_table(df, metric, fmt, same_threshold))
            sections.append("")

    return "\n".join(sections)


def _pivot_measures_table(
    df: pd.DataFrame,
    metric: str,
    fmt: str,
    same_threshold: float,
) -> str:
    """Pivot a single metric into columns of TruthMeasure/EvalMeasure."""
    # Create composite column name
    df = df.copy()
    df["_measure_col"] = df["TruthMeasure"] + " / " + df["EvalMeasure"]

    # Get row identity columns (Judge + any category cols)
    meta = get_meta_columns(df)
    row_id_cols = [c for c in meta if c not in ["Dataset", "TruthMeasure", "EvalMeasure", "_measure_col"]]

    # Pivot
    pivoted = df.pivot_table(
        index=row_id_cols,
        columns="_measure_col",
        values=metric,
        aggfunc="first",
    ).reset_index()

    # Flatten MultiIndex columns if needed
    pivoted.columns.name = None

    # Preserve judge order from the original df
    judge_order = {j: i for i, j in enumerate(df["Judge"].unique())}
    pivoted["_sort"] = pivoted["Judge"].map(lambda j: judge_order.get(j, len(judge_order)))
    pivoted = pivoted.sort_values("_sort").drop(columns=["_sort"])

    measure_cols = [c for c in pivoted.columns if c not in row_id_cols]
    return format_table(pivoted, fmt, highlight_max=True, metric_cols=measure_cols, same_threshold=same_threshold)


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
    "--judges", "-j",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="YAML file mapping judge names to nice names and categories.",
)
@click.option(
    "--all-judges/--no-all-judges",
    default=False,
    help="Include judges not defined in --judges YAML (default: only defined judges).",
)
@click.option(
    "--columns",
    type=click.Choice(["correlations", "measures"]),
    default="correlations",
    help="What to show as columns: 'correlations' (one table per dataset/truth/eval) "
         "or 'measures' (one table per dataset/correlation metric).",
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
    "--same",
    type=float,
    default=0.01,
    help="Threshold for 'statistically same': values within this of the max are also bolded.",
)
@click.option(
    "--different",
    type=float,
    default=0.1,
    help="Threshold for 'statistically different' (reserved for future use).",
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
    judges: Optional[Path],
    all_judges: bool,
    columns: str,
    aggregate: bool,
    correlation: tuple,
    truth_measure: tuple,
    eval_measures: tuple,
    same: float,
    different: float,
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

    # Load and apply judges YAML
    judges_data = None
    if judges:
        judges_data = load_judges_yaml(judges)
        df = apply_judges(df, judges_data, all_judges=all_judges)

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

    # Produce tables based on --columns mode
    if columns == "measures":
        result = measures_as_columns(df, metric_cols, format, same_threshold=same)
    else:
        result = correlation_consistency(df, metric_cols, format, same_threshold=same)

    # Append category comparison if judges YAML provides categories
    if judges_data:
        result += "\n\n---\n\n# Category Comparison\n\n"
        result += category_comparison(df, judges_data, metric_cols, format, same_threshold=same)

    # Output
    if output:
        output.write_text(result)
        click.echo(f"Wrote to {output}", err=True)
    else:
        click.echo(result)

    return 0


if __name__ == "__main__":
    main()
