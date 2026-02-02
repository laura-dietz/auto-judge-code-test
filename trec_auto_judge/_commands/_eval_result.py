"""EvalResult round-trip testing command."""

import click
import tempfile
from pathlib import Path
from typing import Optional, Set

from trec_auto_judge.eval_results import load, write, EvalResult, ALL_TOPIC_ID


EVAL_RESULT_FORMATS = ["trec_eval", "tot", "ir_measures", "ranking", "jsonl"]

FORMAT_HELP = """
trec_eval: measure topic_id value (3 cols, run_id from filename)
tot: run_id measure topic_id value (4 cols)
ir_measures: run_id topic_id measure value (4 cols)
ranking: topic_id Q0 run_id rank value measure (6 cols, TREC ranking format)
jsonl: JSON lines with run_id, topic_id, measure, value
"""


@click.command("eval-result")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--input-format", "-if",
    type=click.Choice(EVAL_RESULT_FORMATS),
    required=True,
    help="Input format." + FORMAT_HELP,
)
@click.option(
    "--output-format", "-of",
    type=click.Choice(EVAL_RESULT_FORMATS),
    default=None,
    help="Output format. Defaults to input format.",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path. If omitted, uses temp file for roundtrip test.",
)
@click.option(
    "--has-header/--no-header",
    default=False,
    help="Input file has header row to skip.",
)
@click.option(
    "--drop-aggregates/--keep-aggregates",
    default=False,
    help="Drop existing aggregate rows before processing.",
)
@click.option(
    "--recompute-aggregates/--no-recompute",
    default=False,
    help="Recompute aggregates from per-topic data (implies --drop-aggregates).",
)
@click.option(
    "--verify/--no-verify",
    default=True,
    help="Run verification checks on loaded data.",
)
@click.option(
    "--on-missing",
    type=click.Choice(["error", "warn", "ignore"]),
    default="warn",
    help="How to handle missing entries.",
)
@click.option(
    "--filter-runs",
    type=str,
    multiple=True,
    help="Run IDs to keep (repeatable). Implies --recompute-aggregates.",
)
@click.option(
    "--filter-topics",
    type=str,
    multiple=True,
    help="Topic IDs to keep (repeatable). Implies --recompute-aggregates.",
)
@click.option(
    "--filter-measures",
    type=str,
    multiple=True,
    help="Measures to keep (repeatable). Filters both per-topic and aggregate entries.",
)
@click.option(
    "--roundtrip/--no-roundtrip",
    default=True,
    help="Perform roundtrip test (write then reload and compare).",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed output.",
)
def eval_result(
    input_path: Path,
    input_format: str,
    output_format: Optional[str],
    output: Optional[Path],
    has_header: bool,
    drop_aggregates: bool,
    recompute_aggregates: bool,
    verify: bool,
    on_missing: str,
    filter_runs: tuple[str, ...],
    filter_topics: tuple[str, ...],
    filter_measures: tuple[str, ...],
    roundtrip: bool,
    verbose: bool,
) -> int:
    """Load, convert, and roundtrip-test EvalResult files.

    INPUT_PATH is the input file or directory to load.

    Examples:

        # Roundtrip test (load, write to temp, reload, compare)
        trec-auto-judge eval-result data.txt --input-format tot

        # Convert tot to jsonl
        trec-auto-judge eval-result data.txt -if tot -of jsonl -o data.jsonl

        # Load directory of trec_eval files
        trec-auto-judge eval-result ./runs/ -if trec_eval -v
    """
    output_format = output_format or input_format

    # recompute_aggregates implies drop_aggregates
    if recompute_aggregates:
        drop_aggregates = True

    click.echo(f"Loading {input_path} (format: {input_format})", err=True)

    # Load input
    original = load(
        input_path,
        format=input_format,
        has_header=has_header,
        drop_aggregates=drop_aggregates,
        recompute_aggregates=recompute_aggregates,
        verify=verify,
        on_missing=on_missing,
    )

    _print_summary(original, "Loaded", verbose)

    # Apply filters if specified
    run_ids_filter: Set[str] | None = None
    topic_ids_filter: Set[str] | None = None

    if filter_runs:
        run_ids_filter = set(filter_runs)
        click.echo(f"Filtering to runs: {run_ids_filter}", err=True)

    if filter_topics:
        topic_ids_filter = set(filter_topics)
        click.echo(f"Filtering to topics: {topic_ids_filter}", err=True)

    if run_ids_filter or topic_ids_filter:
        original = original.filter_and_recompute(
            run_ids=run_ids_filter,
            topic_ids=topic_ids_filter,
        )
        _print_summary(original, "Filtered (runs/topics)", verbose)

    # Filter measures (simple entry filtering, no recompute needed)
    if filter_measures:
        measures_filter = set(filter_measures)
        click.echo(f"Filtering to measures: {measures_filter}", err=True)
        filtered_entries = tuple(e for e in original.entries if e.measure in measures_filter)
        filtered_specs = original.specs.filter(measures_filter)
        original = EvalResult(entries=filtered_entries, specs=filtered_specs)
        _print_summary(original, "Filtered (measures)", verbose)

    # Determine output path
    if output:
        out_path = output
        cleanup = False
    else:
        # Use temp file for roundtrip
        suffix = ".jsonl" if output_format == "jsonl" else ".txt"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        out_path = Path(tmp.name)
        tmp.close()
        cleanup = True

    # Write output
    click.echo(f"Writing to {out_path} (format: {output_format})", err=True)
    write(original, out_path, format=output_format)

    # Roundtrip test
    if roundtrip:
        click.echo(f"Reloading for roundtrip test...", err=True)
        reloaded = load(
            out_path,
            format=output_format,
            has_header=False,  # Written files never have headers
            drop_aggregates=drop_aggregates,
            recompute_aggregates=recompute_aggregates,
            verify=verify,
            on_missing=on_missing,
        )

        _print_summary(reloaded, "Reloaded", verbose)

        # Compare
        errors = _compare_results(original, reloaded, verbose)
        if errors:
            click.echo(click.style(f"FAILED: {len(errors)} differences found", fg="red"), err=True)
            for err in errors[:10]:
                click.echo(f"  {err}", err=True)
            if len(errors) > 10:
                click.echo(f"  ... and {len(errors) - 10} more", err=True)
            return 1
        else:
            click.echo(click.style("PASSED: Roundtrip successful", fg="green"), err=True)

    # Cleanup temp file if not user-specified output
    if cleanup and not output:
        out_path.unlink()

    return 0


def _print_summary(result: EvalResult, label: str, verbose: bool) -> None:
    """Print summary of EvalResult."""
    n_entries = len(result.entries)
    n_runs = len(result.run_ids)
    n_topics = len(result.topic_ids)
    n_measures = len(result.measures)
    has_agg = result.has_aggregates

    click.echo(f"{label}: {n_entries} entries, {n_runs} runs, {n_topics} topics, {n_measures} measures", err=True)
    click.echo(f"  Has aggregates: {has_agg}", err=True)
    click.echo(f"  Dtypes: {dict(result.measure_dtypes)}", err=True)

    if verbose:
        click.echo(f"  Runs: {sorted(result.run_ids)}", err=True)
        click.echo(f"  Topics: {sorted(result.topic_ids)}", err=True)
        click.echo(f"  Measures: {sorted(result.measures)}", err=True)


def _compare_results(original: EvalResult, reloaded: EvalResult, verbose: bool) -> list[str]:
    """Compare two EvalResults and return list of differences."""
    errors = []

    # Check run_ids
    if original.run_ids != reloaded.run_ids:
        errors.append(f"run_ids differ: {original.run_ids} vs {reloaded.run_ids}")

    # Check topic_ids
    if original.topic_ids != reloaded.topic_ids:
        errors.append(f"topic_ids differ: {original.topic_ids} vs {reloaded.topic_ids}")

    # Check measures
    if original.measures != reloaded.measures:
        errors.append(f"measures differ: {original.measures} vs {reloaded.measures}")

    # Check entry count
    if len(original.entries) != len(reloaded.entries):
        errors.append(f"entry count differs: {len(original.entries)} vs {len(reloaded.entries)}")

    # Check individual values
    for run_id in original.run_ids:
        for topic_id in list(original.topic_ids) + [ALL_TOPIC_ID]:
            for measure in original.measures:
                orig_val = original.get_value(run_id, topic_id, measure)
                reload_val = reloaded.get_value(run_id, topic_id, measure)

                if orig_val is None and reload_val is None:
                    continue

                if orig_val is None or reload_val is None:
                    errors.append(f"{run_id}/{topic_id}/{measure}: {orig_val} vs {reload_val}")
                    continue

                # Compare values (float comparison with tolerance)
                if isinstance(orig_val, float) and isinstance(reload_val, float):
                    if abs(orig_val - reload_val) > 1e-9:
                        errors.append(f"{run_id}/{topic_id}/{measure}: {orig_val} vs {reload_val}")
                elif orig_val != reload_val:
                    errors.append(f"{run_id}/{topic_id}/{measure}: {orig_val!r} vs {reload_val!r}")

    return errors