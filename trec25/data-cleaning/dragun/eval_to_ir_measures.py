#!/usr/bin/env python3
"""Convert custom evaluation TSV/WSV format to ir_measures format.

Input format:  team  run_tag  topic_id  score  (whitespace-separated)
Output format: run_tag  topic_id  measure  score  (ir_measures: 4 cols, tab-separated)

The 'team' column is dropped; measure name is provided via --eval-measure.
"""

import click
import glob as glob_module
from pathlib import Path
from typing import List

from trec_auto_judge.leaderboard.format_detection import detect_format


def read_and_analyze_input(input_path: Path, has_header: bool | None) -> bool:
    """Read input file, detect header, and validate format.

    Args:
        input_path: Path to input file
        has_header: Whether file has a header row (None = auto-detect)

    Returns:
        Detected/confirmed has_header value

    Raises:
        ValueError: If format doesn't match expected 4-column layout
    """
    lines = input_path.read_text().strip().split("\n")
    if not lines:
        raise ValueError(f"Empty file: {input_path}")

    # Auto-detect header if not specified
    if has_header is None:
        hint = detect_format(lines)
        has_header = hint.has_header

    # Validate column count
    start_idx = 1 if has_header else 0
    for i, line in enumerate(lines[start_idx:start_idx + 5], start=start_idx + 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(
                f"Line {i} in {input_path}: expected 4 columns (team run_tag topic_id score), "
                f"got {len(parts)}: {line!r}\n\n"
                f"Expected format: team run_tag topic_id score (whitespace-separated)"
            )

    return has_header


def convert_file(input_path: Path, output_path: Path, measure: str, has_header: bool) -> int:
    """Convert a single file from custom format to ir_measures format.

    Writes per-topic rows plus aggregate "all" rows (mean per run_id).

    Returns the number of lines converted (excluding aggregate rows).
    """
    from collections import defaultdict
    from statistics import mean

    lines_converted = 0
    # Collect scores per run_id for computing aggregates
    run_scores: dict[str, list[float]] = defaultdict(list)

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for i, line in enumerate(infile):
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # Skip header row if present
            if has_header and i == 0:
                continue

            if len(parts) < 4:
                raise ValueError(
                    f"Line {i+1} in {input_path}: expected 4 columns (team run_tag topic_id score), "
                    f"got {len(parts)}: {line}"
                )

            # Input: team, run_tag, topic_id, score
            # Output: run_tag, topic_id, measure, score (ir_measures format)
            _team, run_tag, topic_id, score = parts[0], parts[1], parts[2], parts[3]

            outfile.write(f"{run_tag}\t{topic_id}\t{measure}\t{score}\n")
            run_scores[run_tag].append(float(score))
            lines_converted += 1

        # Write aggregate "all" rows (mean per run_id)
        for run_tag in sorted(run_scores.keys()):
            avg_score = mean(run_scores[run_tag])
            outfile.write(f"{run_tag}\tall\t{measure}\t{avg_score}\n")

    return lines_converted


@click.command()
@click.option(
    "--input", "-i",
    "input_patterns",
    type=str,
    multiple=True,
    help="Input file(s), directory, or glob pattern (e.g., --input '*.txt').",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file or directory for ir_measures format.",
)
@click.option(
    "--eval-measure",
    type=str,
    required=True,
    help="Measure name to use in output (e.g., 'nDCG@10', 'P@5').",
)
@click.option(
    "--input-header/--no-input-header",
    default=None,
    help="Input file(s) have header row to skip. Auto-detected if not specified.",
)
@click.argument("input_files", nargs=-1, type=str)
def main(
    input_patterns: tuple,
    output: Path,
    eval_measure: str,
    input_header: bool,
    input_files: tuple,
) -> None:
    """Convert evaluation results from custom TSV/WSV format to ir_measures format.

    Input format:  team  run_tag  topic_id  score
    Output format: run_tag  topic_id  measure  score

    Examples:

    \b
        # Single file conversion
        python eval_to_ir_measures.py -i results.tsv --output results_ir.txt --eval-measure nDCG@10

    \b
        # Directory conversion (all files)
        python eval_to_ir_measures.py -i ./eval_results/ --output ./ir_format/ --eval-measure nDCG@10

    \b
        # Multiple inputs with glob
        python eval_to_ir_measures.py -i '*.tsv' -i other.txt --output ./ir_format/ --eval-measure P@5
    """
    # Combine --input options and positional arguments, expand globs and directories
    all_inputs: List[Path] = []
    for pattern in list(input_patterns) + list(input_files):
        path = Path(pattern)
        if path.is_dir():
            # Directory: include all files
            all_inputs.extend(sorted(p for p in path.iterdir() if p.is_file()))
        else:
            # Glob pattern or literal file
            matches = sorted(glob_module.glob(pattern, recursive=True))
            if matches:
                all_inputs.extend(Path(m) for m in matches if Path(m).is_file())
            else:
                # No glob match - treat as literal path
                all_inputs.append(path)

    # Filter to files only
    all_inputs = [p for p in all_inputs if p.is_file()]

    if not all_inputs:
        raise click.ClickException("No input files specified. Use --input or positional arguments.")

    # Read first file once: auto-detect header and validate format
    has_header = read_and_analyze_input(all_inputs[0], input_header)
    if input_header is None and has_header:
        click.echo(f"Auto-detected header row in {all_inputs[0].name}", err=True)

    # Determine output mode: single file or directory
    is_multi_input = len(all_inputs) > 1
    output_is_dir = output.is_dir() or (is_multi_input and not output.exists())

    if is_multi_input and not output_is_dir:
        # Multiple inputs require directory output
        output.mkdir(parents=True, exist_ok=True)
        output_is_dir = True
    elif output_is_dir:
        output.mkdir(parents=True, exist_ok=True)
    else:
        # Single file output
        output.parent.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_lines = 0

    for in_file in all_inputs:
        if output_is_dir:
            out_file = output / in_file.name
        else:
            out_file = output

        lines = convert_file(in_file, out_file, eval_measure, has_header)
        total_files += 1
        total_lines += lines
        click.echo(f"Converted {in_file.name}: {lines} lines -> {out_file}")

    click.echo(f"Done: {total_files} files, {total_lines} total lines")


if __name__ == "__main__":
    main()
