"""
Path resolution utilities for AutoJudge workflow outputs.

Provides consistent file naming conventions:
- Nuggets: {filebase}.nuggets.jsonl
- Judgment: {filebase}.judgment.json + {filebase}.judgment.qrels
- Config: {filebase}.config.yml
"""

from pathlib import Path


def resolve_nugget_file_path(filebase: Path) -> Path:
    """
    Resolve nugget file path from filebase.

    If filebase already has a recognized extension (.jsonl, .json), use as-is.
    Otherwise, append .nuggets.jsonl extension.
    """
    if filebase.suffix in (".jsonl", ".json"):
        return filebase
    return filebase.parent / f"{filebase.name}.nuggets.jsonl"


def resolve_judgment_file_paths(filebase: Path) -> tuple[Path, Path]:
    """
    Resolve leaderboard and qrels file paths from filebase.

    Returns:
        Tuple of (leaderboard_path, qrels_path):
        - {filebase}.judgment.json
        - {filebase}.judgment.qrels
    """
    if filebase.suffix in (".json",):
        # Already has extension, derive qrels from it
        return filebase, filebase.with_suffix(".qrels")
    return (
        filebase.parent / f"{filebase.name}.judgment.json",
        filebase.parent / f"{filebase.name}.judgment.qrels",
    )


def resolve_config_file_path(filebase: Path) -> Path:
    """Resolve config file path: {filebase}.config.yml"""
    if filebase.suffix in (".yml", ".yaml"):
        return filebase
    return filebase.parent / f"{filebase.name}.config.yml"