"""
Workflow declaration for AutoJudge nugget/judge pipelines.

Participants declare their workflow in workflow.yml to enable TIRA orchestration.
"""

from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, Field


# Built-in NuggetBanks type paths (for convenience)
NUGGET_BANKS_AUTOARGUE = "trec_auto_judge.nugget_data.NuggetBanks"
NUGGET_BANKS_NUGGETIZER = "trec_auto_judge.nugget_data.NuggetizerNuggetBanks"

# Default type
DEFAULT_NUGGET_BANKS_TYPE = NUGGET_BANKS_AUTOARGUE


class Workflow(BaseModel):
    """
    Workflow configuration loaded from workflow.yml.

    Controls which steps are executed:
    - create_nuggets: Whether to call create_nuggets() to generate/refine nuggets
    - judge: Whether to call judge() to produce leaderboard/qrels

    Settings are passed to AutoJudge methods as **kwargs:
    - settings: Shared settings for both phases (fallback)
    - nugget_settings: Override for create_nuggets()
    - judge_settings: Override for judge()

    Example workflow.yml:
        create_nuggets: true
        judge: true

        settings:
          filebase: "{_name}"
          top_k: 20

        nugget_settings:
          extraction_style: "thorough"

        judge_settings:
          threshold: 0.5

        variants:
          strict:
            threshold: 0.8

        sweeps:
          top-k-sweep:
            top_k: [10, 20, 50]
    """

    create_nuggets: bool = False
    """Whether to call create_nuggets() to generate/refine nuggets."""

    judge: bool = True
    """Whether to call judge() to produce leaderboard/qrels."""

    nugget_banks_type: str = DEFAULT_NUGGET_BANKS_TYPE
    """Dotted import path for NuggetBanks container class."""

    nugget_input: Optional[str] = None
    """Path to existing nugget banks to load (for refinement or judge input)."""

    nugget_output: Optional[str] = None
    """Path to store created/refined nugget banks."""

    # Settings dicts passed to AutoJudge methods as **kwargs
    settings: dict[str, Any] = Field(default_factory=dict)
    """Shared settings passed to both create_nuggets() and judge()."""

    nugget_settings: dict[str, Any] = Field(default_factory=dict)
    """Settings passed to create_nuggets(), merged over shared settings."""

    judge_settings: dict[str, Any] = Field(default_factory=dict)
    """Settings passed to judge(), merged over shared settings."""

    # Named configurations and parameter sweeps
    variants: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Named configurations that override settings. Key = variant name."""

    sweeps: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Parameter sweeps. Values with lists expand to multiple configurations."""

    # Lifecycle flags
    nugget_depends_on_responses: bool = True
    """If True, pass responses to create_nuggets(). If False, pass None."""

    judge_uses_nuggets: bool = True
    """If True, pass nuggets to judge(). If False, pass None."""

    force_recreate_nuggets: bool = False
    """If True, recreate nuggets even if file exists. CLI can override."""


def load_workflow(source: Union[str, Path]) -> Workflow:
    """
    Load workflow configuration from a YAML file.

    Args:
        source: Path to workflow.yml

    Returns:
        Workflow configuration

    Example workflow.yml:
        create_nuggets: true
        judge: true
    """
    path = Path(source)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return Workflow.model_validate(data)


def load_workflow_from_directory(directory: Union[str, Path]) -> Optional[Workflow]:
    """
    Load workflow.yml from a judge directory if it exists.

    Args:
        directory: Judge directory (e.g., trec25/judges/my-judge/)

    Returns:
        Workflow if workflow.yml exists, None otherwise
    """
    path = Path(directory) / "workflow.yml"
    if path.is_file():
        return load_workflow(path)
    return None


# Default workflow for judges that don't declare one (judge only, no nuggets)
DEFAULT_WORKFLOW = Workflow()