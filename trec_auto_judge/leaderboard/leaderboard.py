from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Set

from trec_auto_judge.utils import format_preview
from .verification import LeaderboardVerification, LeaderboardVerificationError
from .format_detection import format_error_with_hint

MeasureName = str
AggFn = Callable[[Sequence[Any]], Any]
CastFn = Callable[[Any], Any]
OnMissing = Literal["default", "warn", "error", "fix_aggregate"]
LeaderboardFormat = Literal["trec_eval", "tot", "ir_measures", "ranking"]


#  ==== DataClasses for data storage and serialization ===  

@dataclass(frozen=True)
class LeaderboardEntry:
    """One row in a leaderboard: (run_id, topic_id) plus a mapping of measure -> value."""
    run_id: str
    topic_id: str
    values: Dict[MeasureName, Any]


@dataclass(frozen=True)
class Leaderboard:
    """
    Thin serialization vessel for leaderboard results.

    - `measures` defines the measure names.
    - `entries` contains per-topic rows and and per-measure `all_topic_id` rows.
    
    Developer note:
    - Aggregation logic lives in LeaderboardBuilder.
    """
    measures: Tuple[MeasureName, ...]
    entries: Tuple[LeaderboardEntry, ...]
    all_topic_id: str = "all"
    spec: Optional["LeaderboardSpec"] = None

    def all_measure_names(self) -> Tuple[MeasureName, ...]:
        """Return measure names in schema order."""
        return self.measures

    def write(
        self,
        output: Path,
        format: LeaderboardFormat = "tot",
    ) -> None:
        """
        Write the leaderboard as tab-separated lines.

        Args:
            output: Path to write to
            format: Column order
                - "trec_eval": measure topic value
                - "tot": run measure topic value
                - "ir_measures": run topic measure value

        Only measures present in each entry are written (allows sparse rows).
        """
        lines: List[str] = []
        for e in self.entries:
            for m in self.all_measure_names():
                if m in e.values:
                    if format == "tot":
                        lines.append("\t".join([e.run_id, m, e.topic_id, str(e.values[m])]))
                    elif format == "trec_eval":
                        lines.append("\t".join([m, e.topic_id, str(e.values[m])]))
                    elif format == "ir_measures":
                        lines.append("\t".join([e.run_id, e.topic_id, m, str(e.values[m])]))
                    else:
                        raise ValueError(f"Unknown format: {format!r}")

        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing leaderboard to {output.absolute()}")   # ToDo: use a logger
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: Path,
        format: LeaderboardFormat,
        has_header: bool = False,
        on_missing: OnMissing = "error",
        drop_aggregate: bool = False,
    ) -> "Leaderboard":
        """
        Load a leaderboard from file or directory.

        Args:
            path: Path to leaderboard file or directory of files
            format: Column order (whitespace-separated)
                - "trec_eval": measure topic value (3 cols, run_id from filename)
                - "tot": run measure topic value (4 cols)
                - "ir_measures": run topic measure value (4 cols)
                - "ranking": topic Q0 doc_id rank score run (6 cols)
            has_header: If True, skip the first line (header row)
            drop_aggregate: If True, drop pre-existing "all" entries and recompute

        If path is a directory, all files are loaded and merged into a single
        leaderboard. For trec_eval format, each file's name becomes the run_id.
        """
        if path.is_dir():
            lb = cls._load_directory(path, format, has_header)
        else:
            lb = cls._load_file(path, format, has_header)

        # Always go through the builder to ensure consistent casting and aggregate computation
        result = cls._build_from_entries(lb, drop_aggregate=drop_aggregate, on_missing=on_missing)

        # Verify all entries have all measures
        LeaderboardVerification(result, on_missing=on_missing, warn=(on_missing != "error")).complete_measures(include_all_row=False)

        return result

    @classmethod
    def _load_directory(
        cls,
        directory: Path,
        format: LeaderboardFormat,
        has_header: bool,
    ) -> "Leaderboard":
        """Load and merge all leaderboard files from a directory."""
        files = sorted([f for f in directory.iterdir() if f.is_file() and not f.name.startswith(".")])
        if not files:
            raise ValueError(f"No files found in directory: {directory}")
        print(f"Loading {len(files)} files from {directory}", file=sys.stderr)

        leaderboards = [cls._load_file(f, format, has_header) for f in files]
        spec = leaderboards[0].spec_without_aggregate()
        return LeaderboardBuilder(spec).add_from_all(leaderboards).build()

    @classmethod
    def _load_file(
        cls,
        path: Path,
        format: LeaderboardFormat,
        has_header: bool,
    ) -> "Leaderboard":
        """Load a leaderboard from a single file."""
        text = path.read_text(encoding="utf-8")
        lines = text.strip().split("\n")

        if has_header and lines:
            lines = lines[1:]

        # Collect values grouped by (run_id, topic_id)
        entry_values: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
        measure_names: List[str] = []
        measure_set: Set[str] = set()

        if format == "ranking":
            print(
                "Warning: Loading ranking format with semantic mapping:\n"
                "  ranking doc_id -> leaderboard run_id\n"
                "  ranking run_id -> leaderboard measure\n"
                "  ranking score -> leaderboard value\n"
                "  ranking rank -> ignored",
                file=sys.stderr,
            )

        for line in lines:
            if not line:
                continue
            parts = line.split()

            if format == "trec_eval":
                if len(parts) != 3:
                    raise ValueError(format_error_with_hint("trec_eval", 3, len(parts), line, lines))
                measure, topic_id, value = parts
                run_id = path.name
            elif format == "tot":
                if len(parts) != 4:
                    raise ValueError(format_error_with_hint("tot", 4, len(parts), line, lines))
                run_id, measure, topic_id, value = parts
            elif format == "ir_measures":
                if len(parts) != 4:
                    raise ValueError(format_error_with_hint("ir_measures", 4, len(parts), line, lines))
                run_id, topic_id, measure, value = parts
            elif format == "ranking":
                if len(parts) != 6:
                    raise ValueError(format_error_with_hint("ranking", 6, len(parts), line, lines))
                topic_id, _q0, run_id, _rank, value, measure = parts
            else:
                raise ValueError(f"Unknown format: {format!r}")

            entry_values[(run_id, topic_id)][measure] = value
            if measure not in measure_set:
                measure_names.append(measure)
                measure_set.add(measure)

        # Build entries
        entries = [
            LeaderboardEntry(run_id=run_id, topic_id=topic_id, values=values)
            for (run_id, topic_id), values in entry_values.items()
        ]

        # Infer dtype for each measure from values
        measure_specs = []
        for m in measure_names:
            dtype = _infer_dtype_from_values([e.values.get(m) for e in entries if m in e.values])
            measure_specs.append(MeasureSpec(name=m, dtype=dtype))

        spec = LeaderboardSpec(measures=tuple(measure_specs))

        return cls(
            measures=tuple(measure_names),
            entries=tuple(entries),
            spec=spec,
        )

    @classmethod
    def _build_from_entries(
        cls,
        lb: "Leaderboard",
        drop_aggregate: bool,
        on_missing: OnMissing,
    ) -> "Leaderboard":
        """
        Central method for building a leaderboard from raw entries.

        This ensures consistent casting and aggregate computation for both
        load() and filter_and_recompute().

        Args:
            lb: Source leaderboard with raw entries
            drop_aggregate: If True, filter out pre-existing "all" entries and recompute
            on_missing: Policy for handling missing (run_id, topic_id) pairs
        """
        # Get appropriate spec based on drop_aggregate
        spec = lb.spec_without_aggregate() if drop_aggregate else lb.spec
        builder = LeaderboardBuilder(spec)

        # Add all entries (including "all" rows) - _handle_aggregates() decides what to keep
        for e in lb.entries:
            # Filter values to only include measures in the spec
            # (handles sparse entries and "all"-only measures when drop_aggregate=True)
            filtered_values = {k: v for k, v in e.values.items() if k in builder.spec.name_set}
            if filtered_values:  # Skip entries with no matching measures
                builder.add(run_id=e.run_id, topic_id=e.topic_id, values=filtered_values)

        return builder.build(on_missing=on_missing, drop_aggregate=drop_aggregate)

    def verify(self,  on_missing:OnMissing, expected_topic_ids: Sequence[str], warn:Optional[bool]=False):
        LeaderboardVerification(leaderboard = self, warn=warn, expected_topic_ids=expected_topic_ids, on_missing=on_missing) \
            .complete_measures(include_all_row=True) \
            .complete_topics()

    # =========================================================================
    # Filtering and Transformation Methods
    # =========================================================================

    @property
    def run_ids(self) -> Set[str]:
        """Unique run_ids in entries."""
        return {e.run_id for e in self.entries}

    @property
    def topic_ids(self) -> Set[str]:
        """Unique topic_ids (excluding 'all')."""
        return {e.topic_id for e in self.entries if e.topic_id != self.all_topic_id}

    def spec_without_aggregate(self) -> "LeaderboardSpec":
        """
        Derive a LeaderboardSpec containing only measures present in per-topic entries.

        Measures that only exist in 'all' rows are excluded. This is useful when
        merging leaderboards or when drop_aggregate will be applied later.
        """
        per_topic_measures: Set[str] = set()
        for e in self.entries:
            if e.topic_id != self.all_topic_id:
                per_topic_measures.update(e.values.keys())

        filtered_specs = tuple(
            ms for ms in self.spec.measures if ms.name in per_topic_measures
        )
        return LeaderboardSpec(measures=filtered_specs)

    def filter_and_recompute(
        self,
        run_ids: Optional[Set[str]] = None,
        topic_ids: Optional[Set[str]] = None,
    ) -> "Leaderboard":
        """
        Filter leaderboard and recompute 'all' aggregates.

        Args:
            run_ids: Keep only these run_ids. None = keep all.
            topic_ids: Keep only these topic_ids. None = keep all.
                Also used as expected topics for aggregation defaults.

        Returns:
            New Leaderboard with filtered entries and fresh 'all' rows.
        """
        builder = LeaderboardBuilder(self.spec)

        for e in self.entries:
            if e.topic_id == self.all_topic_id:
                continue
            if run_ids is not None and e.run_id not in run_ids:
                continue
            if topic_ids is not None and e.topic_id not in topic_ids:
                continue
            builder.add(run_id=e.run_id, topic_id=e.topic_id, values=e.values)

        expected = list(topic_ids) if topic_ids is not None else list(self.topic_ids)
        return builder.build(expected_topic_ids=expected, on_missing="fix_aggregate")

    def get_aggregate_ranking(self, measure: str) -> Dict[str, float]:
        """
        Extract run_id -> value mapping from 'all' aggregate rows.

        Args:
            measure: Measure name to extract

        Returns:
            Dict mapping run_id to measure value

        Raises:
            ValueError: If measure not found in aggregate rows
        """
        ranking: Dict[str, float] = {}
        for e in self.entries:
            if e.topic_id == self.all_topic_id and measure in e.values:
                ranking[e.run_id] = float(e.values[measure])

        if not ranking:
            raise ValueError(f"Measure '{measure}' not found in aggregate rows")

        return ranking

    def top_k_run_ids(self, measure: str, k: int) -> List[str]:
        """
        Get top k run_ids by measure value from 'all' aggregate rows.

        Args:
            measure: Measure name to sort by
            k: Number of top run_ids to return

        Returns:
            List of top k run_ids sorted by measure value (descending)

        Raises:
            ValueError: If measure not found in aggregate rows
        """
        ranking = self.get_aggregate_ranking(measure)
        sorted_runs = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
        return [run_id for run_id, _ in sorted_runs[:k]]


@dataclass(frozen=True)
class MeasureSpec:
    """
    Build-time definition of a measure.

    - `name`: key used in entry.values and output.
    - `dtype`: Only `float` or `str` allowed.

    Processing is derived from dtype:
    - float: cast to float, aggregate via mean, default 0.0
    - str: keep as string, aggregate via first value, default ""

    Note: `bool` and `int` are NOT allowed because they don't survive
    save/load round-trips. Use `float` with 1.0/0.0 for boolean data.
    """
    name: MeasureName
    dtype: type = float

    def __post_init__(self):
        if self.dtype not in (float, str):
            raise ValueError(
                f"MeasureSpec dtype must be float or str, got {self.dtype.__name__}. "
                f"Use float with 1.0/0.0 for boolean data, float for integer counts."
            )

    def get_cast(self) -> CastFn:
        """Return cast function based on dtype."""
        if self.dtype == str:
            return str
        return float

    def get_aggregate(self) -> AggFn:
        """Return aggregate function based on dtype."""
        if self.dtype == str:
            return _first_value
        return _mean_of_floats

    def get_default(self) -> Any:
        """Return default value based on dtype."""
        if self.dtype == str:
            return ""
        return 0.0


@dataclass(frozen=True)
class LeaderboardSpec:
    """
    Build-time schema for a leaderboard.

    The spec defines all valid measure names with aggregator and caster. 
    Storing values for different names will raise an error.
    """
    measures: Tuple[MeasureSpec, ...]
    all_topic_id: str = "all"

    @property
    def names(self) -> Tuple[MeasureName, ...]:
        """Measure names in schema order."""
        return tuple(m.name for m in self.measures)

    @property
    def name_set(self) -> set[MeasureName]:
        """Measure names as a set for fast validation."""
        return set(self.names)

    def cast_values(self, values: Mapping[MeasureName, Any]) -> Dict[MeasureName, Any]:
        """
        Cast/normalize measure values using each MeasureSpec's dtype-derived cast.

        Assumes `values` contains all required measure keys.
        """
        return {m.name: m.get_cast()(values[m.name]) for m in self.measures}


#  ==== Convenient Builder for Leaderboards ===

class LeaderboardBuilder:
    """
    Builder/assembler for Leaderboard.

    Responsibilities:
    - Collect per-topic rows (hand-filled or record-derived).
    - Validate measure keys (fail fast on typos/missing keys).
    - Cast values according to the spec.
    - Handle aggregate rows (drop and recompute, or keep existing).
    """

    def __init__(self, spec: LeaderboardSpec):
        """Create a builder for a specific leaderboard specification."""
        self.spec = spec
        self._rows: List[LeaderboardEntry] = []

    def add(
        self,
        *,
        run_id: str,
        topic_id: str,
        values: Optional[Dict[MeasureName, Any]] = None,
        **kw: Any,
    ) -> None:
        """
        Add one per-topic row.

        Provide either:
        - `values={...}` (a dict of measure -> value), OR
        - keyword args (e.g., GRADE=..., IS_MATCH=...).

        This method is strict by default:
        - Unknown measure keys raise KeyError.
        - Missing measure keys raise KeyError.
        """
        if values is None:
            values = kw
        elif kw:
            raise TypeError("Pass either values= or keyword measures, not both.")

        extra = set(values) - self.spec.name_set
        missing = self.spec.name_set - set(values)
        if extra:
            raise KeyError(f"Unknown measure(s): {sorted(extra)}")
        if missing:
            raise KeyError(f"Missing measure(s): {sorted(missing)}")

        casted = self.spec.cast_values(values)
        self._rows.append(LeaderboardEntry(run_id=run_id, topic_id=topic_id, values=casted))

    def add_records(
        self,
        records: Iterable[Any],
        *,
        run_id: Callable[[Any], str],
        topic_id: Callable[[Any], str],
        get_values: Callable[[Any], Dict[MeasureName, Any]],
    ) -> None:
        """
        Add multiple rows from an iterable of arbitrary record objects.

        The caller supplies functions to extract:
        - `run_id(record)`
        - `topic_id(record)`
        - `get_values(record)` -> {measure_name: value, ...}
        """
        for r in records:
            self.add(run_id=run_id(r), topic_id=topic_id(r), values=get_values(r))

    def add_entries(
        self,
        entries: Iterable[LeaderboardEntry],
        skip_all_rows: bool = True,
    ) -> "LeaderboardBuilder":
        """
        Add entries from LeaderboardEntry objects (e.g., from another leaderboard).

        Values are filtered to only include measures in the spec. Validation
        of complete measure coverage happens at build() time.

        Args:
            entries: LeaderboardEntry objects to add
            skip_all_rows: If True (default), skip entries where topic_id == all_topic_id

        Returns:
            self for chaining
        """
        for e in entries:
            if skip_all_rows and e.topic_id == self.spec.all_topic_id:
                continue
            # Filter values to spec - validation happens at build() time
            filtered_values = {k: v for k, v in e.values.items() if k in self.spec.name_set}
            if filtered_values:
                self.add(run_id=e.run_id, topic_id=e.topic_id, values=filtered_values)
        return self

    def add_from_all(
        self,
        leaderboards: Sequence[Leaderboard],
        skip_all_rows: bool = True,
    ) -> "LeaderboardBuilder":
        """
        Add entries from multiple leaderboards.

        Args:
            leaderboards: Leaderboards to add entries from
            skip_all_rows: If True (default), skip 'all' rows (they'll be recomputed)

        Returns:
            self for chaining
        """
        for lb in leaderboards:
            self.add_entries(lb.entries, skip_all_rows=skip_all_rows)
        return self

    def entries(self) -> tuple[LeaderboardEntry, ...]:
        """Return the currently staged per-topic entries (no synthetic 'all' rows)."""
        return tuple(self._rows)


    def _detect_missing_run_topic(
        self,
        expected_topic_ids: Sequence[str],
    ) -> List[tuple[str, str]]:
        """
        Detect missing (run_id, topic_id) pairs.

        Returns list of (run_id, topic_id) tuples for each run that is missing
        expected topics.
        """
        existing_run_topic: Dict[str, Set[str]] = defaultdict(set)
        for e in self._rows:
            if e.topic_id != self.spec.all_topic_id:
                existing_run_topic[e.run_id].add(e.topic_id)

        expected_set = set(expected_topic_ids)
        missing: List[tuple[str, str]] = []
        for run_id in existing_run_topic.keys():
            for topic_id in expected_set - existing_run_topic[run_id]:
                missing.append((run_id, topic_id))
        return missing

    def _compute_aggregates(
        self,
        per_topic_entries: List[LeaderboardEntry],
        phantom_defaults: List[tuple[str, str]],
    ) -> List[LeaderboardEntry]:
        """
        Compute "all" row aggregates from per-topic entries and phantom defaults.

        Args:
            per_topic_entries: Per-topic entries to aggregate (must not contain "all" entries)
            phantom_defaults: (run_id, topic_id) pairs to include in aggregation
                using MeasureSpec default values (no actual entries created)

        Returns:
            List of aggregate "all" row entries, one per run_id
        """
        by_run: Dict[str, Dict[MeasureName, List[Any]]] = defaultdict(lambda: defaultdict(list))

        # Collect values from actual entries
        for e in per_topic_entries:
            for k, v in e.values.items():
                by_run[e.run_id][k].append(v)

        # Include phantom defaults
        for run_id, _ in phantom_defaults:
            for ms in self.spec.measures:
                by_run[run_id][ms.name].append(ms.get_default())

        # Build aggregate rows
        all_rows: List[LeaderboardEntry] = []
        for run_id, m2vals in by_run.items():
            agg_vals: Dict[MeasureName, Any] = {}
            for ms in self.spec.measures:
                vals = m2vals.get(ms.name, [])
                if vals:
                    agg_vals[ms.name] = ms.get_aggregate()(vals)
            all_rows.append(LeaderboardEntry(run_id=run_id, topic_id=self.spec.all_topic_id, values=agg_vals))

        return all_rows

    def _handle_aggregates(
        self,
        entries: List[LeaderboardEntry],
        drop_aggregate: bool,
        phantom_defaults: List[tuple[str, str]],
    ) -> Tuple[List[LeaderboardEntry], List[LeaderboardEntry]]:
        """
        Handle aggregate ("all") rows based on drop_aggregate flag.

        This is the single place where drop_aggregate logic is handled.

        Args:
            entries: All entries (may include existing "all" rows)
            drop_aggregate: If True, drop existing aggregates and recompute.
                           If False, keep existing aggregates as-is.
            phantom_defaults: (run_id, topic_id) pairs for phantom default handling

        Returns:
            Tuple of (per_topic_entries, all_entries)
        """
        # Separate per-topic from "all" entries
        per_topic_entries = [e for e in entries if e.topic_id != self.spec.all_topic_id]
        existing_all_entries = [e for e in entries if e.topic_id == self.spec.all_topic_id]

        if drop_aggregate:
            # Drop existing aggregates and compute fresh ones
            all_entries = self._compute_aggregates(per_topic_entries, phantom_defaults)
        else:
            # Keep existing aggregates (already cast via add())
            all_entries = existing_all_entries

        return per_topic_entries, all_entries

    def build(
        self,
        expected_topic_ids: Optional[Sequence[str]] = None,
        on_missing: OnMissing = "default",
        drop_aggregate: bool = True,
    ) -> Leaderboard:
        """
        Build a Leaderboard with per-run `all_topic_id` rows.

        The returned Leaderboard contains:
        - all per-topic rows that were added
        - plus one row per run_id with topic_id == spec.all_topic_id (aggregate)

        Args:
            expected_topic_ids: If provided, handles missing (run_id, topic_id) pairs.
            on_missing: When expected_topic_ids is provided and gaps exist:
                - "default": silently create per-topic entries with defaults
                - "warn": create per-topic entries with defaults and print warning
                - "fix_aggregate": only fill defaults for "all" row aggregation (no per-topic entries)
                - "error": raise ValueError listing missing (run_id, topic_id) pairs
            drop_aggregate: If True (default), drop any existing "all" entries and
                recompute fresh aggregates. If False, keep existing "all" entries as-is.
        """
        # Step 1: Detect missing pairs
        all_missing: List[tuple[str, str]] = []
        if expected_topic_ids is not None:
            all_missing = self._detect_missing_run_topic(expected_topic_ids)

        # Step 2: Handle missing based on mode
        filled_rows: List[LeaderboardEntry] = []
        phantom_defaults: List[tuple[str, str]] = []

        if all_missing:
            formatted_pairs = [f"({r}, {t})" for r, t in sorted(all_missing)]
            if on_missing == "error":
                raise ValueError(
                    f"Missing leaderboard entries for {len(all_missing)} (run_id, topic_id) pair(s): {format_preview(formatted_pairs)}"
                )

            if on_missing == "warn":
                print(f"Leaderboard Warning: {len(all_missing)} missing entries: {format_preview(formatted_pairs)}", file=sys.stderr)

            if on_missing in ("default", "warn"):
                # Create actual per-topic entries
                default_values = {ms.name: ms.get_default() for ms in self.spec.measures}
                for run_id, topic_id in all_missing:
                    filled_rows.append(LeaderboardEntry(run_id=run_id, topic_id=topic_id, values=default_values))
            elif on_missing == "fix_aggregate":
                phantom_defaults = all_missing

        # Step 3: Handle aggregates via single handler
        all_entries = self._rows + filled_rows
        per_topic_entries, all_rows = self._handle_aggregates(all_entries, drop_aggregate, phantom_defaults)

        return Leaderboard(
            measures=self.spec.names,
            entries=tuple(per_topic_entries + all_rows),
            all_topic_id=self.spec.all_topic_id,
            spec=self.spec,
        )


#  === Aggregator functions (module-private) ====

def _mean_of_floats(values: Sequence[Any]) -> float:
    """Aggregate numeric values via arithmetic mean."""
    return mean(float(v) for v in values)


def _first_value(values: Sequence[Any]) -> Any:
    """Aggregate by taking the first value (used for string measures)."""
    return values[0] if values else None


def _infer_dtype_from_values(values: Sequence[Any]) -> type:
    """
    Infer dtype from a sequence of string values.

    Returns:
        - float: if all values can be parsed as numbers
        - str: otherwise

    Note: Only float and str are valid dtypes. Bool/int are not inferred
    because they can't survive save/load round-trips.
    """
    non_none_values = [v for v in values if v is not None]
    if not non_none_values:
        return float  # Default to float for empty

    # Check if all values can be parsed as numbers
    try:
        for v in non_none_values:
            float(v)
        return float
    except (ValueError, TypeError):
        pass

    return str


