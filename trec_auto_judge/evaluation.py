import click
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from trec_auto_judge.eval_results import load as load_eval_result, EvalResult

# TODO: Consider unifying with leaderboard.OnMissing which uses "fix_aggregate" instead of "skip"
OnMissing = Literal["error", "warn", "skip", "default"]
EvalResultFormat = Literal["trec_eval", "tot", "ir_measures", "ranking", "jsonl"]
BASE_CORRELATION_METHODS = ["kendall", "pearson", "spearman", "tauap_b"]
TOP_K_VALUES = [10]
CORRELATION_METHODS: List[str] = (
    BASE_CORRELATION_METHODS + 
    ["kendall@10"]
    # [f"{m}@{k}" for m in BASE_CORRELATION_METHODS for k in TOP_K_VALUES]
)


def parse_correlation_method(method: str) -> tuple[str, int | None]:
    """Parse correlation method string into (base_method, top_k).

    Examples:
        "kendall" -> ("kendall", None)
        "kendall@10" -> ("kendall", 10)

    Raises:
        ValueError: if base method is invalid or k is not a positive integer.
    """
    if "@" in method:
        base, k_str = method.split("@", 1)
        if base not in BASE_CORRELATION_METHODS:
            raise ValueError(f"Unknown base method '{base}'")
        k = int(k_str)  # raises ValueError if not int
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        return (base, k)
    if method not in BASE_CORRELATION_METHODS:
        raise ValueError(f"Unknown method '{method}'")
    return (method, None)


class CorrelationMethodType(click.ParamType):
    """Click parameter type for correlation methods (method or method@k)."""
    name = "correlation"

    def convert(self, value, param, ctx):
        try:
            parse_correlation_method(value)
            return value
        except ValueError as e:
            methods = ", ".join(BASE_CORRELATION_METHODS)
            self.fail(
                f"{e}. Valid: [{methods}] or method@k (e.g., kendall@15).",
                param,
                ctx,
            )


class LeaderboardEvaluator():
    """Compute correlation between predicted leaderboards and ground truth."""

    def __init__(
        self,
        truth_leaderboard: Path,
        truth_measures: List[str] | None = None,
        eval_measures: List[str] | None = None,
        on_missing: OnMissing = "error",
        truth_format: EvalResultFormat = "ir_measures",
        truth_has_header: bool = False,
        truth_drop_aggregate: bool = False,
        eval_format: EvalResultFormat = "tot",
        eval_has_header: bool = False,
        eval_drop_aggregate: bool = False,
        correlation_methods: List[str] | None = None,
        topic_ids: set[str] | None = None,
    ):
        self.on_missing = on_missing
        self.truth_format = truth_format
        self.truth_has_header = truth_has_header
        self.truth_drop_aggregate = truth_drop_aggregate
        self.eval_format = eval_format
        self.eval_has_header = eval_has_header
        self.eval_drop_aggregate = eval_drop_aggregate
        self.truth_measures_filter = truth_measures  # None means all
        self.eval_measures_filter = eval_measures    # None means all
        self.correlation_methods = correlation_methods if correlation_methods else CORRELATION_METHODS
        self.topic_ids = topic_ids  # Pre-determined topics, or None = use truth's topics

        # Load truth result (required)
        self.truth_result = self._load_eval_result(
            truth_leaderboard, self.truth_format, self.truth_has_header, self.on_missing, self.truth_drop_aggregate
        )

        # Validate truth result has enough runs for correlation
        num_runs = len(self.truth_result.run_ids)
        if num_runs < 3:
            raise ValueError(
                f"Truth result has only {num_runs} run(s). "
                f"Correlation requires at least 3 runs. "
                f"Did you pass a single file instead of a directory?"
            )

    def _handle_missing(self, msg: str) -> bool:
        """Handle missing data according to on_missing policy.

        Returns True if processing should continue (warn/skip/default),
        raises ValueError if on_missing == "error".
        """
        if self.on_missing == "error":
            raise ValueError(msg)
        elif self.on_missing == "warn":
            print(f"Warning: {msg}", file=sys.stderr)
        return True  # continue for warn/skip/default

    def _load_eval_result(
        self,
        path: Path,
        format: EvalResultFormat,
        has_header: bool = False,
        on_missing: OnMissing = "error",
        drop_aggregate: bool = False,
    ) -> EvalResult:
        if not path or not Path(path).exists():
            raise ValueError(f"Path does not exist: {path}")

        # Map on_missing: "skip"/"default" -> "ignore"
        eval_on_missing = "ignore" if on_missing in ("skip", "default") else on_missing

        return load_eval_result(
            Path(path),
            format=format,
            has_header=has_header,
            drop_aggregates=drop_aggregate,
            recompute_aggregates=drop_aggregate,
            verify=True,
            on_missing=eval_on_missing,
        )

    def extract_ranking(self, eval_result: EvalResult, measure: str) -> Dict[str, float] | None:
        """Extract run_id -> value mapping for aggregate rows (topic_id == ALL_TOPIC_ID).

        Returns None if measure is missing and on_missing is skip/warn.
        Returns all 0.0 if measure is missing and on_missing is default.
        """
        if measure not in eval_result.measures:
            self._handle_missing(f"Measure '{measure}' not found in result")
            if self.on_missing == "default":
                # Fill all runs with 0.0
                return {run_id: 0.0 for run_id in eval_result.run_ids}
            return None  # skip for warn/skip
        return eval_result.get_aggregate_ranking(measure)

    def get_measure_pairs(
        self, eval_result: EvalResult
    ) -> List[Tuple[str, str]]:
        """Get list of (truth_measure, eval_measure) pairs to analyze."""
        # Determine eval measures to use
        eval_measures = sorted(eval_result.measures)
        if self.eval_measures_filter:
            missing = set(self.eval_measures_filter) - eval_result.measures
            if missing:
                raise ValueError(f"Eval measures not found in eval result: {sorted(missing)}")
            eval_measures = [m for m in eval_measures if m in self.eval_measures_filter]

        # Determine truth measures to use
        truth_measures = sorted(self.truth_result.measures)
        if self.truth_measures_filter:
            missing = set(self.truth_measures_filter) - self.truth_result.measures
            if missing:
                raise ValueError(f"Truth measures not found in truth result: {sorted(missing)}")
            truth_measures = [m for m in truth_measures if m in self.truth_measures_filter]

        # Generate all pairs
        return [(tm, em) for tm in truth_measures for em in eval_measures]

    def evaluate(
        self,
        eval_file: Path,
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Evaluate and return dict keyed by (truth_measure, eval_measure) pairs.

        Args:
            eval_file: Path to the evaluated result file

        Returns:
            Dict mapping (truth_measure, eval_measure) pairs to correlation results.
            Each method (e.g., "kendall", "kendall@10") is computed with its own
            top-k filtering based on the @k suffix.

        Filtering order:
            1. Filter truth and eval to common runs (intersection for fair comparison)
            2. If topic_ids specified (--only-shared-topics): filter to those topics, recompute
               Otherwise (--all-topics): use provided aggregates, no topic filtering
            For kendall@k specifically:
            3. Get top k runs from truth's aggregates
            4. Filter eval to top k runs (and topics if specified), recompute
            5. Filter truth to top k runs, recompute
            6. Compare rankings
        """
        eval_result = self._load_eval_result(
            eval_file, self.eval_format, self.eval_has_header, self.on_missing, self.eval_drop_aggregate
        )

        # Step 1: Compute common runs (intersection of truth and eval)
        common_run_ids = set(self.truth_result.run_ids) & set(eval_result.run_ids)

        # Step 2: Determine if we need to filter/recompute truth
        # Only @k methods require recomputation (they filter by runs)
        # Topic filtering alone preserves aggregates (aggregate = mean over topics)
        has_top_k_methods = any("@" in m for m in self.correlation_methods)
        needs_recompute = has_top_k_methods

        if needs_recompute:
            # Filter and recompute aggregates (required for @k methods or topic filtering)
            truth_filtered = self.truth_result.filter_and_recompute(
                topic_ids=self.topic_ids,
                run_ids=common_run_ids,
            )
            # Warn about measures lost during filtering
            lost_measures = self.truth_result.measures - truth_filtered.measures
            if lost_measures:
                print(
                    f"Warning: {len(lost_measures)} measure(s) unavailable after filtering "
                    f"(no per-topic data): {sorted(lost_measures)}",
                    file=sys.stderr
                )
        else:
            # No filtering needed - use original truth with provided aggregates
            # Run alignment will handle mismatches via on_missing policy
            truth_filtered = self.truth_result

        ret = {}

        for truth_m, eval_m in self.get_measure_pairs(eval_result):
            correlations = {}

            for method in self.correlation_methods:
                base_method, top_k = parse_correlation_method(method)

                if top_k is not None:
                    # Step 4: Get top k runs from truth (already filtered to common runs + topics)
                    # Check if measure exists before computing top-k
                    if truth_m not in truth_filtered.measures:
                        self._handle_missing(f"Measure '{truth_m}' not found for top-k ranking")
                        continue  # skip this method for this measure pair

                    top_run_ids = set(truth_filtered.top_k_run_ids(truth_m, top_k))

                    # Step 5: Filter eval to topics (if specified) AND top k runs, recompute
                    eval_for_method = eval_result.filter_and_recompute(
                        topic_ids=self.topic_ids,
                        run_ids=top_run_ids
                    )

                    # Step 6: Filter truth to top k runs, recompute
                    truth_for_method = truth_filtered.filter_and_recompute(run_ids=top_run_ids)
                else:
                    # No top-k: filter eval to common runs (and topics if specified)
                    truth_for_method = truth_filtered
                    eval_for_method = eval_result.filter_and_recompute(
                        topic_ids=self.topic_ids,
                        run_ids=common_run_ids,
                    )

                truth_ranking = self.extract_ranking(truth_for_method, truth_m)
                eval_ranking = self.extract_ranking(eval_for_method, eval_m)

                # Skip if either measure is missing
                if truth_ranking is None or eval_ranking is None:
                    continue

                correlations[method] = self._compute_single_correlation(
                    truth_ranking, eval_ranking, base_method
                )

            ret[(truth_m, eval_m)] = correlations

        return ret

    def _align_rankings(
        self, truth_ranking: Dict[str, float], eval_ranking: Dict[str, float]
    ) -> Tuple[List[float], List[float]]:
        """Align truth and eval rankings, handling missing systems per on_missing policy."""
        a, b = [], []

        truth_systems = set(truth_ranking.keys())
        eval_systems = set(eval_ranking.keys())

        missing_in_eval = truth_systems - eval_systems
        missing_in_truth = eval_systems - truth_systems

        if missing_in_eval or missing_in_truth:
            msg_parts = []
            if missing_in_eval:
                msg_parts.append(f"missing in evaluated: {sorted(missing_in_eval)}")
            if missing_in_truth:
                msg_parts.append(f"missing in ground truth: {sorted(missing_in_truth)}")
            msg = f"Run ID mismatch: {'; '.join(msg_parts)}. \nShared RunIDs: {sorted(truth_systems & eval_systems)}"

            self._handle_missing(msg)

        # Include all systems from ground truth
        for system in truth_systems:
            a.append(float(truth_ranking[system]))
            if system in eval_ranking:
                b.append(float(eval_ranking[system]))
            elif self.on_missing == "default":
                b.append(0.0)
            else:
                # skip: only include common systems
                a.pop()  # remove the truth value we just added

        return a, b

    def _compute_single_correlation(
        self, truth_ranking: Dict[str, float], eval_ranking: Dict[str, float], base_method: str
    ) -> float:
        """Compute a single correlation between aligned rankings."""
        a, b = self._align_rankings(truth_ranking, eval_ranking)

        if base_method == "tauap_b":
            return tauap_b(a, b)
        else:
            return correlation(a, b, base_method)


def _check_input_or_raise(a, b):
    if len(a) < 3:
        raise ValueError(f"Can not calculate correlations on only {len(a)} elements.")
    if len(a) != len(b):
        raise ValueError(f"Can not calculate correlations on unequal elements: {len(a)} != {len(b)}")

def correlation(a, b, method):
    _check_input_or_raise(a, b)

    df = pd.DataFrame([{"a": i, "b": j} for i, j in zip(a, b)])

    return float(df.corr(method).iloc[0]["b"])

def tauap_b(a, b):
    from .pyircore import tauap_b as method

    _check_input_or_raise(a, b)
    return method(a, b)
