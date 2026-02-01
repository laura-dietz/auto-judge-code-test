import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from trec_auto_judge import Leaderboard

# TODO: Consider unifying with leaderboard.OnMissing which uses "fix_aggregate" instead of "skip"
OnMissing = Literal["error", "warn", "skip", "default"]
LeaderboardFormat = Literal["trec_eval", "tot", "ir_measures", "ranking"]
BASE_CORRELATION_METHODS = ["kendall", "pearson", "spearman", "tauap_b"]
TOP_K_VALUES = [10]
CORRELATION_METHODS: List[str] = (
    BASE_CORRELATION_METHODS +
    [f"{m}@{k}" for m in BASE_CORRELATION_METHODS for k in TOP_K_VALUES]
)


def parse_correlation_method(method: str) -> tuple[str, int | None]:
    """Parse correlation method string into (base_method, top_k).

    Examples:
        "kendall" -> ("kendall", None)
        "kendall@10" -> ("kendall", 10)
    """
    if "@" in method:
        base, k_str = method.split("@", 1)
        return (base, int(k_str))
    return (method, None)


class LeaderboardEvaluator():
    """Compute correlation between predicted leaderboards and ground truth."""

    def __init__(
        self,
        truth_leaderboard: Path,
        truth_measures: List[str] | None = None,
        eval_measures: List[str] | None = None,
        on_missing: OnMissing = "error",
        truth_format: LeaderboardFormat = "ir_measures",
        truth_has_header: bool = False,
        truth_drop_aggregate: bool = False,
        eval_format: LeaderboardFormat = "tot",
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

        # Load truth leaderboard (required)
        self.truth_leaderboard = self.load_leaderboard(
            truth_leaderboard, self.truth_format, self.truth_has_header, self.on_missing, self.truth_drop_aggregate
        )

        # Validate truth leaderboard has enough runs for correlation
        num_runs = len(self.truth_leaderboard.run_ids)
        if num_runs < 3:
            raise ValueError(
                f"Truth leaderboard has only {num_runs} run(s). "
                f"Correlation requires at least 3 runs. "
                f"Did you pass a single file instead of a directory?"
            )

    def load_leaderboard(self, leaderboard_path: Path, format: LeaderboardFormat, has_header: bool = False, on_missing: OnMissing = "error", drop_aggregate: bool = False) -> Leaderboard:
        if not leaderboard_path or not Path(leaderboard_path).exists():
            raise ValueError(f"Leaderboard path does not exist: {leaderboard_path}")
        return Leaderboard.load(Path(leaderboard_path), format=format, has_header=has_header, on_missing=on_missing, drop_aggregate=drop_aggregate)

    def extract_ranking(self, leaderboard: Leaderboard, measure: str) -> Dict[str, float]:
        """Extract run_id -> value mapping for aggregate rows (topic_id == all_topic_id)."""
        return leaderboard.get_aggregate_ranking(measure)

    def get_measure_pairs(
        self, eval_leaderboard: Leaderboard
    ) -> List[Tuple[str, str]]:
        """Get list of (truth_measure, eval_measure) pairs to analyze."""
        # Determine eval measures to use
        eval_measures = list(eval_leaderboard.measures)
        if self.eval_measures_filter:
            missing = set(self.eval_measures_filter) - set(eval_leaderboard.measures)
            if missing:
                raise ValueError(f"Eval measures not found in eval leaderboard: {sorted(missing)}")
            eval_measures = [m for m in eval_measures if m in self.eval_measures_filter]

        # Determine truth measures to use
        truth_measures = list(self.truth_leaderboard.measures)
        if self.truth_measures_filter:
            missing = set(self.truth_measures_filter) - set(self.truth_leaderboard.measures)
            if missing:
                raise ValueError(f"Truth measures not found in truth leaderboard: {sorted(missing)}")
            truth_measures = [m for m in truth_measures if m in self.truth_measures_filter]

        # Generate all pairs
        return [(tm, em) for tm in truth_measures for em in eval_measures]

    def evaluate(
        self,
        leaderboard_file: Path,
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Evaluate and return dict keyed by (truth_measure, eval_measure) pairs.

        Args:
            leaderboard_file: Path to the evaluated leaderboard file

        Returns:
            Dict mapping (truth_measure, eval_measure) pairs to correlation results.
            Each method (e.g., "kendall", "kendall@10") is computed with its own
            top-k filtering based on the @k suffix.

        Filtering order for kendall@k:
            1. Filter truth to pre-determined topic_ids, recompute aggregates
            2. Get top k runs from truth's new aggregates
            3. Filter eval to topic_ids AND top k runs, recompute aggregates
            4. Filter truth to top k runs, recompute aggregates
            5. Compare rankings
        """
        eval_leaderboard = self.load_leaderboard(
            leaderboard_file, self.eval_format, self.eval_has_header, self.on_missing, self.eval_drop_aggregate
        )

        # Step 1: Determine topics to use
        # Use pre-determined topic_ids, or fall back to truth's topics
        topics_to_use = self.topic_ids if self.topic_ids is not None else self.truth_leaderboard.topic_ids

        # Step 2: Filter truth to topics, recompute aggregates
        truth_by_topics = self.truth_leaderboard.filter_and_recompute(topic_ids=topics_to_use)

        ret = {}

        for truth_m, eval_m in self.get_measure_pairs(eval_leaderboard):
            correlations = {}

            for method in self.correlation_methods:
                base_method, top_k = parse_correlation_method(method)

                if top_k is not None:
                    # Step 3: Get top k runs from truth (based on topic-filtered aggregates)
                    top_run_ids = set(truth_by_topics.top_k_run_ids(truth_m, top_k))

                    # Step 4: Filter eval to topics AND top k runs, recompute
                    eval_for_method = eval_leaderboard.filter_and_recompute(
                        topic_ids=topics_to_use,
                        run_ids=top_run_ids
                    )

                    # Step 5: Filter truth to top k runs (already has topics), recompute
                    truth_for_method = truth_by_topics.filter_and_recompute(run_ids=top_run_ids)
                else:
                    # No top-k: just filter both to topics
                    truth_for_method = truth_by_topics
                    eval_for_method = eval_leaderboard.filter_and_recompute(topic_ids=topics_to_use)

                truth_ranking = self.extract_ranking(truth_for_method, truth_m)
                eval_ranking = self.extract_ranking(eval_for_method, eval_m)

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

            if self.on_missing == "error":
                raise ValueError(msg)
            elif self.on_missing == "warn":
                print(f"Warning: {msg}", file=sys.stderr)

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
