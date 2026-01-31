import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from trec_auto_judge import Leaderboard

OnMissing = Literal["error", "warn", "skip", "default"]
LeaderboardFormat = Literal["trec_eval", "tot", "ir_measures", "ranking"]
CorrelationMethod = Literal["kendall", "pearson", "spearman", "tauap_b"]
CORRELATION_METHODS: List[CorrelationMethod] = ["kendall", "pearson", "spearman", "tauap_b"]


class TrecLeaderboardEvaluation():
    """Compute correlation between predicted leaderboards and ground truth."""

    def __init__(
        self,
        truth_leaderboard: Path,
        truth_measures: List[str] | None = None,
        eval_measures: List[str] | None = None,
        on_missing: OnMissing = "error",
        truth_format: LeaderboardFormat = "ir_measures",
        truth_has_header: bool = False,
        eval_format: LeaderboardFormat = "tot",
        eval_has_header: bool = False,
        correlation_methods: List[CorrelationMethod] | None = None,
    ):
        self.on_missing = on_missing
        self.truth_format = truth_format
        self.truth_has_header = truth_has_header
        self.eval_format = eval_format
        self.eval_has_header = eval_has_header
        self.truth_measures_filter = truth_measures  # None means all
        self.eval_measures_filter = eval_measures    # None means all
        self.correlation_methods = correlation_methods if correlation_methods else CORRELATION_METHODS

        # Load truth leaderboard (required)
        self.truth_leaderboard = self.load_leaderboard(
            truth_leaderboard, self.truth_format, self.truth_has_header
        )

    def load_leaderboard(self, leaderboard_path: Path, format: LeaderboardFormat, has_header: bool = False) -> Leaderboard:
        if not leaderboard_path or not Path(leaderboard_path).exists():
            raise ValueError(f"Leaderboard path does not exist: {leaderboard_path}")
        return Leaderboard.load(Path(leaderboard_path), format=format, has_header=has_header)

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
        top_k: int | None = None,
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Evaluate and return dict keyed by (truth_measure, eval_measure) pairs.

        Args:
            leaderboard_file: Path to the evaluated leaderboard file
            top_k: If set, compute correlation only for top k run_ids from truth.
                The truth leaderboard is first cleaned to match eval's run_ids,
                then top k are selected based on each truth measure.

        Returns:
            Dict mapping (truth_measure, eval_measure) pairs to correlation results.
        """
        eval_leaderboard = self.load_leaderboard(
            leaderboard_file, self.eval_format, self.eval_has_header
        )

        # Clean truth to match eval's run_ids and recompute aggregates
        cleaned_truth = self.truth_leaderboard.filter_and_recompute(
            run_ids=eval_leaderboard.run_ids
        )

        ret = {}

        for truth_m, eval_m in self.get_measure_pairs(eval_leaderboard):
            # Get the appropriate leaderboards for this measure pair
            truth_for_measure = cleaned_truth
            eval_for_measure = eval_leaderboard

            if top_k is not None:
                # Get top k run_ids from cleaned truth
                top_run_ids = set(cleaned_truth.top_k_run_ids(truth_m, top_k))
                # Filter both to top k and recompute
                truth_for_measure = cleaned_truth.filter_and_recompute(run_ids=top_run_ids)
                eval_for_measure = eval_leaderboard.filter_and_recompute(run_ids=top_run_ids)

            truth_ranking = self.extract_ranking(truth_for_measure, truth_m)
            eval_ranking = self.extract_ranking(eval_for_measure, eval_m)
            ret[(truth_m, eval_m)] = self.correlation_to_truth(truth_ranking, eval_ranking)

        return ret

    def correlation_to_truth(
        self, truth_ranking: Dict[str, float], eval_ranking: Dict[str, float]
    ) -> Dict[str, float]:
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

        result = {}
        for method in self.correlation_methods:
            if method == "tauap_b":
                result[method] = tauap_b(a, b)
            else:
                result[method] = correlation(a, b, method)
        return result


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
