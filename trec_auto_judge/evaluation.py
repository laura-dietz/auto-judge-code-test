import pandas as pd
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Literal, Optional

from trec_auto_judge import Leaderboard

OnMissing = Literal["error", "warn", "skip", "default"]
LeaderboardFormat = Literal["trec_eval", "tot", "ir_measures", "ranking"]


class TrecLeaderboardEvaluation():
    def __init__(
        self,
        truth_leaderboard: Optional[Path],
        truth_measure: Optional[str],
        eval_measure: Optional[str],
        on_missing: OnMissing = "error",
        truth_format: LeaderboardFormat = "ir_measures",
        truth_has_header: bool = False,
        eval_format: LeaderboardFormat = "tot",
        eval_has_header: bool = False,
    ):
        self.on_missing = on_missing
        self.truth_format = truth_format
        self.truth_has_header = truth_has_header
        self.eval_format = eval_format
        self.eval_has_header = eval_has_header

        # if only one measure is provided, assume both are the same.
        if not eval_measure:
            eval_measure = truth_measure
        if not truth_measure:
            truth_measure = eval_measure

        if truth_leaderboard and truth_measure:
            parsed_leaderboard = self.load_leaderboard(truth_leaderboard, self.truth_format, self.truth_has_header)
            self.ground_truth_ranking = self.extract_ranking(parsed_leaderboard, truth_measure)
        else:
            self.ground_truth_ranking = None

    def load_leaderboard(self, leaderboard_path: Path, format: LeaderboardFormat, has_header: bool = False) -> Leaderboard:
        if not leaderboard_path or not Path(leaderboard_path).is_file():
            raise ValueError(f"I expected that {leaderboard_path} is a file.")
        return Leaderboard.load(Path(leaderboard_path), format=format, has_header=has_header)

    def extract_ranking(self, leaderboard: Leaderboard, measure: str) -> Dict[str, float]:
        """Extract run_id -> value mapping for aggregate rows (topic_id == all_topic_id)."""
        ret = {}
        for entry in leaderboard.entries:
            if entry.topic_id != leaderboard.all_topic_id:
                continue
            if measure in entry.values:
                ret[entry.run_id] = float(entry.values[measure])

        if len(ret) == 0:
            raise ValueError(f"Measure {measure} does not exist, I found: {sorted(leaderboard.measures)}")
        return ret

    def evaluate(self, leaderboard_file: Path) -> Dict[str, Dict]:
        leaderboard = self.load_leaderboard(leaderboard_file, self.eval_format, self.eval_has_header)
        ret = {}

        for m in leaderboard.measures:
            if self.ground_truth_ranking:
                ret[m] = self.correlation_to_truth(self.extract_ranking(leaderboard, m))
            else:
                ret[m] = self.basic_statistics(leaderboard, m)

        return ret

    def basic_statistics(self, leaderboard: Leaderboard, measure: str) -> Dict[str, float]:
        """Compute mean/stdev across all entries (including per-topic rows)."""
        vals = []
        for entry in leaderboard.entries:
            if measure in entry.values:
                vals.append(float(entry.values[measure]))

        if len(vals) == 0:
            raise ValueError(f"Measure {measure} does not exist.")

        return {"mean-value": mean(vals), "stdev-value": stdev(vals)}

    def correlation_to_truth(self, ranking: Dict[str, float]) -> Dict[str, float]:
        a, b = [], []

        truth_systems = set(self.ground_truth_ranking.keys())
        eval_systems = set(ranking.keys())

        missing_in_eval = truth_systems - eval_systems
        missing_in_truth = eval_systems - truth_systems

        if missing_in_eval or missing_in_truth:
            msg_parts = []
            if missing_in_eval:
                msg_parts.append(f"missing in evaluated: {sorted(missing_in_eval)}")
            if missing_in_truth:
                msg_parts.append(f"missing in ground truth: {sorted(missing_in_truth)}")
            msg = f"Run ID mismatch: {'; '.join(msg_parts)}. \nShared RunIDs: {sorted(truth_systems &
            eval_systems)}"
            

            if self.on_missing == "error":
                raise ValueError(msg)
            elif self.on_missing == "warn":
                print(f"Warning: {msg}", file=sys.stderr)

        # Include all systems from ground truth
        for system in truth_systems:
            a.append(float(self.ground_truth_ranking[system]))
            if system in ranking:
                b.append(float(ranking[system]))
            elif self.on_missing == "default":
                b.append(0.0)
            else:
                # skip: only include common systems
                a.pop()  # remove the truth value we just added

        return {
            "kendall": correlation(a, b, "kendall"),
            "pearson": correlation(a, b, "pearson"),
            "spearman": correlation(a, b, "spearman"),
            "tauap_b": tauap_b(a, b)
        }


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
