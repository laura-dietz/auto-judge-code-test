"""
Test to investigate why v1 and v2 meta-evaluate produce different correlation numbers.

Hypothesis: v2 always recomputes eval aggregates via filter_and_recompute(),
even when no filtering is needed. If the file's pre-computed aggregates differ
from mean-of-per-topic values, this causes correlation differences.

Run with: pytest tests/test_v1_v2_aggregate_difference.py -v -s
"""

import pytest
from pathlib import Path


# Update these paths to your actual test files
TRUTH_JSONL = Path("TRUTH")  # Update this
EVAL_TOT = Path("EVAL")  # Update this
MEASURE = "nugget_coverage_macro"


@pytest.fixture
def truth_path():
    """Path to truth leaderboard file."""
    if not TRUTH_JSONL.exists():
        pytest.skip(f"Truth file not found: {TRUTH_JSONL}")
    return TRUTH_JSONL


@pytest.fixture
def eval_path():
    """Path to eval leaderboard file."""
    if not EVAL_TOT.exists():
        pytest.skip(f"Eval file not found: {EVAL_TOT}")
    return EVAL_TOT


class TestAggregatePreservation:
    """Test whether aggregates are preserved or recomputed."""

    def test_eval_result_original_vs_recomputed(self, truth_path):
        """Compare original aggregates from file vs recomputed aggregates."""
        from trec_auto_judge.eval_results import load as load_eval_result

        # Load with original aggregates preserved
        original = load_eval_result(
            truth_path,
            format="jsonl",
            has_header=False,
            drop_aggregates=False,
            recompute_aggregates=False,
            verify=False,
            on_missing="ignore",
        )

        print(f"\nOriginal: {len(original.run_ids)} runs, {len(original.topic_ids)} topics")
        print(f"Original measures: {sorted(original.measures)[:5]}...")

        # Recompute aggregates (simulating what v2 does)
        recomputed = original.filter_and_recompute(
            run_ids=set(original.run_ids),  # same runs
            topic_ids=None,  # no topic filtering
        )

        print(f"Recomputed: {len(recomputed.run_ids)} runs, {len(recomputed.topic_ids)} topics")
        print(f"Recomputed measures: {sorted(recomputed.measures)[:5]}...")

        # Compare aggregates for the test measure
        if MEASURE not in original.measures:
            pytest.skip(f"Measure {MEASURE} not in original")
        if MEASURE not in recomputed.measures:
            pytest.skip(f"Measure {MEASURE} not in recomputed (lost during recompute?)")

        orig_ranking = original.get_aggregate_ranking(MEASURE)
        recomp_ranking = recomputed.get_aggregate_ranking(MEASURE)

        print(f"\nComparing '{MEASURE}' aggregates:")
        differences = []
        for run_id in sorted(orig_ranking.keys()):
            orig_val = orig_ranking.get(run_id)
            recomp_val = recomp_ranking.get(run_id)
            if orig_val != recomp_val:
                diff = abs(orig_val - recomp_val) if recomp_val else float('inf')
                differences.append((run_id, orig_val, recomp_val, diff))
                print(f"  {run_id}: orig={orig_val:.6f}, recomp={recomp_val:.6f}, diff={diff:.6f}")

        if not differences:
            print("  No differences found - aggregates match!")
        else:
            print(f"\n  Total: {len(differences)} runs with different aggregates")

        # This test documents the difference, doesn't assert (yet)
        # Uncomment to make it fail on differences:
        # assert len(differences) == 0, f"{len(differences)} aggregates differ"

    def test_leaderboard_original_aggregates(self, truth_path):
        """Check what Leaderboard (v1) loads for aggregates."""
        from trec_auto_judge import Leaderboard

        lb = Leaderboard.load(truth_path, format="jsonl", has_header=False)

        print(f"\nLeaderboard: {len(lb.entries)} entries")
        print(f"Measures: {lb.measures[:5]}...")

        # Count aggregate vs per-topic entries
        agg_entries = [e for e in lb.entries if e.topic_id == lb.all_topic_id]
        per_topic = [e for e in lb.entries if e.topic_id != lb.all_topic_id]
        print(f"Aggregate entries (topic_id='{lb.all_topic_id}'): {len(agg_entries)}")
        print(f"Per-topic entries: {len(per_topic)}")

        # Get aggregate ranking
        if MEASURE in lb.measures:
            ranking = {}
            for e in lb.entries:
                if e.topic_id == lb.all_topic_id and MEASURE in e.values:
                    ranking[e.run_id] = float(e.values[MEASURE])
            print(f"\nAggregate ranking for '{MEASURE}':")
            for run_id, val in sorted(ranking.items())[:5]:
                print(f"  {run_id}: {val:.6f}")
            print(f"  ... ({len(ranking)} total runs)")


class TestCorrelationComparison:
    """Compare actual correlation outputs from v1 vs v2."""

    def test_v1_correlation(self, truth_path, eval_path):
        """Compute correlation using v1 (TrecLeaderboardEvaluation)."""
        from trec_auto_judge.evaluation import TrecLeaderboardEvaluation

        te = TrecLeaderboardEvaluation(
            truth_path,
            truth_measures=[MEASURE],
            eval_measures=None,  # use all eval measures
            truth_format="jsonl",
            truth_has_header=False,
            eval_format="tot",
            eval_has_header=False,
            on_missing="skip",
        )

        result = te.evaluate(eval_path)
        print(f"\nv1 results for truth_measure='{MEASURE}':")
        for (truth_m, eval_m), metrics in result.items():
            print(f"  {truth_m} vs {eval_m}:")
            for method, value in metrics.items():
                print(f"    {method}: {value:.6f}")

    def test_v2_correlation(self, truth_path, eval_path):
        """Compute correlation using v2 (LeaderboardEvaluator)."""
        from trec_auto_judge.evaluation_v2 import LeaderboardEvaluator

        te = LeaderboardEvaluator(
            truth_path,
            truth_measures=[MEASURE],
            eval_measures=None,
            truth_format="jsonl",
            truth_has_header=False,
            truth_drop_aggregate=False,
            eval_format="tot",
            eval_has_header=False,
            eval_drop_aggregate=False,
            on_missing="skip",
            correlation_methods=["kendall"],
            topic_ids=None,  # --all-topics
        )

        result = te.evaluate(eval_path)
        print(f"\nv2 results for truth_measure='{MEASURE}':")
        for (truth_m, eval_m), metrics in result.items():
            print(f"  {truth_m} vs {eval_m}:")
            for method, value in metrics.items():
                print(f"    {method}: {value:.6f}")

    def test_v1_vs_v2_side_by_side(self, truth_path, eval_path):
        """Run both and compare side by side."""
        from trec_auto_judge.evaluation import TrecLeaderboardEvaluation
        from trec_auto_judge.evaluation_v2 import LeaderboardEvaluator

        # v1
        te_v1 = TrecLeaderboardEvaluation(
            truth_path,
            truth_measures=[MEASURE],
            eval_measures=None,
            truth_format="jsonl",
            truth_has_header=False,
            eval_format="tot",
            eval_has_header=False,
            on_missing="skip",
            correlation_methods=["kendall"],
        )
        result_v1 = te_v1.evaluate(eval_path)

        # v2
        te_v2 = LeaderboardEvaluator(
            truth_path,
            truth_measures=[MEASURE],
            eval_measures=None,
            truth_format="jsonl",
            truth_has_header=False,
            truth_drop_aggregate=False,
            eval_format="tot",
            eval_has_header=False,
            eval_drop_aggregate=False,
            on_missing="skip",
            correlation_methods=["kendall"],
            topic_ids=None,
        )
        result_v2 = te_v2.evaluate(eval_path)

        print(f"\n{'='*60}")
        print(f"Comparison for truth_measure='{MEASURE}'")
        print(f"{'='*60}")

        for (truth_m, eval_m), v1_metrics in result_v1.items():
            v2_metrics = result_v2.get((truth_m, eval_m), {})
            print(f"\n{truth_m} vs {eval_m}:")

            all_methods = set(v1_metrics.keys()) | set(v2_metrics.keys())
            for method in sorted(all_methods):
                v1_val = v1_metrics.get(method)
                v2_val = v2_metrics.get(method)
                if v1_val is not None and v2_val is not None:
                    diff = abs(v1_val - v2_val)
                    match = "✓" if diff < 0.0001 else "✗"
                    print(f"  {method}: v1={v1_val:.6f}, v2={v2_val:.6f}, diff={diff:.6f} {match}")
                else:
                    print(f"  {method}: v1={v1_val}, v2={v2_val}")