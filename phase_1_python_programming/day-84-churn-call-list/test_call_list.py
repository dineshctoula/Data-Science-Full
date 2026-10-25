"""Tests for Day 84 — the economics, the call-list policies, and their scoring."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DATA_DIR
from churn_data import FILENAME
from economics import Economics
from policy_engine import (
    bootstrap_profit_diff,
    call_if_profitable,
    call_top_frac,
    evaluate,
    hindsight_best,
    profit_curve,
    sensitivity,
    sharpen,
)

CACHED = (DATA_DIR / FILENAME).exists()


def _bills(*monthly):
    return pd.DataFrame({"MonthlyCharges": np.asarray(monthly, dtype=float)})


class EconomicsTests(unittest.TestCase):
    def test_hand_numbers(self):
        e = Economics(call_cost=10, credit=20, save_rate=0.25, months_kept=12, margin=0.5)
        df = _bills(50.0, 100.0)
        np.testing.assert_allclose(e.value(df), [300.0, 600.0])
        np.testing.assert_allclose(e.break_even(df), [30 / 75, 30 / 150])
        # exactly at break-even, a call is worth nothing either way
        np.testing.assert_allclose(e.expected_profit(e.break_even(df), df), 0.0, atol=1e-12)
        np.testing.assert_allclose(e.realized_profit([1, 0], df), [0.25 * 300 - 30, -30.0])

    def test_bad_assumptions_are_rejected(self):
        for kwargs in ({"save_rate": 0}, {"save_rate": 1.5}, {"margin": 0}, {"call_cost": -1}, {"months_kept": 0}):
            with self.assertRaises(ValueError):
                Economics(**kwargs)

    def test_changing_the_save_rate_leaves_the_original_alone(self):
        e = Economics()
        f = e.with_save_rate(0.4)
        self.assertEqual((e.save_rate, f.save_rate), (0.25, 0.4))
        self.assertEqual(e.cost, f.cost)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(84)
        n = 400
        self.df = _bills(*rng.uniform(20, 115, size=n))
        self.p = rng.beta(2, 5, size=n)
        self.y = (rng.random(n) < self.p).astype(int)
        self.e = Economics()

    def test_top_frac_calls_the_right_count_and_the_highest_scores(self):
        called = call_top_frac(self.p, 0.2)
        self.assertEqual(called.sum(), 80)
        self.assertGreaterEqual(self.p[called].min(), self.p[~called].max())

    def test_break_even_list_is_a_prefix_of_the_value_ranking(self):
        # p * s * v - cost > 0  is the same as  p * v > cost / s, so the rule is "rank by p * v, stop early"
        called = call_if_profitable(self.p, self.df, self.e)
        order = np.argsort(-(self.p * self.e.value(self.df)), kind="stable")
        k = int(called.sum())
        self.assertTrue(called[order[:k]].all())
        self.assertFalse(called[order[k:]].any())

    def test_scoring_adds_up(self):
        called = call_top_frac(self.p, 0.3)
        r = evaluate(called, self.y, self.df, self.e)
        gain = self.e.realized_profit(self.y, self.df)
        self.assertAlmostEqual(r["profit"], gain[called].sum())
        self.assertAlmostEqual(r["cost"], called.sum() * 30.0)
        curve = profit_curve(np.arange(len(self.y)), self.y, self.df, self.e)
        self.assertEqual(curve[0], 0.0)
        self.assertAlmostEqual(curve[-1], gain.sum())

    def test_hindsight_is_a_ceiling_for_prefix_lists(self):
        score = self.p * self.e.value(self.df)
        _, best = hindsight_best(score, self.y, self.df, self.e)
        self.assertGreaterEqual(best, 0.0)
        for called in (call_if_profitable(self.p, self.df, self.e), call_top_frac(score, 0.2)):
            self.assertGreaterEqual(best + 1e-9, evaluate(called, self.y, self.df, self.e)["profit"])

    def test_sharpen_keeps_the_order_and_the_middle(self):
        q = sharpen(self.p, 2.0)
        np.testing.assert_array_equal(np.argsort(q, kind="stable"), np.argsort(self.p, kind="stable"))
        self.assertAlmostEqual(float(sharpen([0.5], 3.0)[0]), 0.5)
        self.assertGreater(sharpen([0.8], 2.0)[0], 0.8)

    def test_same_list_bootstraps_to_zero(self):
        called = call_top_frac(self.p, 0.2)
        d = bootstrap_profit_diff(called, called, self.y, self.df, self.e, n_boot=200)
        self.assertEqual((d["diff"], d["lo"], d["hi"]), (0.0, 0.0, 0.0))

    def test_a_better_save_rate_never_shrinks_the_list(self):
        table = sensitivity(self.p, self.y, self.df, self.e, [0.1, 0.2, 0.3, 0.4])
        self.assertTrue((np.diff(table["rule_calls"]) >= 0).all())


@unittest.skipUnless(CACHED, "Telco CSV not cached; run the Day 81 main.py once")
class RealDataTests(unittest.TestCase):
    def test_rule_beats_the_fixed_list_and_skips_hopeless_rates(self):
        from churn_data import TARGET, load_clean
        from policy_engine import champion_scores
        from project_paths import SPLIT_FILE
        from split_engine import load_split

        parts = load_split(load_clean(DATA_DIR), SPLIT_FILE)
        train, val = parts["train"], parts["validation"]
        p = champion_scores(train, val)
        y = val[TARGET].to_numpy()
        e = Economics()
        rule = evaluate(call_if_profitable(p, val, e), y, val, e)["profit"]
        fixed = evaluate(call_top_frac(p, 0.2), y, val, e)["profit"]
        self.assertGreater(rule, fixed)
        self.assertEqual(call_if_profitable(p, val, e.with_save_rate(0.05)).sum(), 0)


class PlotSmokeTests(unittest.TestCase):
    def test_this_days_visualizer_is_imported(self):
        import visualizer

        self.assertTrue(hasattr(visualizer, "CallPlots"))

    def test_plots_write_files(self):
        from visualizer import CallPlots

        df = _bills(30, 60, 90, 110)
        p = np.array([0.1, 0.5, 0.3, 0.7])
        e = Economics()
        table = sensitivity(p, [0, 1, 0, 1], df, e, [0.1, 0.3])
        with tempfile.TemporaryDirectory() as tmp:
            plots = CallPlots(tmp)
            paths = [
                plots.profit_curves({"a": np.array([0.0, 5.0, 3.0])}, [("m", 1, 5.0)]),
                plots.break_even_map(df["MonthlyCharges"], p, np.minimum(e.break_even(df), 1), p > 0.4),
                plots.sensitivity(table, 0.25),
            ]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
