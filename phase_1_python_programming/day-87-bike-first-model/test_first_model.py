"""Tests for Day 87 — no peeking at the forecast day, the log-ratio ridge, and the folds."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DATA_DIR
from bike_data import ZIP_NAME
from correction_model import CorrectionRidge
from evaluate_engine import error_by, rolling_folds
from features import LADDER, blocks, design, hinge_block, offset, versus_lags

CACHED = (DATA_DIR / ZIP_NAME).exists()


def _grid(weeks=7, seed=0):
    """A synthetic hourly grid with the columns the features read."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2011-01-03", periods=168 * weeks, freq="h")
    return pd.DataFrame(
        {
            "ts": ts,
            "date": ts.normalize(),
            "hr": ts.hour,
            "workingday": (ts.dayofweek < 5).astype(int),
            "holiday": 0,
            "weathersit": rng.integers(1, 4, size=len(ts)),
            "temp": rng.uniform(0.1, 0.9, size=len(ts)),
            "hum": rng.uniform(0.3, 1.0, size=len(ts)),
            "outage": False,
            "cnt": rng.integers(5, 500, size=len(ts)).astype(float),
        }
    )


class NoPeekingTests(unittest.TestCase):
    def test_a_days_own_counts_never_reach_its_features(self):
        g = _grid()
        day = g["date"].unique()[-3]
        target = g["date"] == day
        before = pd.concat([design(g, rung) for rung in LADDER] + [offset(g)], axis=1)
        changed = g.copy()
        changed.loc[target, "cnt"] = 99999.0
        after = pd.concat([design(changed, rung) for rung in LADDER] + [offset(changed)], axis=1)
        # the forecast day itself and everything before it are untouched
        pd.testing.assert_frame_equal(before[g["date"] <= day], after[g["date"] <= day])

    def test_versus_lags_of_a_steady_series_is_zero(self):
        s = pd.Series(np.full(168 * 6, 0.4))
        self.assertTrue(np.allclose(versus_lags(s).dropna(), 0.0))
        # like the baseline, it uses whatever lag weeks exist: only week one has none
        self.assertTrue(versus_lags(s).iloc[:168].isna().all())
        self.assertFalse(versus_lags(s).iloc[168:].isna().any())

    def test_hinge_block_columns(self):
        block = hinge_block(pd.Series(np.linspace(0, 1, 168 * 5)), (0.3, 0.7), "temp")
        self.assertEqual(list(block.columns), ["temp_vs_lags", "temp_over_0.3_vs_lags", "temp_over_0.7_vs_lags"])

    def test_ladder_rungs_only_add_columns(self):
        g = _grid()
        cache = blocks(g)
        widths = [design(g, rung, cache).shape[1] for rung in LADDER]
        self.assertEqual(widths, sorted(widths))
        self.assertEqual(widths[0], 48)


class RidgeTests(unittest.TestCase):
    def test_recovers_a_known_log_ratio(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(5000, 2)), columns=["a", "b"])
        base = rng.uniform(20, 400, size=5000)
        # y = (1 + base) * exp(0.1 + 0.3 a − 0.2 b) − 1, with no noise
        y = np.expm1(np.log1p(base) + 0.1 + 0.3 * X["a"] - 0.2 * X["b"])
        model = CorrectionRidge(lam=0.0).fit(X, y, base)
        np.testing.assert_allclose(model.raw_effects(), [0.3, -0.2], atol=1e-9)
        np.testing.assert_allclose(model.predict(X, base), y, rtol=1e-9)
        self.assertEqual(model.names, ("a", "b"))

    def test_predictions_are_floored_at_zero(self):
        X = np.array([[0.0], [1.0], [2.0]])
        model = CorrectionRidge(lam=0.0).fit(X, [10.0, 1.0, 0.0], [10.0, 10.0, 10.0])
        self.assertTrue((model.predict(np.array([[50.0]]), [10.0]) >= 0).all())

    def test_bad_inputs_are_refused(self):
        with self.assertRaises(ValueError):
            CorrectionRidge(lam=-1)
        with self.assertRaises(ValueError):
            CorrectionRidge().fit(np.array([[np.nan]]), [1.0], [1.0])
        with self.assertRaises(ValueError):
            CorrectionRidge().fit(np.array([[1.0]]), [-1.0], [1.0])
        with self.assertRaises(RuntimeError):
            CorrectionRidge().predict(np.zeros((1, 1)), [1.0])


class FoldTests(unittest.TestCase):
    def test_each_fold_only_fits_on_earlier_days(self):
        g = _grid(weeks=20)
        slices = pd.Series("train", index=g.index)
        months = ("2011-03", "2011-04")
        for name, fit, forecast in rolling_folds(g, slices, months):
            self.assertLess(g.loc[fit, "ts"].max(), g.loc[forecast, "ts"].min())
            self.assertTrue((g.loc[forecast, "ts"].dt.strftime("%Y-%m") == name).all())

    def test_a_month_outside_train_is_refused(self):
        g = _grid(weeks=8)
        with self.assertRaises(ValueError):
            rolling_folds(g, pd.Series("train", index=g.index), ("2015-01",))

    def test_error_by_groups(self):
        table = error_by([10, 20, 30], [12, 20, 20], ["a", "a", "b"])
        self.assertEqual(table.set_index("group")["mae"].to_dict(), {"a": 1.0, "b": 10.0})


@unittest.skipUnless(CACHED, "bike zip not cached; run the Day 86 main.py once")
class RealDataTests(unittest.TestCase):
    def test_the_ladder_improves_under_cv_and_beats_the_bar_on_validation(self):
        from baseline_engine import mae
        from bike_data import load_grid
        from evaluate_engine import baseline_cv, score_ladder
        from project_paths import SPLIT_FILE
        from split_engine import assign_slices, load_split_dates, scorable

        grid = load_grid(DATA_DIR)
        slices = assign_slices(grid, load_split_dates(SPLIT_FILE))
        results = score_ladder(grid, slices)
        cv = [np.mean(list(baseline_cv(grid, slices).values()))] + [r.cv_mae for r in results]
        self.assertTrue(np.all(np.diff(cv) < 0), cv)
        base = offset(grid)
        val = (slices == "validation") & scorable(grid) & base.notna()
        bar = mae(grid.loc[val, "cnt"], base[val])
        self.assertLess(results[1].val_mae, bar)
        self.assertLess(results[2].val_mae, results[1].val_mae)


class PlotSmokeTests(unittest.TestCase):
    def test_this_days_visualizer_is_imported_and_writes(self):
        import visualizer

        self.assertTrue(hasattr(visualizer, "ModelPlots"))
        with tempfile.TemporaryDirectory() as tmp:
            plots = visualizer.ModelPlots(tmp)
            xs = np.linspace(0, 1, 10)
            paths = [
                plots.ladder([("a", 40.0, 42.0), ("b", 35.0, 39.0)]),
                plots.cv_months({"a": {"2012-01": 40.0, "2012-02": 38.0}}),
                plots.weather_curves([("t", xs, np.exp(xs), "0")]),
                plots.error_by_weather([("clear", 10, 40.0, 35.0)]),
            ]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
