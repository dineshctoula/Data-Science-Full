"""Tests for Day 89 — the recipe, the drift fallback, the engine, and the gate wiring.

None of these read the test slice: synthetic grids and the validation
hours stand in for it, which is the point of having a gate.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DATA_DIR, HERE
import final_recipe as R
from bike_data import ZIP_NAME
from features import blocks
from final_engine import card, day_interval, robust_blocks, run_models

CACHED = (DATA_DIR / ZIP_NAME).exists()


def _grid(weeks=8, outage_day=None, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2011-01-03", periods=168 * weeks, freq="h")
    g = pd.DataFrame(
        {
            "ts": ts,
            "date": ts.normalize(),
            "hr": ts.hour,
            "workingday": (ts.dayofweek < 5).astype(int),
            "holiday": 0,
            "weathersit": rng.choice([1.0, 2.0, 3.0], size=len(ts), p=[0.6, 0.3, 0.1]),
            "temp": rng.uniform(0.1, 0.9, size=len(ts)),
            "hum": rng.uniform(0.3, 1.0, size=len(ts)),
            "outage": False,
            "cnt": rng.integers(5, 500, size=len(ts)).astype(float),
        }
    )
    if outage_day is not None:
        hit = g["date"] == g["date"].unique()[outage_day]
        g.loc[hit, "outage"] = True
        g.loc[hit, "cnt"] = np.nan
    return g


class RecipeTests(unittest.TestCase):
    def test_recipe_names_what_the_engine_produces(self):
        self.assertNotIn("test", R.FIT_ON)
        self.assertEqual(R.FORECAST.name, "typical forecast")
        self.assertEqual(len(R.SEEDS), 10)
        self.assertEqual(R.LAM, 0.1)
        # the sample week is fixed in advance and lies in the test quarter
        self.assertTrue("2012-10-01" <= R.SAMPLE_WEEK[0] <= R.SAMPLE_WEEK[1] <= "2012-12-31")


class DriftTests(unittest.TestCase):
    def test_the_day_after_an_outage_uses_the_last_good_day(self):
        g = _grid(outage_day=40)
        days = g["date"].unique()
        after = g["date"] == days[41]
        plain = blocks(g)["holiday + drift"]["drift"]
        robust = robust_blocks(g)["holiday + drift"]["drift"]
        self.assertTrue(plain[after].isna().all())
        self.assertTrue(np.isfinite(robust[after]).all())
        # the stand-in "yesterday" is the day before the outage
        day_before = g.loc[g["date"] == days[39], "cnt"].mean()
        lag_days = np.nanmean([g.loc[g["date"] == days[41 - 7 * k], "cnt"].mean() for k in range(1, 5)])
        self.assertAlmostEqual(float(robust[after].iloc[0]), float(np.log1p(day_before) - np.log1p(lag_days)))

    def test_days_not_touching_an_outage_are_unchanged(self):
        g = _grid(outage_day=40)
        plain = blocks(g)["holiday + drift"]["drift"]
        robust = robust_blocks(g)["holiday + drift"]["drift"]
        both = plain.notna()
        pd.testing.assert_series_equal(plain[both], robust[both])


class EngineTests(unittest.TestCase):
    def test_every_model_is_scored_on_the_same_hours(self):
        g = _grid(weeks=10)
        cut = g["date"] >= g["date"].unique()[56]
        run = run_models(g, ~cut, cut, R)
        for pred in list(run.preds.values()) + list(run.primary_by_seed.values()):
            self.assertTrue(pred.index.equals(g.index[run.scored]))
        self.assertEqual(len(run.primary_by_seed), len(R.SEEDS))
        frame = card(g, run, R)
        self.assertEqual(list(frame["model"])[-1], R.PRIMARY)
        self.assertIn(R.COMPARISON, list(frame["model"]))

    def test_scored_hours_never_feed_the_fit(self):
        g = _grid(weeks=10)
        cut = g["date"] >= g["date"].unique()[56]
        first_day = g["date"] == g["date"].unique()[56]
        before = run_models(g, ~cut, cut, R)
        later = g.copy()
        # change the last scored day: it can't reach the fit, or the first scored day's features
        later.loc[later["date"] == later["date"].unique()[-1], "cnt"] = 1e5
        after = run_models(later, ~cut, cut, R)
        for label in before.preds:
            pd.testing.assert_series_equal(before.preds[label][first_day[before.scored].to_numpy()], after.preds[label][first_day[after.scored].to_numpy()])

    def test_day_interval_brackets_the_point(self):
        g = _grid(weeks=2)
        point, lo, hi = day_interval(g["cnt"], g["cnt"] * 1.1, g["date"], n_boot=200, seed=1)
        self.assertLessEqual(lo, point)
        self.assertLessEqual(point, hi)


class GateWiringTests(unittest.TestCase):
    def test_main_reaches_test_only_through_the_gate(self):
        source = (HERE / "main.py").read_text()
        self.assertIn("gate.open_test(", source)
        # the test mask is built in exactly one place: inside the gate call
        self.assertEqual(source.count('slices == "test"'), 1)
        self.assertIn('gate.open_test({"test": slices == "test"}', source)


@unittest.skipUnless(CACHED, "bike zip not cached; run the Day 86 main.py once")
class ValidationTests(unittest.TestCase):
    def test_validation_reproduces_days_87_and_88(self):
        from bike_data import load_grid
        from project_paths import SPLIT_FILE
        from split_engine import assign_slices, load_split_dates

        grid = load_grid(DATA_DIR)
        slices = assign_slices(grid, load_split_dates(SPLIT_FILE))
        frame = card(grid, run_models(grid, slices == "train", slices == "validation", R), R).set_index("model")
        self.assertAlmostEqual(frame.loc["4-week baseline", "mae"], 47.08, places=2)
        self.assertAlmostEqual(frame.loc["no weather", "mae"], 44.72, places=1)
        self.assertAlmostEqual(frame.loc["weather, actual (upper bound)", "mae"], 41.19, places=1)
        self.assertAlmostEqual(frame.loc[R.PRIMARY, "mae"], 44.8, delta=0.1)


class PlotSmokeTests(unittest.TestCase):
    def test_this_days_visualizer_is_imported_and_writes(self):
        import visualizer

        self.assertTrue(hasattr(visualizer, "FinalPlots"))
        frame = pd.DataFrame({"model": ["a", "b"], "mae": [40.0, 38.0], "lo": [36.0, 34.0], "hi": [44.0, 42.0]})
        gaps = pd.DataFrame({"seed": [0, 1], "diff": [-1.0, 0.5], "lo": [-3.0, -1.5], "hi": [1.0, 2.5]})
        with tempfile.TemporaryDirectory() as tmp:
            plots = visualizer.FinalPlots(tmp)
            for path in (plots.cards({"val": frame, "test": frame}), plots.seed_gaps(gaps, "b")):
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
