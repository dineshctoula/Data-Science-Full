"""Tests for Day 88 — the noise generator, today-vs-history features, and the summary."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DATA_DIR
from bike_data import ZIP_NAME
import features as day87
from forecast_features import NO_WEATHER_RUNG, WEATHER_RUNG, design
from stress_engine import Run, summarize
from weather_noise import PERFECT, SCENARIOS, Scenario, perturb, realized_errors

CACHED = (DATA_DIR / ZIP_NAME).exists()


def _grid(weeks=6, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2011-01-03", periods=168 * weeks, freq="h")
    return pd.DataFrame(
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


class NoiseTests(unittest.TestCase):
    def test_a_perfect_forecast_changes_nothing(self):
        g = _grid()
        f = perturb(g, PERFECT, seed=3)
        pd.testing.assert_frame_equal(f, g, check_dtype=False)

    def test_only_weather_columns_move(self):
        g = _grid()
        f = perturb(g, SCENARIOS[1], seed=3)
        for col in ("cnt", "hr", "workingday", "holiday", "date"):
            pd.testing.assert_series_equal(f[col], g[col])
        self.assertFalse(np.allclose(f["temp"], g["temp"]))

    def test_temperature_error_is_shared_within_a_day(self):
        g = _grid()
        day_only = Scenario("day bias only", 3.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        err = perturb(g, day_only, seed=1)["temp"] - g["temp"]
        per_day = err.groupby(g["date"]).std()
        self.assertTrue(np.allclose(per_day, 0.0))
        self.assertGreater(err.groupby(g["date"]).mean().std(), 0.0)

    def test_rain_rates_come_out_near_the_assumptions(self):
        g = _grid(weeks=60)
        sc = Scenario("t", 0.0, 0.0, 0.0, rain_hit=0.5, rain_false=0.1, mist_flip=0.0)
        e = realized_errors(g, perturb(g, sc, seed=2))
        self.assertAlmostEqual(e["rain_hit"], 0.5, delta=0.03)
        self.assertAlmostEqual(e["rain_false"], 0.1, delta=0.01)

    def test_same_seed_same_forecast_and_bad_rates_refused(self):
        g = _grid()
        pd.testing.assert_frame_equal(perturb(g, SCENARIOS[0], 7), perturb(g, SCENARIOS[0], 7))
        with self.assertRaises(ValueError):
            Scenario("bad", 1.0, 1.0, 1.0, rain_hit=1.5, rain_false=0.0, mist_flip=0.0)


class FeatureTests(unittest.TestCase):
    def test_perfect_forecast_reproduces_day_87(self):
        g = _grid()
        pd.testing.assert_frame_equal(design(g, g, WEATHER_RUNG), day87.design(g, WEATHER_RUNG))

    def test_a_days_forecast_only_moves_that_day(self):
        g = _grid()
        day = g["date"].unique()[-10]
        forecast = g.copy()
        forecast.loc[g["date"] == day, "temp"] += 0.2
        before, after = design(g, g, WEATHER_RUNG), design(g, forecast, WEATHER_RUNG)
        # NaN != NaN, so the first week (no lag history yet) has to count as unchanged explicitly
        same = (before == after) | (before.isna() & after.isna())
        changed = ~same.all(axis=1)
        # history always reads actual weather, so a week later is untouched
        self.assertTrue(changed[g["date"] == day].all())
        self.assertFalse(changed[g["date"] != day].any())

    def test_the_weather_rung_needs_a_forecast(self):
        g = _grid()
        with self.assertRaises(ValueError):
            design(g, None, WEATHER_RUNG)
        self.assertEqual(design(g, None, NO_WEATHER_RUNG).shape[1], 50)


class SummaryTests(unittest.TestCase):
    def test_kept_share_formula(self):
        refs = {"no weather": {"val_mae": 50.0, "cv_mae": 40.0}, "perfect": {"val_mae": 40.0, "cv_mae": 30.0}}
        runs = [Run("s", "m", i, v, c, 0.8, pd.Series(dtype=float)) for i, (v, c) in enumerate([(44.0, 35.0), (46.0, 35.0)])]
        row = summarize(runs, refs).iloc[0]
        self.assertAlmostEqual(row["val_mae"], 45.0)
        self.assertAlmostEqual(row["val_mae_kept"], 0.5)
        self.assertAlmostEqual(row["cv_mae_kept"], 0.5)


@unittest.skipUnless(CACHED, "bike zip not cached; run the Day 86 main.py once")
class RealDataTests(unittest.TestCase):
    def test_training_on_forecasts_helps_and_trusts_rain_less(self):
        from bike_data import load_grid
        from project_paths import SPLIT_FILE
        from split_engine import assign_slices, load_split_dates
        from stress_engine import run_scenario

        grid = load_grid(DATA_DIR)
        slices = assign_slices(grid, load_split_dates(SPLIT_FILE))
        runs = run_scenario(grid, slices, SCENARIOS[1], seeds=[0, 1])
        for seed in (0, 1):
            actual = next(r for r in runs if r.seed == seed and r.mode == "trained on actual")
            matched = next(r for r in runs if r.seed == seed and r.mode == "trained on forecasts")
            self.assertLess(matched.val_mae, actual.val_mae)
            self.assertLess(matched.cv_mae, actual.cv_mae)
            self.assertGreater(matched.rain_effect, actual.rain_effect)


class PlotSmokeTests(unittest.TestCase):
    def test_this_days_visualizer_is_imported_and_writes(self):
        import visualizer

        self.assertTrue(hasattr(visualizer, "StressPlots"))
        refs = {"no weather": {"val_mae": 45.0, "cv_mae": 45.0}, "perfect": {"val_mae": 41.0, "cv_mae": 35.0}}
        summary = pd.DataFrame(
            [
                {"scenario": "a", "mode": m, "val_mae": 44.0, "val_mae_sd": 1.0, "cv_mae": 40.0, "cv_mae_sd": 0.5, "rain_effect": 0.7}
                for m in ("trained on actual", "trained on forecasts")
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            plots = visualizer.StressPlots(tmp)
            for path in (plots.survival(summary, refs, ["a"]), plots.rain_trust(summary, 0.64, ["a"])):
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
