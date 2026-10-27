"""Tests for Day 86 — the grid rebuild, the time split, and the day-ahead baselines."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_engine import day_block_bootstrap, lag, mae, recent_weeks_mean, rmse, train_profile, wape
from bike_data import RAW_COLUMNS, ZIP_NAME, complete_grid, fetch
from split_engine import assign_slices, load_split_dates, scorable, write_split_dates

HERE = Path(__file__).resolve().parent
CACHED = (HERE / "data" / ZIP_NAME).exists()


def _raw(days=3, drop=()):
    """A tiny hour.csv: `days` days of hours, minus the (day, hour) pairs in `drop`."""
    rows = []
    for d in range(days):
        date = pd.Timestamp("2011-01-01") + pd.Timedelta(days=d)
        for h in range(24):
            if (d, h) in drop:
                continue
            rows.append(
                {
                    "instant": len(rows) + 1, "dteday": date.strftime("%Y-%m-%d"), "season": 1, "yr": 0,
                    "mnth": 1, "hr": h, "holiday": 0, "weekday": (date.dayofweek + 1) % 7,
                    "workingday": int(date.dayofweek < 5), "weathersit": 1, "temp": 0.2 + 0.01 * h,
                    "atemp": 0.2, "hum": 0.5, "windspeed": 0.1, "casual": 1, "registered": h + 1, "cnt": h + 2,
                }
            )
    return pd.DataFrame(rows, columns=list(RAW_COLUMNS))


class GridTests(unittest.TestCase):
    def test_a_quiet_gap_becomes_zero_and_a_shutdown_day_becomes_an_outage(self):
        shutdown = {(1, h) for h in range(8)}
        grid = complete_grid(_raw(drop={(0, 3)} | shutdown))
        self.assertEqual(len(grid), 72)
        quiet = grid[(grid["date"] == "2011-01-01") & (grid["hr"] == 3)].iloc[0]
        self.assertEqual(quiet["cnt"], 0)
        self.assertFalse(quiet["outage"])
        day2 = grid[grid["date"] == "2011-01-02"]
        self.assertTrue(day2["outage"].all())
        # even the hours that were recorded on a shutdown day lose their counts
        self.assertTrue(day2["cnt"].isna().all())

    def test_calendar_and_weather_are_filled_for_missing_hours(self):
        grid = complete_grid(_raw(drop={(0, 5)}))
        row = grid[(grid["date"] == "2011-01-01") & (grid["hr"] == 5)].iloc[0]
        # 2011-01-01 was a Saturday: weekday 6 in the file's convention, not a working day
        self.assertEqual((row["weekday"], row["workingday"]), (6, 0))
        self.assertAlmostEqual(row["temp"], 0.25)
        self.assertEqual(row["weathersit"], 1)

    def test_a_repeated_hour_is_refused(self):
        raw = _raw(days=1)
        with self.assertRaisesRegex(ValueError, "twice"):
            complete_grid(pd.concat([raw, raw.iloc[[0]]], ignore_index=True))

    def test_a_cached_file_with_the_wrong_hash_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / ZIP_NAME).write_bytes(b"not the real zip")
            with self.assertRaisesRegex(ValueError, "sha256"):
                fetch(tmp)


class SplitTests(unittest.TestCase):
    def test_ranges_must_be_back_to_back_and_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.tsv"
            write_split_dates(path)
            self.assertEqual(len(load_split_dates(path)), 3)
            bad = [
                (("train", "2011-01-01", "2012-06-30"), ("validation", "2012-07-02", "2012-09-30"), ("test", "2012-10-01", "2012-12-31")),
                (("train", "2011-01-01", "2012-07-15"), ("validation", "2012-07-01", "2012-09-30"), ("test", "2012-10-01", "2012-12-31")),
                (("validation", "2011-01-01", "2012-06-30"), ("train", "2012-07-01", "2012-09-30"), ("test", "2012-10-01", "2012-12-31")),
            ]
            for ranges in bad:
                write_split_dates(path, ranges)
                with self.assertRaises(ValueError):
                    load_split_dates(path)

    def test_every_hour_gets_a_slice_and_outages_are_never_scorable(self):
        grid = complete_grid(_raw(days=3, drop={(2, h) for h in range(10)}))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_split_dates(
                Path(tmp) / "s.tsv",
                (("train", "2011-01-01", "2011-01-01"), ("validation", "2011-01-02", "2011-01-02"), ("test", "2011-01-03", "2011-01-03")),
            )
            slices = assign_slices(grid, load_split_dates(path))
        self.assertEqual(slices.value_counts().to_dict(), {"train": 24, "validation": 24, "test": 24})
        self.assertFalse(scorable(grid)[slices == "test"].any())


class BaselineTests(unittest.TestCase):
    def _grid(self, weeks=5):
        hours = 168 * weeks
        ts = pd.date_range("2011-01-03", periods=hours, freq="h")
        return pd.DataFrame(
            {"ts": ts, "date": ts.normalize(), "hr": ts.hour, "workingday": (ts.dayofweek < 5).astype(int), "cnt": np.arange(hours, dtype=float)}
        )

    def test_short_lags_are_refused(self):
        with self.assertRaisesRegex(ValueError, "24 hours"):
            lag(self._grid(), 23)

    def test_four_week_mean_skips_a_missing_week(self):
        g = self._grid()
        g.loc[4 * 168 - 168 : 4 * 168 - 1, "cnt"] = np.nan  # week 4 (the one just before the last) is an outage
        row = 4 * 168 + 10
        got = recent_weeks_mean(g, 4).iloc[row]
        expected = np.mean([row - 168 * k for k in (2, 3, 4)])
        self.assertAlmostEqual(got, expected)

    def test_training_profile_ignores_later_rows(self):
        g = self._grid()
        train = g["ts"] < "2011-01-17"
        before = train_profile(g, train)
        g.loc[~train, "cnt"] = 1e6
        after = train_profile(g, train)
        pd.testing.assert_series_equal(before, after)

    def test_metrics_on_a_hand_example(self):
        y, p = [10, 0, 30], [12, 3, 24]
        self.assertAlmostEqual(mae(y, p), 11 / 3)
        self.assertAlmostEqual(rmse(y, p), np.sqrt((4 + 9 + 36) / 3))
        self.assertAlmostEqual(wape(y, p), 11 / 40)
        with self.assertRaises(ValueError):
            wape([0, 0], [1, 1])

    def test_day_bootstrap_of_identical_forecasts_is_zero(self):
        g = self._grid(weeks=1)
        out = day_block_bootstrap(g["cnt"], g["cnt"] + 1, g["cnt"] + 1, g["date"], n_boot=100)
        self.assertEqual((out["diff"], out["lo"], out["hi"], out["days"]), (0.0, 0.0, 0.0, 7))


@unittest.skipUnless(CACHED, "bike zip not cached; run main.py once")
class RealDataTests(unittest.TestCase):
    def test_committed_split_and_the_baseline_ranking(self):
        from baseline_engine import baselines
        from bike_data import load_grid

        grid = load_grid(HERE / "data")
        slices = assign_slices(grid, load_split_dates(HERE / "split_dates.tsv"))
        self.assertEqual(int(grid.loc[slices == "validation", "outage"].sum()), 0)
        self.assertEqual(grid.loc[grid["outage"], "date"].nunique(), 8)
        preds = baselines(grid, slices == "train")
        val = (slices == "validation") & scorable(grid)
        y = grid.loc[val, "cnt"]
        four = mae(y, preds["mean of last 4 weeks, same hour"][val])
        week = mae(y, preds["last week, same hour"][val])
        frozen = mae(y, preds["training-period profile"][val])
        self.assertLess(four, week)
        self.assertLess(week, frozen)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import BikePlots

        grid = complete_grid(_raw(days=3, drop={(1, h) for h in range(8)}))
        ranges = pd.DataFrame(
            {"slice": ["train", "validation", "test"], "first_day": pd.to_datetime(["2011-01-01", "2011-01-02", "2011-01-03"]), "last_day": pd.to_datetime(["2011-01-01", "2011-01-02", "2011-01-03"])}
        )
        shapes = pd.DataFrame({"working day": np.arange(24.0), "weekend / holiday": np.arange(24.0)[::-1]})
        with tempfile.TemporaryDirectory() as tmp:
            plots = BikePlots(tmp)
            paths = [
                plots.timeline(grid, ranges),
                plots.daily_shapes(shapes),
                plots.sample_week(grid["ts"][:24], grid["cnt"][:24], {"a": grid["cnt"][:24] + 1}),
            ]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
