"""Tests for Day 81 — the guards around the data, the split, and the metric."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from audit_engine import churn_by, shadow_levels, tenure_buckets
from baseline_engine import RateTable, recall_at_top, run_baselines
from churn_data import FILENAME, RAW_COLUMNS, clean, fetch, load_clean, load_raw
from split_engine import load_split, stratified_split, write_split

HERE = Path(__file__).resolve().parent
CACHED = (HERE / "data" / FILENAME).exists()


def _raw_rows(n=40, seed=0):
    """A small raw table in the real file's shape: every value is text."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, size=n)
    monthly = rng.uniform(20, 110, size=n).round(2)
    rows = {col: ["No"] * n for col in RAW_COLUMNS}
    rows["customerID"] = [f"{i:04d}-TEST" for i in range(n)]
    rows["gender"] = rng.choice(["Male", "Female"], size=n).tolist()
    rows["SeniorCitizen"] = rng.choice(["0", "1"], size=n).tolist()
    rows["tenure"] = [str(t) for t in tenure]
    rows["InternetService"] = rng.choice(["DSL", "Fiber optic", "No"], size=n).tolist()
    rows["Contract"] = rng.choice(["Month-to-month", "One year", "Two year"], size=n).tolist()
    rows["MonthlyCharges"] = [str(m) for m in monthly]
    rows["TotalCharges"] = [f"{t * m:.2f}" for t, m in zip(tenure, monthly)]
    rows["Churn"] = rng.choice(["No", "Yes"], size=n, p=[0.7, 0.3]).tolist()
    return pd.DataFrame(rows, columns=list(RAW_COLUMNS))


class CleanTests(unittest.TestCase):
    def test_blank_total_for_a_new_customer_becomes_zero(self):
        raw = _raw_rows()
        raw.loc[0, ["tenure", "TotalCharges"]] = ["0", " "]
        df = clean(raw)
        self.assertEqual(df.loc[0, "TotalCharges"], 0.0)
        self.assertTrue(set(df["SeniorCitizen"]) <= {"No", "Yes"})
        self.assertTrue(set(df["Churn"]) <= {0, 1})

    def test_blank_total_for_an_old_customer_is_refused(self):
        raw = _raw_rows()
        raw.loc[0, ["tenure", "TotalCharges"]] = ["12", ""]
        with self.assertRaisesRegex(ValueError, "tenure > 0"):
            clean(raw)

    def test_odd_target_and_repeated_ids_are_refused(self):
        raw = _raw_rows()
        raw.loc[3, "Churn"] = "Maybe"
        with self.assertRaises(ValueError):
            clean(raw)
        raw = _raw_rows()
        raw.loc[1, "customerID"] = raw.loc[0, "customerID"]
        with self.assertRaisesRegex(ValueError, "repeats"):
            clean(raw)

    def test_wrong_header_and_wrong_hash_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / FILENAME
            _raw_rows().rename(columns={"Churn": "Left"}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "unexpected columns"):
                load_raw(path)
            # the file already exists, so fetch() checks it without downloading
            with self.assertRaisesRegex(ValueError, "sha256"):
                fetch(tmp)


class AuditTests(unittest.TestCase):
    def test_mirrored_levels_are_found(self):
        df = pd.DataFrame(
            {
                "InternetService": ["No", "DSL", "No", "Fiber optic"],
                "TechSupport": ["No internet service", "Yes", "No internet service", "No"],
                "gender": ["Male", "Female", "Female", "Male"],
            }
        )
        groups = shadow_levels(df, cols=df.columns)
        self.assertIn([("InternetService", "No"), ("TechSupport", "No internet service")], groups)
        self.assertFalse(any(col == "gender" for group in groups for col, _ in group))

    def test_tenure_zero_lands_in_the_first_band(self):
        df = pd.DataFrame({"tenure": [0, 3, 6, 7, 72], "Churn": [1, 1, 0, 0, 0]})
        bands = tenure_buckets(df)
        self.assertEqual(bands.loc[0, "band"], "0–6")
        self.assertEqual(int(bands.loc[0, "rows"]), 3)
        self.assertEqual(int(bands["rows"].sum()), 5)

    def test_churn_by_puts_the_riskiest_level_first(self):
        df = pd.DataFrame({"Contract": ["a", "a", "b", "b"], "Churn": [0, 0, 1, 0]})
        self.assertEqual(churn_by(df, "Contract").loc[0, "Contract"], "b")


class SplitTests(unittest.TestCase):
    def test_each_slice_keeps_the_churn_rate(self):
        df = clean(_raw_rows(n=400, seed=1))
        slices = stratified_split(df, seed=5)
        overall = df["Churn"].mean()
        for name in ("train", "validation", "test"):
            self.assertAlmostEqual(df.loc[slices == name, "Churn"].mean(), overall, delta=0.02)

    def test_round_trip_and_mismatch(self):
        df = clean(_raw_rows(n=60, seed=2))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_split(df, stratified_split(df), Path(tmp) / "split.tsv")
            parts = load_split(df, path)
            self.assertEqual(sum(len(p) for p in parts.values()), len(df))
            ids = [set(p["customerID"]) for p in parts.values()]
            self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
            with self.assertRaisesRegex(ValueError, "different customers"):
                load_split(df.iloc[:-1], path)


class MetricTests(unittest.TestCase):
    def test_constant_score_gets_exactly_the_call_share(self):
        y = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
        self.assertAlmostEqual(recall_at_top(y, np.zeros(10), frac=0.2), 0.2)

    def test_ties_at_the_cut_are_shared(self):
        # 2 slots: the 0.9 row (a leaver) is in, then one slot shared by three 0.5 rows
        y = [1, 1, 0, 0, 1]
        score = [0.9, 0.5, 0.5, 0.5, 0.1]
        self.assertAlmostEqual(recall_at_top(y, score, frac=0.4), (1 + 1 / 3) / 3)

    def test_rate_table_shrinks_small_cells_and_falls_back(self):
        train = pd.DataFrame({"Contract": ["a"] * 8 + ["b"] * 2, "Churn": [0] * 8 + [1, 1]})
        table = RateTable(["Contract"], prior_rows=2).fit(train)
        # overall 0.2; cell b is (2 + 2 * 0.2) / (2 + 2) = 0.6, not 1.0
        p = table.predict(pd.DataFrame({"Contract": ["b", "a", "zzz"]}))
        np.testing.assert_allclose(p, [0.6, (0 + 0.4) / 10, 0.2])


@unittest.skipUnless(CACHED, "Telco CSV not cached; run main.py once")
class RealDataTests(unittest.TestCase):
    def test_committed_split_matches_the_seed_and_keeps_its_sizes(self):
        df = load_clean(HERE / "data")
        parts = load_split(df, HERE / "split_ids.tsv")
        self.assertEqual([len(p) for p in parts.values()], [4930, 1056, 1057])
        fresh = stratified_split(df, seed=81)
        frozen = df["customerID"].map(pd.read_csv(HERE / "split_ids.tsv", sep="\t").set_index("customerID")["slice"])
        self.assertTrue((fresh == frozen).all())

    def test_more_keys_make_a_better_call_list(self):
        df = load_clean(HERE / "data")
        parts = load_split(df, HERE / "split_ids.tsv")
        results = run_baselines(parts["train"], parts["validation"])
        recalls = results["top20_recall"].to_numpy()
        self.assertAlmostEqual(recalls[0], 0.2)
        self.assertTrue(np.all(np.diff(recalls) > 0))


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import KickoffPlots

        table = pd.DataFrame({"Contract": ["a", "b"], "rate": [0.4, 0.1], "rows": [10, 12]})
        y = np.array([1, 0, 0, 1, 0])
        with tempfile.TemporaryDirectory() as tmp:
            plots = KickoffPlots(tmp)
            paths = [plots.churn_profile([("Contract", table)], 0.25), plots.gains(y, {"s": np.arange(5.0)})]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
