"""Tests for Day 82 — the encoder, the Newton fit, and the selection rule."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import DATA_DIR, SPLIT_FILE
from churn_data import FILENAME, TARGET
from features import ADDONS, Encoder, add_log_tenure, collapse_mirrors
from logit_engine import NewtonLogistic, sigmoid
from select_engine import Trial, paired_bootstrap, pick_simplest

CACHED = (DATA_DIR / FILENAME).exists()


class FeatureTests(unittest.TestCase):
    def test_collapse_folds_the_mirrored_levels(self):
        df = pd.DataFrame({col: ["No internet service", "Yes", "No"] for col in ADDONS})
        df["MultipleLines"] = ["No phone service", "Yes", "No"]
        out = collapse_mirrors(df)
        for col in list(ADDONS) + ["MultipleLines"]:
            self.assertEqual(out[col].tolist(), ["No", "Yes", "No"])
        # the input is left alone
        self.assertEqual(df["MultipleLines"].iloc[0], "No phone service")

    def test_log_tenure_handles_zero(self):
        out = add_log_tenure(pd.DataFrame({"tenure": [0, 1, 71]}))
        np.testing.assert_allclose(out["log_tenure"], np.log([1, 2, 72]))

    def test_encoder_drops_the_most_common_level_and_uses_train_stats(self):
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "c": ["b", "a", "b", "c"]})
        enc = Encoder(["x"], ["c"]).fit(train)
        self.assertEqual(enc.reference("c"), "b")
        self.assertEqual(enc.names, ("x", "c=a", "c=c"))
        X = enc.transform(train)
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-12)
        # a level never seen in training lands on the reference: all one-hot columns 0
        raw = enc._unscaled(pd.DataFrame({"x": [9.0], "c": ["zzz"]}))
        np.testing.assert_array_equal(raw[0, 1:], [0.0, 0.0])

    def test_transform_before_fit_is_rejected(self):
        with self.assertRaises(RuntimeError):
            Encoder(["x"], []).transform(pd.DataFrame({"x": [1.0]}))


class NewtonTests(unittest.TestCase):
    def _data(self, n=20000, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 3))
        true_w = np.array([1.5, -2.0, 0.0])
        y = (rng.random(n) < sigmoid(-0.5 + X @ true_w)).astype(float)
        return X, y, true_w

    def test_recovers_known_coefficients(self):
        X, y, true_w = self._data()
        model = NewtonLogistic(lam=0.0).fit(X, y)
        np.testing.assert_allclose(model.coef, true_w, atol=0.08)
        self.assertAlmostEqual(model.intercept, -0.5, delta=0.08)
        self.assertLess(model.n_iter, 15)

    def test_gradient_is_zero_at_the_answer(self):
        X, y, _ = self._data(n=2000, seed=1)
        lam = 0.05
        model = NewtonLogistic(lam=lam).fit(X, y)
        err = model.predict_proba(X) - y
        np.testing.assert_allclose(X.T @ err / len(y) + lam * model.coef, 0.0, atol=1e-8)
        # the intercept has no penalty, so its own gradient is just the mean error
        self.assertAlmostEqual(err.mean(), 0.0, places=8)

    def test_huge_penalty_leaves_only_the_base_rate(self):
        X, y, _ = self._data(n=2000, seed=2)
        model = NewtonLogistic(lam=1e6).fit(X, y)
        np.testing.assert_allclose(model.coef, 0.0, atol=1e-4)
        self.assertAlmostEqual(sigmoid(model.intercept), y.mean(), places=4)

    def test_bad_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            NewtonLogistic(lam=-1)
        with self.assertRaises(ValueError):
            NewtonLogistic().fit(np.zeros((3, 2)), [0, 1, 2])
        with self.assertRaises(RuntimeError):
            NewtonLogistic().predict_proba(np.zeros((1, 2)))


def _trial(variant, lam, cols, loss):
    return Trial(variant, lam, cols, loss, 0.8, 0.5, None, None)


class SelectionTests(unittest.TestCase):
    def test_ties_go_to_fewer_columns_then_more_penalty(self):
        trials = [
            _trial("big", 1e-4, 30, 0.3700),
            _trial("small", 1e-4, 22, 0.3708),
            _trial("small", 1e-2, 22, 0.3709),
            _trial("tiny", 1e-1, 10, 0.3800),
        ]
        picked = pick_simplest(trials, tol=0.001)
        self.assertEqual((picked.variant, picked.lam), ("small", 1e-2))
        self.assertEqual(pick_simplest(trials, tol=0.0).variant, "big")

    def test_paired_bootstrap_signs(self):
        rng = np.random.default_rng(3)
        y = (rng.random(400) < 0.3).astype(int)
        noise = rng.random(400)
        same = paired_bootstrap(y, noise, noise, n_boot=200)
        self.assertEqual(same["lo"], 0.0)
        self.assertEqual(same["hi"], 0.0)
        better = paired_bootstrap(y, y + 0.1 * noise, noise, n_boot=200)
        self.assertGreater(better["lo"], 0.0)


@unittest.skipUnless(CACHED, "Telco CSV not cached; run the Day 81 main.py once")
class RealDataTests(unittest.TestCase):
    def test_picked_model_beats_the_day_81_bar(self):
        from baseline_engine import RateTable, add_tenure_band, recall_at_top
        from churn_data import load_clean
        from select_engine import run_grid
        from split_engine import load_split

        parts = load_split(load_clean(DATA_DIR), SPLIT_FILE)
        train, val = parts["train"], parts["validation"]
        picked = pick_simplest(run_grid(train, val))
        base = RateTable(["Contract", "tenure_band", "InternetService"]).fit(add_tenure_band(train))
        bar = recall_at_top(val[TARGET], base.predict(add_tenure_band(val)))
        self.assertEqual(picked.variant, "collapsed, + log tenure")
        self.assertGreater(picked.val_top20, bar)


class PlotSmokeTests(unittest.TestCase):
    def test_this_days_visualizer_is_the_one_imported(self):
        import visualizer

        self.assertTrue(hasattr(visualizer, "ModelPlots"))

    def test_plots_write_files(self):
        from visualizer import ModelPlots

        rng = np.random.default_rng(4)
        y = (rng.random(60) < 0.3).astype(int)
        p = np.clip(y * 0.5 + rng.random(60) * 0.5, 0, 1)
        with tempfile.TemporaryDirectory() as tmp:
            plots = ModelPlots(tmp)
            paths = [
                plots.calibration(y, p, n_bins=5),
                plots.coefficients(["a", "b", "c"], [0.5, -0.2, 0.1]),
                plots.bootstrap_diff(rng.normal(0.05, 0.02, 300), 0.01, 0.09),
            ]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
