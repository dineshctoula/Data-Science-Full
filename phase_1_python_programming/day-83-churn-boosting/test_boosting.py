"""Tests for Day 83 — the binner, the Newton trees, the stump reader, and the CV pick."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from project_paths import DATA_DIR
from baseline_engine import log_loss
from boost_engine import Binner, BoostedTrees, build_tree, sigmoid
from churn_data import FILENAME
from cv_engine import Config, Pick, pick_config, stratified_folds

CACHED = (DATA_DIR / FILENAME).exists()


class BinnerTests(unittest.TestCase):
    def test_few_values_get_one_cut_between_each(self):
        X = np.array([[0.0], [1.0], [1.0], [0.0]])
        b = Binner().fit(X)
        np.testing.assert_array_equal(b.cuts[0], [0.0])
        np.testing.assert_array_equal(b.transform(X)[:, 0], [0, 1, 1, 0])

    def test_code_k_means_at_most_the_kth_cut(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(500, 1))
        b = Binner(max_bins=8).fit(x)
        codes = b.transform(x)[:, 0]
        self.assertLessEqual(len(b.cuts[0]), 7)
        for k, cut in enumerate(b.cuts[0]):
            np.testing.assert_array_equal(codes <= k, x[:, 0] <= cut)

    def test_misuse_is_rejected(self):
        with self.assertRaises(RuntimeError):
            Binner().transform(np.zeros((2, 1)))
        b = Binner().fit(np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            b.transform(np.zeros((3, 3)))


class TreeTests(unittest.TestCase):
    def test_leaf_value_is_the_newton_step(self):
        codes = np.zeros((4, 1), dtype=np.int32)
        g = np.array([0.5, 0.5, -0.1, 0.2])
        h = np.array([0.25, 0.25, 0.25, 0.25])
        tree = build_tree(codes, g, h, n_bins=2, max_depth=0, lam=1.0)
        self.assertAlmostEqual(tree.value[0], -g.sum() / (h.sum() + 1.0))

    def test_split_lands_on_the_step(self):
        x = np.repeat(np.arange(10), 20).reshape(-1, 1).astype(float)
        y = (x[:, 0] >= 6).astype(float)
        codes = Binner().fit(x).transform(x)
        p = np.full(len(y), y.mean())
        tree = build_tree(codes, p - y, p * (1 - p), n_bins=10, max_depth=1)
        # code 5 is "x <= 5", so the step between 5 and 6 is found exactly
        self.assertEqual((tree.feature[0], tree.code[0]), (0, 5))

    def test_min_child_weight_blocks_tiny_leaves(self):
        x = np.arange(10, dtype=float).reshape(-1, 1)
        y = np.zeros(10)
        y[0] = 1.0
        codes = Binner().fit(x).transform(x)
        p = np.full(10, 0.1)
        h = p * (1 - p)
        # the only useful split isolates one row whose curvature is 0.09
        tree = build_tree(codes, p - y, h, n_bins=10, max_depth=1, min_child_weight=0.5)
        self.assertEqual(tree.feature[0], -1)


class BoosterTests(unittest.TestCase):
    def _xor(self, n=2000, seed=1):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, size=(n, 2))
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
        return X, y

    def test_depth_two_learns_xor_and_stumps_cannot(self):
        # XOR has no one-column signal at all; only a tree that combines columns can see it.
        # Each tree's first split is close to a noise split, so depth 2 needs a few hundred trees.
        X, y = self._xor()
        stumps = BoostedTrees(n_trees=300, max_depth=1).fit(X, y)
        pairs = BoostedTrees(n_trees=300, max_depth=2).fit(X, y)
        self.assertGreater(log_loss(y, stumps.predict_proba(X)), 0.6)
        self.assertLess(log_loss(y, pairs.predict_proba(X)), 0.1)

    def test_start_is_the_base_rate_and_stages_end_at_the_full_model(self):
        X, y = self._xor(n=300, seed=2)
        model = BoostedTrees(n_trees=20, max_depth=2).fit(X, y)
        self.assertAlmostEqual(sigmoid(model.base), y.mean())
        np.testing.assert_allclose(sigmoid(model.staged_decision(X)[-1]), model.predict_proba(X))
        np.testing.assert_allclose(sigmoid(model.staged_decision(X)[4]), model.predict_proba(X, n_trees=5))

    def test_stump_parts_add_back_to_the_model(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(800, 3))
        y = (rng.random(800) < sigmoid(X[:, 0] - X[:, 1] ** 2)).astype(float)
        model = BoostedTrees(n_trees=60, max_depth=1).fit(X, y)
        parts, shift = model.additive_parts(X)
        np.testing.assert_allclose(model.base + shift + parts.sum(axis=1), model.staged_decision(X)[-1])
        with self.assertRaises(ValueError):
            BoostedTrees(n_trees=2, max_depth=2).fit(X, y).additive_parts(X)

    def test_gain_goes_to_the_column_that_matters(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(1000, 3))
        y = (X[:, 1] > 0).astype(float)
        gains = BoostedTrees(n_trees=20, max_depth=1).fit(X, y).gain_by_column(3)
        self.assertEqual(int(np.argmax(gains)), 1)
        self.assertGreater(gains[1], 10 * (gains[0] + gains[2]))

    def test_bad_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            BoostedTrees(learning_rate=0)
        with self.assertRaises(ValueError):
            BoostedTrees(max_depth=0)
        with self.assertRaises(ValueError):
            BoostedTrees().fit(np.zeros((3, 1)), [0, 1, 2])


class CVTests(unittest.TestCase):
    def test_folds_partition_the_rows_and_keep_the_rate(self):
        y = np.array([1] * 30 + [0] * 70)
        folds = stratified_folds(y, k=5, seed=0)
        np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(100))
        for f in folds:
            self.assertEqual(int(y[f].sum()), 6)

    def test_pick_prefers_shallow_then_few_trees_inside_the_tie_band(self):
        deep, shallow = Config(3, 0.1), Config(1, 0.1)
        curves = {
            # deep reaches 0.400 at tree 3; shallow only gets within 0.001 at tree 5
            deep: np.array([[0.50, 0.45, 0.400, 0.41, 0.42, 0.43]]),
            shallow: np.array([[0.60, 0.50, 0.45, 0.41, 0.4008, 0.4005]]),
        }
        pick = pick_config(curves, tol=0.001)
        self.assertEqual((pick.config, pick.n_trees), (shallow, 5))
        strict = pick_config(curves, tol=0.0)
        self.assertEqual((strict.config, strict.n_trees), (deep, 3))
        self.assertIsInstance(strict, Pick)


class ImportTests(unittest.TestCase):
    def test_this_days_visualizer_wins_and_day_82_is_reachable(self):
        import features
        import visualizer

        self.assertTrue(hasattr(visualizer, "BoostPlots"))
        self.assertIn("collapsed, + log tenure", features.VARIANTS)


@unittest.skipUnless(CACHED, "Telco CSV not cached; run the Day 81 main.py once")
class RealDataTests(unittest.TestCase):
    def test_a_short_stump_run_beats_the_day_81_lookup(self):
        from baseline_engine import RateTable, add_tenure_band, recall_at_top
        from churn_data import TARGET, load_clean
        from cv_engine import encode
        from project_paths import SPLIT_FILE
        from split_engine import load_split

        parts = load_split(load_clean(DATA_DIR), SPLIT_FILE)
        train, val = parts["train"], parts["validation"]
        X_tr, X_va, _ = encode(train, val)
        p = BoostedTrees(n_trees=150, max_depth=1).fit(X_tr, train[TARGET]).predict_proba(X_va)
        base = RateTable(["Contract", "tenure_band", "InternetService"]).fit(add_tenure_band(train))
        self.assertGreater(recall_at_top(val[TARGET], p), recall_at_top(val[TARGET], base.predict(add_tenure_band(val))))


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import BoostPlots

        cfg = Config(1, 0.1)
        curves = {cfg: np.array([[0.5, 0.45, 0.44], [0.52, 0.46, 0.45]])}
        pick = Pick(cfg, 3, 0.445, 0.445)
        xs = np.linspace(0, 10, 20)
        with tempfile.TemporaryDirectory() as tmp:
            plots = BoostPlots(tmp)
            paths = [
                plots.cv_curves(curves, pick, np.array([0.43, 0.44])),
                plots.shapes([("x", xs, {"stumps": np.sign(xs - 5), "logistic": xs / 5})]),
                plots.gain_importance(["a", "b"], [3.0, 1.0]),
            ]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
