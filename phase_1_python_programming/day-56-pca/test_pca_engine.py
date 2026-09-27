"""Tests for Day 56 PCA."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pca_engine import PCA, make_noisy_signal, make_stretched_cloud


class PCATests(unittest.TestCase):
    def test_full_rank_reconstructs(self):
        X = make_stretched_cloud(n=80, seed=1)
        model = PCA(n_components=2).fit(X)
        mse = model.reconstruction_mse(X)
        self.assertLess(mse, 1e-8)
        # ratios should add to ~1
        self.assertAlmostEqual(float(model.explained_variance_ratio_.sum()), 1.0, places=5)

    def test_first_component_dominates_stretched_cloud(self):
        X = make_stretched_cloud(n=150, seed=2)
        model = PCA(n_components=2).fit(X)
        # long axis should eat most of the variance
        self.assertGreater(model.explained_variance_ratio_[0], 0.85)

    def test_dropping_dims_raises_error(self):
        X = make_noisy_signal(n=100, p=6, seed=3)
        full = PCA(n_components=6).fit(X)
        thin = PCA(n_components=2).fit(X)
        self.assertLess(full.reconstruction_mse(X), 1e-8)
        self.assertGreater(thin.reconstruction_mse(X), full.reconstruction_mse(X))
        # two components should still keep the bulk of the signal
        self.assertGreater(float(thin.explained_variance_ratio_[:2].sum()), 0.8)

    def test_transform_shape(self):
        X = make_noisy_signal(n=40, p=5, seed=4)
        Z = PCA(n_components=2).fit(X).transform(X)
        self.assertEqual(Z.shape, (40, 2))

    def test_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "n_components"):
            PCA(n_components=0)
        model = PCA(n_components=2)
        with self.assertRaisesRegex(RuntimeError, "fit"):
            model.transform([[1.0, 2.0]])
        X = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "bigger"):
            PCA(n_components=4).fit(X)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import PCAPlots

        X = make_stretched_cloud(n=60, seed=5)
        model = PCA(n_components=2).fit(X)
        hat = model.inverse_transform(model.transform(X, n_components=1))
        with tempfile.TemporaryDirectory() as tmp:
            plots = PCAPlots(tmp)
            paths = [
                plots.stretched_cloud(model, X),
                plots.scree(model.explained_variance_ratio_),
                plots.recon_scatter(X, hat),
            ]
            for p in paths:
                self.assertTrue(Path(p).exists())


if __name__ == "__main__":
    unittest.main()
