"""Tests for Day 63 encodings — order, unknowns, and the one-hot shape."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from encode_engine import (
    OneHotEncoder,
    OrdinalEncoder,
    compare_encodings,
    make_plan_data,
)


class OrdinalTests(unittest.TestCase):
    def test_respects_given_order(self):
        enc = OrdinalEncoder(order=("basic", "plus", "pro"))
        codes = enc.fit_transform(["pro", "basic", "plus", "basic"])
        self.assertTrue(np.array_equal(codes, [2, 0, 1, 0]))

    def test_unseen_is_minus_one(self):
        enc = OrdinalEncoder(order=("basic", "plus", "pro")).fit(["basic", "plus"])
        out = enc.transform(["pro", "nope", "basic"])
        # "pro" was in the order but never in fit(), so it isn't a column
        self.assertEqual(out.tolist(), [-1, -1, 0])

    def test_default_order_is_sorted(self):
        enc = OrdinalEncoder().fit(["b", "a", "b"])
        self.assertEqual(enc.categories_, ("a", "b"))


class OneHotTests(unittest.TestCase):
    def test_known_rows_sum_to_one(self):
        enc = OneHotEncoder().fit(["email", "ads", "email"])
        Z = enc.transform(["ads", "email"])
        self.assertEqual(Z.shape, (2, 2))
        self.assertTrue(np.allclose(Z.sum(axis=1), 1))

    def test_unknown_row_is_zeros(self):
        enc = OneHotEncoder().fit(["email", "ads"])
        Z = enc.transform(["sms"])
        self.assertTrue(np.allclose(Z, 0))
        self.assertNotIn("sms", enc.categories_)


class CompareTests(unittest.TestCase):
    def test_flip_does_not_change_linear_mse(self):
        plan, channel, hours, score = make_plan_data(n=220, seed=7)
        rows = {r.name: r for r in compare_encodings(plan, channel, hours, score, seed=7)}
        # reversed integers are an affine rewrite, OLS absorbs that
        self.assertAlmostEqual(rows["ordinal plan"].mse, rows["flipped plan order"].mse, places=4)
        self.assertLess(rows["one-hot plan+channel"].mse, rows["ordinal plan"].mse)


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import EncodePlots

        plan, channel, hours, score = make_plan_data(n=40, seed=8)
        rows = compare_encodings(plan, channel, hours, score, seed=8)
        with tempfile.TemporaryDirectory() as tmp:
            plots = EncodePlots(tmp)
            paths = [
                plots.mean_bars(plan, score, title="by plan", filename="plan_means.png"),
                plots.mse_bars(rows),
            ]
            for p in paths:
                self.assertTrue(Path(p).exists())
                self.assertGreater(Path(p).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
