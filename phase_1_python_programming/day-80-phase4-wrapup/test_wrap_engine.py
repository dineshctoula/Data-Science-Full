"""Tests for the Day 80 wrap-up — the habits should hold, not just the numbers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from wrap_engine import (
    ChurnLogistic,
    DecayTrial,
    Prep,
    log_loss,
    make_churn_table,
    permutation_importance,
    pick_decay,
    pick_threshold,
    roc_auc,
    scores_at,
    search_decay,
    split_three,
)


def _prepared(seed=80, n=900):
    table, y = make_churn_table(n=n, seed=seed)
    (tr, y_tr), (va, y_va), (te, y_te) = split_three(table, y, seed=seed)
    prep = Prep().fit(tr)
    return prep, (tr, y_tr), (va, y_va), (te, y_te)


class SplitTests(unittest.TestCase):
    def test_slices_cover_every_row_once(self):
        table, y = make_churn_table(n=200, seed=1)
        # tag each row with its position so the slices can be traced back
        table["tenure"] = np.arange(200, dtype=float)
        parts = split_three(table, y, seed=1)
        seen = np.concatenate([part["tenure"] for part, _ in parts])
        self.assertEqual([len(labels) for _, labels in parts], [120, 40, 40])
        np.testing.assert_array_equal(np.sort(seen), np.arange(200))

    def test_bad_fracs_are_rejected(self):
        table, y = make_churn_table(n=50, seed=2)
        for fracs in [(0.5, 0.5), (0.6, 0.3, 0.3), (0.8, 0.2, 0.0)]:
            with self.assertRaises(ValueError):
                split_three(table, y, fracs=fracs)


class PrepTests(unittest.TestCase):
    def test_statistics_come_from_the_training_rows_only(self):
        prep, (tr, _), (va, _), _ = _prepared()
        before = dict(prep.medians), prep.mean.copy()
        # wreck the validation rows; nothing fitted should notice
        va = dict(va)
        va["spend"] = np.full(len(va["spend"]), 1e6)
        prep.transform(va)
        self.assertEqual(prep.medians, before[0])
        np.testing.assert_array_equal(prep.mean, before[1])
        self.assertAlmostEqual(prep.medians["spend"], float(np.nanmedian(tr["spend"])))

    def test_training_columns_come_out_standardized(self):
        prep, (tr, _), _, _ = _prepared()
        X = prep.transform(tr)
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(X.std(axis=0), 1.0, atol=1e-9)

    def test_unseen_plan_gets_an_empty_one_hot_block(self):
        prep, _, (va, _), _ = _prepared()
        va = dict(va)
        va["plan"] = np.array(["trial"] * len(va["plan"]))
        raw = prep._unscaled(va)
        plan_cols = [j for j, name in enumerate(prep.names) if name.startswith("plan=")]
        np.testing.assert_array_equal(raw[:, plan_cols], 0.0)

    def test_blank_spend_is_filled_and_flagged(self):
        prep, (tr, _), _, _ = _prepared()
        raw = prep._unscaled(tr)
        blank = np.isnan(tr["spend"])
        self.assertTrue(blank.any())
        spend_col = prep.names.index("spend")
        flag_col = prep.names.index("spend_missing")
        np.testing.assert_allclose(raw[blank, spend_col], prep.medians["spend"])
        np.testing.assert_array_equal(raw[:, flag_col], blank.astype(float))

    def test_transform_before_fit_is_rejected(self):
        table, _ = make_churn_table(n=20, seed=3)
        with self.assertRaises(RuntimeError):
            Prep().transform(table)


class MetricTests(unittest.TestCase):
    def test_scores_on_a_hand_example(self):
        y = [1, 1, 0, 0, 1]
        p = [0.9, 0.2, 0.7, 0.1, 0.6]
        s = scores_at(y, p, 0.5)
        # tp=2 (0.9, 0.6), fp=1 (0.7), fn=1 (0.2)
        self.assertAlmostEqual(s["precision"], 2 / 3)
        self.assertAlmostEqual(s["recall"], 2 / 3)
        self.assertAlmostEqual(s["accuracy"], 3 / 5)

    def test_auc_extremes(self):
        self.assertEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)

    def test_log_loss_of_a_coin_flip(self):
        self.assertAlmostEqual(log_loss([0, 1], [0.5, 0.5]), np.log(2))


class ModelTests(unittest.TestCase):
    def test_bad_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            ChurnLogistic(decay=-0.1)
        with self.assertRaises(ValueError):
            ChurnLogistic(batch=0)

    def test_predict_before_fit_is_rejected(self):
        with self.assertRaises(RuntimeError):
            ChurnLogistic().predict_proba(np.zeros((2, 3)))

    def test_kept_epoch_is_the_best_validation_epoch(self):
        prep, (tr, y_tr), (va, y_va), _ = _prepared()
        X_tr, X_va = prep.transform(tr), prep.transform(va)
        model = ChurnLogistic(seed=1).fit(X_tr, y_tr, X_va, y_va)
        h = model.history
        self.assertEqual(h.best_epoch, int(np.argmin(h.val_loss)) + 1)
        # the stored weights should reproduce the kept epoch's loss
        self.assertAlmostEqual(log_loss(y_va, model.predict_proba(X_va)), h.val_loss[h.best_epoch - 1])
        # and training stopped once patience ran out
        self.assertLessEqual(len(h.val_loss), h.best_epoch + model.patience)

    def test_more_decay_means_smaller_slopes(self):
        prep, (tr, y_tr), (va, y_va), _ = _prepared()
        X_tr, X_va = prep.transform(tr), prep.transform(va)
        light = ChurnLogistic(decay=0.0, max_epochs=60, patience=60).fit(X_tr, y_tr, X_va, y_va)
        heavy = ChurnLogistic(decay=0.5, max_epochs=60, patience=60).fit(X_tr, y_tr, X_va, y_va)
        self.assertLess(np.linalg.norm(heavy.w), np.linalg.norm(light.w))


class WorkflowTests(unittest.TestCase):
    def test_pick_decay_reads_validation_and_breaks_ties_low(self):
        trials = [DecayTrial(0.1, 5, 0.4, 0.50, None), DecayTrial(0.0, 5, 0.3, 0.50, None), DecayTrial(1.0, 5, 0.6, 0.55, None)]
        self.assertEqual(pick_decay(trials).decay, 0.0)

    def test_pick_threshold_on_a_hand_example(self):
        # both leavers sit at 0.3, so any cut-off above 0.3 misses them
        y = [1, 1, 0, 0, 0]
        p = [0.3, 0.3, 0.1, 0.1, 0.6]
        threshold, f1 = pick_threshold(y, p)
        self.assertLessEqual(threshold, 0.3)
        self.assertGreater(threshold, 0.1)
        self.assertAlmostEqual(f1, 0.8)

    def test_model_beats_the_train_rate_and_region_matters_least(self):
        prep, (tr, y_tr), (va, y_va), (te, y_te) = _prepared()
        X_tr, X_va, X_te = prep.transform(tr), prep.transform(va), prep.transform(te)
        picked = pick_decay(search_decay(X_tr, y_tr, X_va, y_va, [0.0, 0.03]))
        model_loss = log_loss(y_te, picked.model.predict_proba(X_te))
        base_loss = log_loss(y_te, np.full(len(y_te), y_tr.mean()))
        self.assertLess(model_loss, base_loss - 0.05)
        _, rises = permutation_importance(picked.model, prep, va, y_va, n_repeats=5)
        self.assertEqual(min(rises, key=rises.get), "region")


class PlotSmokeTests(unittest.TestCase):
    def test_plots_write_files(self):
        from visualizer import WrapPlots

        prep, (tr, y_tr), (va, y_va), _ = _prepared(n=200)
        X_tr, X_va = prep.transform(tr), prep.transform(va)
        trials = search_decay(X_tr, y_tr, X_va, y_va, [0.0, 0.1], max_epochs=20)
        picked = pick_decay(trials)
        with tempfile.TemporaryDirectory() as tmp:
            plots = WrapPlots(tmp)
            paths = [
                plots.loss_curves(picked.model.history),
                plots.decay_grid(trials, picked),
                plots.importance({"a": 0.02, "b": -0.001}),
            ]
            for path in paths:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
