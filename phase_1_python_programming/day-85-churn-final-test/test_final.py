"""Tests for Day 85 — the gate, the recipe, and the score card.

None of these tests read the test slice. Everything that needs real rows
uses train and validation, which is the point of the gate.
"""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from project_paths import DATA_DIR, HERE
import final_recipe as R
from churn_data import FILENAME, TARGET
from holdout_gate import HoldoutAlreadyOpened, fingerprint, open_test, read_lock

CACHED = (DATA_DIR / FILENAME).exists()


class GateTests(unittest.TestCase):
    def test_first_open_writes_the_lock_and_reruns_are_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "x.lock"
            parts = {"test": [1, 2, 3]}
            when = datetime(2026, 10, 26, 14, 0, tzinfo=timezone.utc)
            rows, first = open_test(parts, lock, "abc", now=when)
            self.assertTrue(first)
            self.assertEqual(rows, [1, 2, 3])
            self.assertEqual(read_lock(lock), {"fingerprint": "abc", "opened": "2026-10-26 14:00 UTC", "rows": "3"})
            _, again = open_test(parts, lock, "abc")
            self.assertFalse(again)

    def test_a_changed_recipe_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "x.lock"
            open_test({"test": []}, lock, "abc")
            with self.assertRaises(HoldoutAlreadyOpened):
                open_test({"test": []}, lock, "abd")

    def test_fingerprint_sees_content_and_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.py", Path(tmp) / "b.py"
            a.write_text("x = 1\n")
            b.write_text("y = 2\n")
            base = fingerprint([a, b])
            self.assertEqual(base, fingerprint([a, b]))
            self.assertNotEqual(base, fingerprint([b, a]))
            a.write_text("x = 2\n")
            self.assertNotEqual(base, fingerprint([a, b]))

    def test_a_lock_without_a_fingerprint_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "x.lock"
            lock.write_text("opened: someday\n")
            with self.assertRaises(ValueError):
                read_lock(lock)

    def test_main_reaches_test_only_through_the_gate(self):
        # a plain text check: main.py must never index the test slice itself
        source = (HERE / "main.py").read_text()
        self.assertNotIn('parts["test"]', source)
        self.assertNotIn("parts['test']", source)
        self.assertIn("open_test(parts", source)


class RecipeTests(unittest.TestCase):
    def test_recipe_names_real_choices(self):
        from features import VARIANTS

        self.assertIn(R.VARIANT, VARIANTS)
        self.assertEqual(R.FIT_ON, ("train", "validation"))
        self.assertNotIn("test", R.FIT_ON)
        self.assertEqual(R.ECONOMICS.cost, 30.0)


@unittest.skipUnless(CACHED, "Telco CSV not cached; run the Day 81 main.py once")
class EngineOnValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from churn_data import load_clean
        from project_paths import SPLIT_FILE
        from split_engine import load_split

        parts = load_split(load_clean(DATA_DIR), SPLIT_FILE)
        # deliberately keep only train and validation
        cls.train, cls.val = parts["train"], parts["validation"]

    def test_champion_refit_reproduces_day_82(self):
        from baseline_engine import log_loss, recall_at_top
        from final_engine import fit_champion

        p = fit_champion(self.train, R.VARIANT, R.LAM).predict(self.val)
        self.assertAlmostEqual(log_loss(self.val[TARGET], p), 0.3720, places=4)
        self.assertAlmostEqual(recall_at_top(self.val[TARGET], p), 0.554, places=3)

    def test_score_card_intervals_bracket_the_point(self):
        from final_engine import fit_champion, score_card

        p = fit_champion(self.train, R.VARIANT, R.LAM).predict(self.val)
        card = score_card(self.val[TARGET], p, n_boot=200, seed=1)
        for key in ("top20_recall", "auc", "log_loss"):
            point, lo, hi = card[key]
            self.assertLessEqual(lo, point)
            self.assertLessEqual(point, hi)

    def test_policy_card_matches_day_84(self):
        from final_engine import fit_champion, policy_card

        p = fit_champion(self.train, R.VARIANT, R.LAM).predict(self.val)
        pc = policy_card(p, self.val[TARGET].to_numpy(), self.val, R.ECONOMICS, R.TOP_FRAC, n_boot=200, seed=1)
        self.assertEqual(pc["rule"]["calls"], 406)
        self.assertAlmostEqual(pc["rule"]["profit"], 16244.175, places=2)
        self.assertLessEqual(pc["rule"]["profit"], pc["ceiling"])


class PlotSmokeTests(unittest.TestCase):
    def test_this_days_visualizer_is_imported_and_writes(self):
        import visualizer

        card = {"top20_recall": (0.5, 0.45, 0.55), "auc": (0.8, 0.78, 0.82), "log_loss": (0.4, 0.38, 0.42)}
        with tempfile.TemporaryDirectory() as tmp:
            path = visualizer.FinalPlots(tmp).val_vs_test({"a": card, "b": card})
            self.assertTrue(Path(path).exists())
            self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
