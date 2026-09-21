"""Tests for Day 47 ROC visualization output and validation."""

import tempfile
import unittest
from pathlib import Path

from roc_engine import RocEngine
from visualizer import RocVisualizer


class RocVisualizerTests(unittest.TestCase):
    def test_visualizations_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = RocVisualizer(temporary_directory)
            labels, scores = RocEngine.generate_scored_labels(n=80, seed=3)
            curve = RocEngine.build_curves(labels, scores)
            best = curve.best_youden_threshold()
            poor_labels, poor_scores = RocEngine.generate_poor_ranking(n=80, seed=3)
            poor = RocEngine.build_curves(poor_labels, poor_scores)

            roc_path = visualizer.plot_roc_curve(curve, best)
            pr_path = visualizer.plot_precision_recall(curve, best)
            tradeoff_path = visualizer.plot_threshold_tradeoff(curve)
            auc_path = visualizer.plot_auc_comparison(
                {"Useful ranking": curve.auc_roc, "Random scores": poor.auc_roc}
            )
            self.assertTrue(Path(roc_path).is_file())
            self.assertTrue(Path(pr_path).is_file())
            self.assertTrue(Path(tradeoff_path).is_file())
            self.assertTrue(Path(auc_path).is_file())

    def test_invalid_inputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = RocVisualizer(temporary_directory)
            with self.assertRaisesRegex(ValueError, "CurveResult"):
                visualizer.plot_roc_curve("bad")
            with self.assertRaisesRegex(ValueError, "At least one"):
                visualizer.plot_auc_comparison({})


if __name__ == "__main__":
    unittest.main(verbosity=2)
