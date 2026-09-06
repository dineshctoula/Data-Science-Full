"""Tests for Day 37 visualizer input validation and image output."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from visualizer import DeterminantVisualizer


class DeterminantVisualizerTests(unittest.TestCase):
    def test_visualizations_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = DeterminantVisualizer(temporary_directory)
            area_path = visualizer.plot_area_scaling({"Identity": np.eye(2)})
            basis_path = visualizer.plot_basis_transformation(np.array([[2.0, 0.0], [0.0, 1.0]]))
            self.assertTrue(Path(area_path).is_file())
            self.assertTrue(Path(basis_path).is_file())

    def test_non_two_by_two_matrix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            visualizer = DeterminantVisualizer(temporary_directory)
            with self.assertRaisesRegex(ValueError, "2 × 2"):
                visualizer.plot_basis_transformation(np.ones((3, 3)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
