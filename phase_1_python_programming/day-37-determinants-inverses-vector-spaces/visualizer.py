"""Visual explanations for determinant area scaling and invertibility."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class DeterminantVisualizer:
    def __init__(self, output_dir: str = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_area_scaling(self, matrices: dict[str, np.ndarray]) -> Path:
        """Plot how matrices transform the unit square; |det(A)| scales its area."""
        unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
        figure, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4.5))
        axes = np.atleast_1d(axes)

        for axis, (name, matrix) in zip(axes, matrices.items()):
            transformed = unit_square @ matrix.T
            determinant = np.linalg.det(matrix)
            axis.fill(unit_square[:, 0], unit_square[:, 1], alpha=0.22, color="#4C78A8", label="Unit square")
            axis.plot(unit_square[:, 0], unit_square[:, 1], color="#4C78A8")
            axis.fill(transformed[:, 0], transformed[:, 1], alpha=0.30, color="#F58518", label="Transformed")
            axis.plot(transformed[:, 0], transformed[:, 1], color="#F58518", linewidth=2)
            axis.axhline(0, color="black", linewidth=0.7)
            axis.axvline(0, color="black", linewidth=0.7)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.25)
            axis.set_title(f"{name}\ndet(A) = {determinant:.2f}, area scale = {abs(determinant):.2f}")
            axis.legend(loc="upper left")

        figure.suptitle("Determinants describe signed area scaling", fontsize=14, fontweight="bold")
        figure.tight_layout()
        path = self.output_dir / "determinant_area_scaling.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path
