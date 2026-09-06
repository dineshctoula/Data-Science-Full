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
        if not matrices:
            raise ValueError("Provide at least one 2 × 2 matrix to visualize.")
        unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
        figure, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4.5))
        axes = np.atleast_1d(axes)

        for axis, (name, matrix) in zip(axes, matrices.items()):
            matrix = np.asarray(matrix, dtype=float)
            if matrix.shape != (2, 2):
                raise ValueError("Area scaling is defined here for 2 × 2 matrices only.")
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

    def plot_basis_transformation(self, matrix: np.ndarray) -> Path:
        """Compare the standard basis with its image under a 2 × 2 matrix."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (2, 2):
            raise ValueError("Basis transformation requires a 2 × 2 matrix.")

        basis = np.eye(2)
        transformed_basis = matrix @ basis
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
        labels = ("e₁", "e₂")

        for axis, vectors, title, color in (
            (axes[0], basis, "Standard basis", "#4C78A8"),
            (axes[1], transformed_basis, "Transformed basis: A eᵢ", "#54A24B"),
        ):
            for vector, label in zip(vectors.T, labels):
                axis.quiver(0, 0, vector[0], vector[1], angles="xy", scale_units="xy", scale=1,
                            color=color, width=0.012)
                axis.annotate(label, vector, xytext=(6, 5), textcoords="offset points", color=color)
            axis.axhline(0, color="black", linewidth=0.7)
            axis.axvline(0, color="black", linewidth=0.7)
            axis.grid(alpha=0.25)
            axis.set_aspect("equal", adjustable="box")
            axis.set_title(title)
            axis.set_xlabel("x")

        # One shared scale makes changes in vector direction and magnitude comparable.
        limit = max(1.5, float(np.max(np.abs(transformed_basis))) + 0.5)
        for axis in axes:
            axis.set_xlim(-limit, limit)
            axis.set_ylim(-limit, limit)
        axes[0].set_ylabel("y")
        figure.suptitle(f"A = {np.array2string(matrix, precision=2)}", fontsize=14, fontweight="bold")
        figure.tight_layout()
        path = self.output_dir / "basis_transformation.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        return path
