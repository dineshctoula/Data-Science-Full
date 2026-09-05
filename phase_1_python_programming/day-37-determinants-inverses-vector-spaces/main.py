"""Day 37 — Determinants, Matrix Inverses, and Vector Spaces."""

import numpy as np

from matrix_space_engine import MatrixSpaceEngine
from visualizer import DeterminantVisualizer


def format_basis(name: str, basis: np.ndarray) -> None:
    print(f"  {name} dimension: {basis.shape[1]}")
    print(f"  {name} basis:\n{np.round(basis, 4)}")


def run_pipeline() -> None:
    print("=" * 80)
    print("DAY 37: DETERMINANTS, MATRIX INVERSES & VECTOR SPACES")
    print("=" * 80)

    matrix = np.array([[4.0, 7.0], [2.0, 6.0]])
    print("\n1. Determinant, cofactors, and inverse")
    print(f"Matrix A:\n{matrix}")
    cofactor_determinant = MatrixSpaceEngine.determinant_by_cofactor(matrix)
    report = MatrixSpaceEngine.invertibility_report(matrix)
    inverse = MatrixSpaceEngine.inverse_by_adjugate(matrix)
    print(f"det(A), cofactor expansion: {cofactor_determinant:.2f}")
    print(f"det(A), NumPy check:       {report['determinant']:.2f}")
    print(f"A is invertible: {report['is_invertible']} | condition number: {report['condition_number']:.2f}")
    print(f"A⁻¹ via adjugate:\n{np.round(inverse, 4)}")
    print(f"Verification A @ A⁻¹:\n{np.round(matrix @ inverse, 4)}")

    print("\n2. Vector spaces from a rank-deficient matrix")
    dependent_matrix = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [1.0, 1.0, 1.0],
    ])
    spaces = MatrixSpaceEngine.analyze_subspaces(dependent_matrix)
    print(f"B:\n{dependent_matrix}\n  rank(B): {spaces.rank}")
    format_basis("Column space", spaces.column_space_basis)
    format_basis("Row space", spaces.row_space_basis)
    format_basis("Null space", spaces.null_space_basis)
    print(f"  Null-space check ||B @ N||: {np.linalg.norm(dependent_matrix @ spaces.null_space_basis):.2e}")
    print("  Rank-nullity check: rank + nullity = "
          f"{spaces.rank} + {spaces.null_space_basis.shape[1]} = {dependent_matrix.shape[1]}")

    print("\n3. Exact vs. least-squares solutions")
    target = np.array([6.0, 12.0, 4.0])
    solution, residual = MatrixSpaceEngine.solve_or_least_squares(dependent_matrix, target)
    print(f"  Minimum-norm least-squares solution: {np.round(solution, 4)}")
    print(f"  Residual norm ||Bx - b||: {residual:.2e}")

    print("\n4. Generating determinant geometry visualization")
    image = DeterminantVisualizer().plot_area_scaling({
        "Expansion": np.array([[2.0, 0.0], [0.0, 1.5]]),
        "Shear (area preserved)": np.array([[1.0, 1.5], [0.0, 1.0]]),
        "Singular flattening": np.array([[1.0, 1.0], [2.0, 2.0]]),
    })
    print(f"  Saved: {image}")
    print("\nDay 37 pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
