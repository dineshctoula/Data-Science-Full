"""
Vector and Matrix Analytics Engine
Day 36: Math & Statistics for Data Science — Linear Algebra Fundamentals

This module provides a production-ready, object-oriented engine for vector operations,
norm calculations, cosine similarity, orthogonal projections, Gram-Schmidt orthogonalization,
matrix transformations, rank determination, and linear system diagnostics.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class VectorEngine:
    """
    High-performance vector analytics engine supporting algebraic operations,
    vector norms, projections, angle computations, and Gram-Schmidt orthonormalization.
    """

    @staticmethod
    def add(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Element-wise addition of two vectors."""
        v1_arr = np.asarray(v1, dtype=np.float64)
        v2_arr = np.asarray(v2, dtype=np.float64)
        if v1_arr.shape != v2_arr.shape:
            raise ValueError(f"Vector dimensions mismatch: {v1_arr.shape} vs {v2_arr.shape}")
        return v1_arr + v2_arr

    @staticmethod
    def scalar_multiply(v: np.ndarray, scalar: float) -> np.ndarray:
        """Multiplies a vector by a scalar constant."""
        return np.asarray(v, dtype=np.float64) * scalar

    @staticmethod
    def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes the inner (dot) product of two vectors."""
        v1_arr = np.asarray(v1, dtype=np.float64)
        v2_arr = np.asarray(v2, dtype=np.float64)
        if v1_arr.shape != v2_arr.shape:
            raise ValueError(f"Vector dimensions mismatch: {v1_arr.shape} vs {v2_arr.shape}")
        return float(np.dot(v1_arr, v2_arr))

    @staticmethod
    def cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Computes the cross product of two 3D vectors."""
        v1_arr = np.asarray(v1, dtype=np.float64)
        v2_arr = np.asarray(v2, dtype=np.float64)
        if v1_arr.shape != (3,) or v2_arr.shape != (3,):
            raise ValueError(f"Cross product requires 3D vectors. Got shapes {v1_arr.shape} and {v2_arr.shape}")
        return np.cross(v1_arr, v2_arr)

    @staticmethod
    def compute_norms(v: np.ndarray) -> Dict[str, float]:
        """
        Computes L1 (Manhattan), L2 (Euclidean), and L-infinity (Maximum) norms.
        """
        v_arr = np.asarray(v, dtype=np.float64)
        return {
            "L1_norm": float(np.linalg.norm(v_arr, ord=1)),
            "L2_norm": float(np.linalg.norm(v_arr, ord=2)),
            "Linf_norm": float(np.linalg.norm(v_arr, ord=np.inf))
        }

    @staticmethod
    def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Computes cosine similarity between two vectors: cos(theta) = (v1 . v2) / (||v1|| * ||v2||).
        Returns a value between -1.0 and 1.0.
        """
        v1_arr = np.asarray(v1, dtype=np.float64)
        v2_arr = np.asarray(v2, dtype=np.float64)
        norm1 = np.linalg.norm(v1_arr)
        norm2 = np.linalg.norm(v2_arr)
        if norm1 == 0.0 or norm2 == 0.0:
            raise ValueError("Cosine similarity is undefined for zero vectors.")
        return float(np.dot(v1_arr, v2_arr) / (norm1 * norm2))

    @staticmethod
    def angle_between(v1: np.ndarray, v2: np.ndarray, degrees: bool = True) -> float:
        """Computes the angle between two vectors in degrees or radians."""
        cos_sim = VectorEngine.cosine_similarity(v1, v2)
        # Clip to prevent floating point inaccuracies beyond [-1, 1]
        cos_sim_clipped = np.clip(cos_sim, -1.0, 1.0)
        radians = float(np.arccos(cos_sim_clipped))
        return float(np.degrees(radians)) if degrees else radians

    @staticmethod
    def project(v: np.ndarray, onto_u: np.ndarray) -> np.ndarray:
        """
        Computes the orthogonal projection of vector v onto vector u.
        proj_u(v) = ((v . u) / ||u||^2) * u
        """
        v_arr = np.asarray(v, dtype=np.float64)
        u_arr = np.asarray(onto_u, dtype=np.float64)
        u_norm_sq = np.dot(u_arr, u_arr)
        if u_norm_sq == 0.0:
            raise ValueError("Cannot project onto a zero vector.")
        scale = np.dot(v_arr, u_arr) / u_norm_sq
        return scale * u_arr

    @staticmethod
    def gram_schmidt(vectors: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies Gram-Schmidt process to convert a set of linearly independent vectors
        into an orthonormal basis set.
        """
        orthonormal_basis = []
        for v in vectors:
            v_arr = np.asarray(v, dtype=np.float64)
            # Subtract projections onto previously computed basis vectors
            u = v_arr.copy()
            for b in orthonormal_basis:
                u -= np.dot(v_arr, b) * b
            
            norm_u = np.linalg.norm(u)
            if norm_u < 1e-10:
                # Vector is linearly dependent
                continue
            e = u / norm_u
            orthonormal_basis.append(e)
            
        return orthonormal_basis


class MatrixEngine:
    """
    Advanced matrix transformation and diagnostics engine.
    Supports geometric transformations, matrix rank, determinants, invertibility,
    trace, eigenvalues/eigenvectors, and linear equation system solving.
    """

    @staticmethod
    def multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiplication A @ B."""
        A_arr = np.asarray(A, dtype=np.float64)
        B_arr = np.asarray(B, dtype=np.float64)
        if A_arr.shape[1] != B_arr.shape[0]:
            raise ValueError(f"Shape mismatch for matrix multiplication: {A_arr.shape} @ {B_arr.shape}")
        return np.matmul(A_arr, B_arr)

    @staticmethod
    def get_diagnostics(A: np.ndarray) -> Dict[str, Any]:
        """
        Computes key matrix properties: shape, determinant, rank, trace,
        condition number, invertibility, and symmetry.
        """
        A_arr = np.asarray(A, dtype=np.float64)
        rows, cols = A_arr.shape
        is_square = (rows == cols)
        
        det = float(np.linalg.det(A_arr)) if is_square else None
        rank = int(np.linalg.matrix_rank(A_arr))
        trace = float(np.trace(A_arr)) if is_square else None
        is_symmetric = bool(np.allclose(A_arr, A_arr.T)) if is_square else False
        is_invertible = bool(is_square and abs(det) > 1e-10 and rank == rows)
        cond_num = float(np.linalg.cond(A_arr)) if is_square and is_invertible else float('inf')

        return {
            "shape": (rows, cols),
            "is_square": is_square,
            "determinant": det,
            "rank": rank,
            "trace": trace,
            "is_symmetric": is_symmetric,
            "is_invertible": is_invertible,
            "condition_number": cond_num
        }

    @staticmethod
    def inverse(A: np.ndarray) -> np.ndarray:
        """Computes matrix inverse A^-1 if invertible."""
        A_arr = np.asarray(A, dtype=np.float64)
        diag = MatrixEngine.get_diagnostics(A_arr)
        if not diag["is_invertible"]:
            raise ValueError("Matrix is singular (non-invertible) or not square.")
        return np.linalg.inv(A_arr)

    @staticmethod
    def rotation_matrix_2d(angle_degrees: float) -> np.ndarray:
        """
        Creates a 2D rotation matrix for a given angle in degrees.
        R(theta) = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
        """
        theta = np.radians(angle_degrees)
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, -s],
            [s,  c]
        ], dtype=np.float64)

    @staticmethod
    def scaling_matrix_2d(sx: float, sy: float) -> np.ndarray:
        """Creates a 2D non-uniform scaling matrix S(sx, sy)."""
        return np.array([
            [sx, 0.0],
            [0.0, sy]
        ], dtype=np.float64)

    @staticmethod
    def shear_matrix_2d(kx: float, ky: float) -> np.ndarray:
        """Creates a 2D shear matrix H(kx, ky)."""
        return np.array([
            [1.0, kx],
            [ky, 1.0]
        ], dtype=np.float64)

    @staticmethod
    def apply_transform(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Applies linear transformation matrix to a set of 2D points (N x 2).
        Returns transformed points (N x 2).
        """
        pts = np.asarray(points, dtype=np.float64)
        T = np.asarray(transform_matrix, dtype=np.float64)
        # pts is N x 2, T is 2 x 2. Transformed: (T @ pts.T).T = pts @ T.T
        return np.dot(pts, T.T)

    @staticmethod
    def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solves linear equation system A x = b.
        Returns solution vector x.
        """
        A_arr = np.asarray(A, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)
        return np.linalg.solve(A_arr, b_arr)

    @staticmethod
    def eigen_analysis(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes eigenvalues and eigenvectors of a square matrix A.
        Returns (eigenvalues, eigenvectors).
        """
        A_arr = np.asarray(A, dtype=np.float64)
        if A_arr.shape[0] != A_arr.shape[1]:
            raise ValueError("Eigen analysis requires a square matrix.")
        eigenvalues, eigenvectors = np.linalg.eig(A_arr)
        return eigenvalues, eigenvectors


if __name__ == "__main__":
    # Internal validation tests
    v1 = np.array([3.0, 4.0, 0.0])
    v2 = np.array([1.0, 2.0, 2.0])
    
    norms = VectorEngine.compute_norms(v1)
    assert np.isclose(norms["L2_norm"], 5.0)
    
    cos_sim = VectorEngine.cosine_similarity(v1, v2)
    assert -1.0 <= cos_sim <= 1.0
    
    proj = VectorEngine.project(v1, v2)
    assert proj.shape == (3,)
    
    gs_basis = VectorEngine.gram_schmidt([np.array([1.0, 1.0]), np.array([1.0, 0.0])])
    assert len(gs_basis) == 2
    assert np.isclose(np.dot(gs_basis[0], gs_basis[1]), 0.0)
    
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    diag = MatrixEngine.get_diagnostics(A)
    assert diag["is_invertible"] is True
    assert diag["is_symmetric"] is True
    
    print("VectorEngine and MatrixEngine validation tests passed successfully!")
