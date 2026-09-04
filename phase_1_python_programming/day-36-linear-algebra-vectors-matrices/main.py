"""
Pipeline Orchestrator — Day 36: Linear Algebra & Vector Analytics Platform

Executes comprehensive linear algebra calculations:
1. Vector norm, angle, projection & Gram-Schmidt Orthonormalization.
2. Matrix diagnostics, invertibility, trace, condition number & linear system solving (A x = b).
3. Embedding vector cosine similarity heatmap for NLP document representation.
4. Generates publication-ready visualizations of vector spaces and geometric transformations.
"""

import sys
import os
import numpy as np
from vector_matrix_engine import VectorEngine, MatrixEngine
from linear_algebra_visualizer import LinearAlgebraVisualizer


def run_pipeline():
    print("=" * 80)
    print("⚡ DAY 36: MATH & STATISTICS FOR DATA SCIENCE — LINEAR ALGEBRA ANALYTICS ENGINE")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. VECTOR OPERATIONS & GRAM-SCHMIDT ORTHONORMALIZATION
    # ---------------------------------------------------------
    print("\n🔹 Step 1: Vector Algebra & Vector Space Operations")
    v1 = np.array([4.0, 2.0, 0.0])
    v2 = np.array([1.0, 3.0, 2.0])
    v3 = np.array([0.0, 1.0, 5.0])

    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")

    dot_prod = VectorEngine.dot_product(v1, v2)
    cos_sim = VectorEngine.cosine_similarity(v1, v2)
    angle_deg = VectorEngine.angle_between(v1, v2)
    norms_v1 = VectorEngine.compute_norms(v1)

    print(f"Dot Product (v1 · v2): {dot_prod:.4f}")
    print(f"Cosine Similarity cos(θ): {cos_sim:.4f}")
    print(f"Angle between v1 & v2: {angle_deg:.2f}°")
    print(f"v1 Norms: L1={norms_v1['L1_norm']:.2f}, L2={norms_v1['L2_norm']:.2f}, Linf={norms_v1['Linf_norm']:.2f}")

    # Orthogonal projection of v1 onto v2
    proj_v1_on_v2 = VectorEngine.project(v1, v2)
    print(f"Orthogonal Projection of v1 onto v2: proj_v2(v1) = {np.round(proj_v1_on_v2, 4)}")

    # Gram-Schmidt Orthonormalization
    print("\n--- Gram-Schmidt Orthonormalization ---")
    raw_vectors = [v1, v2, v3]
    orthonormal_basis = VectorEngine.gram_schmidt(raw_vectors)
    print(f"Input Vectors Count: {len(raw_vectors)}")
    print(f"Orthonormal Basis Count: {len(orthonormal_basis)}")
    for i, e in enumerate(orthonormal_basis):
        norm_e = np.linalg.norm(e)
        print(f"  Basis e{i+1}: {np.round(e, 4)} | Norm: {norm_e:.4f}")
    
    # Verify pairwise orthogonality
    dot_e1_e2 = np.dot(orthonormal_basis[0], orthonormal_basis[1])
    dot_e1_e3 = np.dot(orthonormal_basis[0], orthonormal_basis[2])
    print(f"Orthogonality Check e1 · e2: {dot_e1_e2:.6e} (Target: 0)")
    print(f"Orthogonality Check e1 · e3: {dot_e1_e3:.6e} (Target: 0)")

    # ---------------------------------------------------------
    # 2. MATRIX DIAGNOSTICS & SYSTEM SOLVING (A x = b)
    # ---------------------------------------------------------
    print("\n🔹 Step 2: Matrix Diagnostics & Linear System Solving")
    A = np.array([
        [3.0, 1.0, -1.0],
        [2.0, 4.0,  1.0],
        [-1.0, 2.0, 5.0]
    ])
    b = np.array([4.0, 13.0, 16.0])

    print("Coefficient Matrix A:")
    print(A)
    print(f"Target Vector b: {b}")

    diag = MatrixEngine.get_diagnostics(A)
    print("\nMatrix A Properties:")
    print(f"  Shape: {diag['shape']}")
    print(f"  Determinant det(A): {diag['determinant']:.4f}")
    print(f"  Rank: {diag['rank']}")
    print(f"  Trace: {diag['trace']:.4f}")
    print(f"  Symmetric: {diag['is_symmetric']}")
    print(f"  Invertible: {diag['is_invertible']}")
    print(f"  Condition Number: {diag['condition_number']:.4f}")

    # Solve A x = b
    x_sol = MatrixEngine.solve_linear_system(A, b)
    print(f"\nSolution Vector x = A^-1 b: {np.round(x_sol, 4)}")
    
    # Residual error validation ||A x - b||
    residual = np.linalg.norm(np.dot(A, x_sol) - b)
    print(f"Residual Error ||A x - b||: {residual:.6e}")

    # Eigenvalue decomposition
    evals, evecs = MatrixEngine.eigen_analysis(A)
    print("\nEigenvalue Analysis:")
    for i, ev in enumerate(evals):
        print(f"  λ_{i+1} = {ev:.4f} | Eigenvector v_{i+1} = {np.round(evecs[:, i], 4)}")

    # ---------------------------------------------------------
    # 3. HIGH-DIMENSIONAL FEATURE COSINE SIMILARITY (NLP DEMO)
    # ---------------------------------------------------------
    print("\n🔹 Step 3: High-Dimensional Feature Space & Cosine Similarity")
    doc_labels = ["Doc1: Linear Algebra", "Doc2: Vector Calculus", "Doc3: Machine Learning", "Doc4: Organic Chemistry"]
    # Synthetic TF-IDF term frequency vectors (5 features: Math, Vectors, Learning, Science, Chemistry)
    feature_vectors = np.array([
        [0.85, 0.90, 0.20, 0.10, 0.00],  # Doc1
        [0.80, 0.85, 0.30, 0.15, 0.00],  # Doc2
        [0.70, 0.60, 0.95, 0.20, 0.00],  # Doc3
        [0.05, 0.00, 0.10, 0.85, 0.95]   # Doc4
    ])

    sim_doc1_doc2 = VectorEngine.cosine_similarity(feature_vectors[0], feature_vectors[1])
    sim_doc1_doc4 = VectorEngine.cosine_similarity(feature_vectors[0], feature_vectors[3])

    print(f"Cosine Similarity (Doc1 vs Doc2 - Math topics): {sim_doc1_doc2:.4f}")
    print(f"Cosine Similarity (Doc1 vs Doc4 - Math vs Chem): {sim_doc1_doc4:.4f}")

    # ---------------------------------------------------------
    # 4. VISUALIZATION SUITE GENERATION
    # ---------------------------------------------------------
    print("\n🔹 Step 4: Generating Linear Algebra Visualizations...")
    visualizer = LinearAlgebraVisualizer(output_dir="output")

    v_a = np.array([4.0, 2.0])
    v_b = np.array([1.0, 3.0])
    fig1 = visualizer.plot_vector_operations(v_a, v_b)
    print(f"  ✓ Saved Vector Operations Plot: {fig1}")

    fig2 = visualizer.plot_geometric_transformations()
    print(f"  ✓ Saved Geometric Transformations Plot: {fig2}")

    fig3 = visualizer.plot_cosine_similarity_heatmap(feature_vectors, doc_labels)
    print(f"  ✓ Saved Cosine Similarity Heatmap: {fig3}")

    print("\n" + "=" * 80)
    print("✅ DAY 36 PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
