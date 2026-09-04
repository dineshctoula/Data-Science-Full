"""
Linear Algebra Visualizer & Geometric Transformation Suite
Day 36: Math & Statistics for Data Science — Linear Algebra Fundamentals

This module handles 2D vector space rendering, geometric grid transformations (rotation, shear, scaling),
orthogonal projections, and high-dimensional feature vector cosine similarity heatmaps using Matplotlib & Seaborn.
"""

import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vector_matrix_engine import VectorEngine, MatrixEngine

# Set modern aesthetic styling
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11


class LinearAlgebraVisualizer:
    """
    Visualization engine for vector spaces, grid transformations, and cosine similarity.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_vector_operations(
        self,
        v1: np.ndarray,
        v2: np.ndarray,
        filename: str = "vector_transformations.png"
    ) -> str:
        """
        Plots 2D vector addition, subtraction, and orthogonal projection.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        
        # Subplot 1: Vector Addition & Parallelogram Rule
        ax1 = axes[0]
        ax1.set_title("Vector Addition & Parallelogram Rule", fontsize=13, fontweight='bold', pad=12, color='#00F0FF')
        
        v_sum = VectorEngine.add(v1[:2], v2[:2])
        
        # Vector origins and coordinates
        ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='#00F0FF', label=f'v1 = {v1[:2]}')
        ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='#FF007F', label=f'v2 = {v2[:2]}')
        ax1.quiver(0, 0, v_sum[0], v_sum[1], angles='xy', scale_units='xy', scale=1, color='#7928CA', label=f'v1 + v2 = {v_sum}')
        
        # Dashed parallelogram lines
        ax1.plot([v1[0], v_sum[0]], [v1[1], v_sum[1]], ':', color='#FF007F', alpha=0.7)
        ax1.plot([v2[0], v_sum[0]], [v2[1], v_sum[1]], ':', color='#00F0FF', alpha=0.7)
        
        max_val = max(np.abs(v1[:2]).max(), np.abs(v2[:2]).max(), np.abs(v_sum).max()) + 2
        ax1.set_xlim(-1, max_val)
        ax1.set_ylim(-1, max_val)
        ax1.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax1.axvline(0, color='gray', linewidth=0.8, linestyle='--')
        ax1.grid(True, linestyle=':', alpha=0.4)
        ax1.legend(loc='upper left', framealpha=0.8)

        # Subplot 2: Orthogonal Projection
        ax2 = axes[1]
        ax2.set_title("Orthogonal Projection proj_u(v)", fontsize=13, fontweight='bold', pad=12, color='#00FF66')
        
        proj_v = VectorEngine.project(v1[:2], v2[:2])
        perp_v = v1[:2] - proj_v
        
        ax2.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='#FF007F', label=f'Target Basis u = {v2[:2]}')
        ax2.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='#00F0FF', label=f'Vector v = {v1[:2]}')
        ax2.quiver(0, 0, proj_v[0], proj_v[1], angles='xy', scale_units='xy', scale=1, color='#00FF66', label=f'proj_u(v) = {np.round(proj_v, 2)}')
        
        # Perpendicular drop line
        ax2.plot([v1[0], proj_v[0]], [v1[1], proj_v[1]], '--', color='#FFCC00', alpha=0.9, label='Orthogonal Residual')
        
        ax2.set_xlim(-1, max_val)
        ax2.set_ylim(-1, max_val)
        ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax2.axvline(0, color='gray', linewidth=0.8, linestyle='--')
        ax2.grid(True, linestyle=':', alpha=0.4)
        ax2.legend(loc='upper left', framealpha=0.8)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        return filepath

    def plot_geometric_transformations(
        self,
        filename: str = "geometric_transformations.png"
    ) -> str:
        """
        Visualizes grid deformation under 2D linear transformations (Rotation, Shear, Scaling).
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
        
        # Create a unit grid of points
        x = np.linspace(-2, 2, 9)
        y = np.linspace(-2, 2, 9)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        transformations = [
            ("Rotation (45°)", MatrixEngine.rotation_matrix_2d(45), '#00F0FF'),
            ("Shear (kx=0.8)", MatrixEngine.shear_matrix_2d(0.8, 0.0), '#FF007F'),
            ("Scaling (sx=1.5, sy=0.5)", MatrixEngine.scaling_matrix_2d(1.5, 0.5), '#00FF66')
        ]
        
        basis_i = np.array([1.0, 0.0])
        basis_j = np.array([0.0, 1.0])

        for ax, (title, T, color) in zip(axes, transformations):
            # Transform grid points
            transformed_pts = MatrixEngine.apply_transform(grid_points, T)
            
            # Original grid points (subtle background)
            ax.scatter(grid_points[:, 0], grid_points[:, 1], color='gray', alpha=0.2, s=15)
            
            # Transformed grid points
            ax.scatter(transformed_pts[:, 0], transformed_pts[:, 1], color=color, alpha=0.6, s=25)
            
            # Transform basis vectors
            t_i = MatrixEngine.apply_transform(basis_i.reshape(1, -1), T)[0]
            t_j = MatrixEngine.apply_transform(basis_j.reshape(1, -1), T)[0]
            
            # Plot transformed basis vectors
            ax.quiver(0, 0, t_i[0], t_i[1], angles='xy', scale_units='xy', scale=1, color='#FFCC00', label=f"i' = {np.round(t_i, 2)}")
            ax.quiver(0, 0, t_j[0], t_j[1], angles='xy', scale_units='xy', scale=1, color='#7928CA', label=f"j' = {np.round(t_j, 2)}")
            
            det_val = np.linalg.det(T)
            ax.set_title(f"{title}\nDet(T) = Area Scale = {det_val:.2f}", fontsize=12, fontweight='bold', pad=10)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.grid(True, linestyle=':', alpha=0.3)
            ax.legend(loc='lower right', framealpha=0.8)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        return filepath

    def plot_cosine_similarity_heatmap(
        self,
        feature_matrix: np.ndarray,
        labels: List[str],
        filename: str = "cosine_similarity_heatmap.png"
    ) -> str:
        """
        Computes pairwise cosine similarity matrix across feature vectors and plots Seaborn heatmap.
        """
        num_docs = feature_matrix.shape[0]
        cos_matrix = np.zeros((num_docs, num_docs))
        
        for i in range(num_docs):
            for j in range(num_docs):
                cos_matrix[i, j] = VectorEngine.cosine_similarity(feature_matrix[i], feature_matrix[j])
                
        plt.figure(figsize=(9, 7.5), dpi=300)
        ax = sns.heatmap(
            cos_matrix,
            annot=True,
            fmt=".3f",
            cmap="mako",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Cosine Similarity (cos θ)'},
            linewidths=0.5
        )
        
        plt.title("High-Dimensional Feature Vector Cosine Similarity Heatmap", fontsize=13, fontweight='bold', pad=14, color='#00F0FF')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        return filepath
