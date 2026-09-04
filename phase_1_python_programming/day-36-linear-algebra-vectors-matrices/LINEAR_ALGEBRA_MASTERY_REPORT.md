# 📐 Day 36 Technical Report: Linear Algebra Foundations for Data Science & Machine Learning

## 📑 Executive Overview

Linear algebra serves as the mathematical foundation for modern data science, high-dimensional representation, machine learning models, and deep learning neural architectures. On **Day 36 of the 100-Day Data Science Challenge**, we launched **Phase 3 (Math & Statistics for Data Science)** by constructing a production-grade, object-oriented **Linear Algebra Vector & Matrix Analytics Engine**, geometric grid transformation visualizer, and feature space cosine similarity analyzer.

---

## 🔬 Key Mathematical Concepts & Implementations

### 1. Vector Operations & Inner Products
- **Vector Space ($\mathbb{R}^n$)**: An ordered array of scalars representing points or direction vectors in continuous $n$-dimensional space.
- **Dot Product ($u \cdot v$)**:
  $$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \cos(\theta)$$
- **Norms**:
  - **$L_1$ Norm (Manhattan)**: $\|\mathbf{v}\|_1 = \sum |v_i|$ (used in Lasso regularization).
  - **$L_2$ Norm (Euclidean)**: $\|\mathbf{v}\|_2 = \sqrt{\sum v_i^2}$ (used in Ridge regularization, distance metrics).
  - **$L_\infty$ Norm (Maximum)**: $\|\mathbf{v}\|_\infty = \max |v_i|$.
- **Cosine Similarity**:
  $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  - Evaluates orientation independent of magnitude. Vital for document embeddings (TF-IDF, word2vec, LLM vector search).

### 2. Orthogonal Projection & Gram-Schmidt Orthonormalization
- **Orthogonal Projection**:
  $$\text{proj}_{\mathbf{u}}(\mathbf{v}) = \left(\frac{\mathbf{v} \cdot \mathbf{u}}{\|\mathbf{u}\|_2^2}\right) \mathbf{u}$$
- **Gram-Schmidt Process**: Converts a set of linearly independent vectors $\{v_1, v_2, \dots, v_k\}$ into an orthonormal basis set $\{e_1, e_2, \dots, e_k\}$ where:
  $$e_i \cdot e_j = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

### 3. Matrix Transformations & System Diagnostics
- **Matrix Multiplication**: $\mathbf{C} = \mathbf{A} \mathbf{B}$ where $C_{ij} = \sum_k A_{ik} B_{kj}$.
- **Determinant $\det(\mathbf{A})$**: Measures the scaling factor of the volume transformed by matrix $\mathbf{A}$. $\det(\mathbf{A}) = 0$ indicates a singular (non-invertible) matrix that collapses space into a lower dimension.
- **Matrix Rank**: Count of linearly independent rows/columns. Full rank ($\text{rank}(\mathbf{A}) = n$) guarantees unique solution for $\mathbf{A}\mathbf{x} = \mathbf{b}$.
- **Condition Number**: $\kappa(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\|$. Indicates numerical stability when solving linear equations.

---

## 📊 Summary of Executed Benchmark Results

| Metric / Operation | Tested Inputs / Matrices | Output Result | Interpretation |
| :--- | :--- | :--- | :--- |
| **Dot Product** | $v_1=[4,2,0], v_2=[1,3,2]$ | `10.0000` | Positive inner product indicating acute angle ($\theta < 90^\circ$). |
| **Cosine Similarity** | Math Doc vs Vector Doc | `0.9949` | Near-perfect orientation alignment in 5D feature space. |
| **Cosine Similarity** | Math Doc vs Chem Doc | `0.0916` | Nearly orthogonal ($\approx 90^\circ$), indicating distinct domain topics. |
| **Orthogonal Projection** | $\text{proj}_{v2}(v1)$ | `[0.7143, 2.1429, 1.4286]` | Decomposed shadow vector onto $v_2$ axis. |
| **Gram-Schmidt Basis** | 3D Random Inputs | Orthogonal Error: $< 10^{-16}$ | Numerical precision confirmed exact orthonormal basis. |
| **Linear System Solution** | $A x = b$ ($3\times 3$ system) | $x = [1.7429, 1.6571, 2.8857]$ | Residual error $\|Ax - b\| = 1.78 \times 10^{-15}$. |

---

## 📈 Visual Artifacts Generated

1. **`output/vector_transformations.png`**:
   - Parallelogram rule of 2D vector addition.
   - Perpendicular drop line illustrating orthogonal projection $\text{proj}_u(v)$ and residual error vector.
2. **`output/geometric_transformations.png`**:
   - Grid point deformation under 2D Rotation ($45^\circ$), Shear ($k_x=0.8$), and Non-Uniform Scaling ($s_x=1.5, s_y=0.5$).
   - Transformed basis vectors $\hat{i}', \hat{j}'$ and determinant area scaling factors.
3. **`output/cosine_similarity_heatmap.png`**:
   - Seaborn heatmap illustrating pairwise document embedding similarity across technical domains.

---

## 🎯 Verification & Conclusions

- The vector engine passed all unit test assertions for $L_1/L_2/L_\infty$ norms, Gram-Schmidt orthogonality, and matrix determinants.
- The linear system solver achieved near-zero floating point error ($1.78 \times 10^{-15}$), validating the mathematical stability of MatrixEngine.
- All code modules (`vector_matrix_engine.py`, `linear_algebra_visualizer.py`, `main.py`) adhere strictly to clean code practices, PEP 8 standards, and type hint guidelines.
