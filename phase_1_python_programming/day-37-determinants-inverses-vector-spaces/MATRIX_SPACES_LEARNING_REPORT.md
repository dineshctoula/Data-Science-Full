# Day 37 — Determinants, Matrix Inverses & Vector Spaces

## Goal

Connect three closely related ideas: a determinant measures signed geometric scaling, a non-zero determinant permits an inverse, and rank describes how many independent directions a matrix preserves.

## What I built

- `matrix_space_engine.py` — cofactor-expansion determinants, minors, cofactors, adjugate inverses, invertibility diagnostics, SVD-based four-subspace analysis, and least-squares solving.
- `visualizer.py` — a unit-square transformation chart that makes determinant area scaling visible.
- `main.py` — an executable learning walkthrough and numerical verification suite.

## Key results

| Concept | Result | Interpretation |
| --- | ---: | --- |
| Determinant of `[[4, 7], [2, 6]]` | `10` | The transformation scales areas by 10 and is invertible. |
| Inverse check | `A @ A⁻¹ = I` | The adjugate formula agrees with the defining inverse identity. |
| Rank of the example matrix `B` | `2` | Only two independent directions remain. |
| Nullity of `B` | `1` | One direction is mapped to the zero vector. |
| Rank–nullity | `2 + 1 = 3` | The dimensions account for all three input coordinates. |
| Null-space validation | `||B @ N|| ≈ 2.99e-16` | Floating-point zero confirms the computed null-space basis. |

## Takeaways

1. `det(A) = 0` means a square matrix collapses volume into a lower-dimensional space; it cannot be inverted.
2. `|det(A)|` measures area/volume scaling, while its sign also records orientation reversal.
3. The SVD gives numerically stable orthonormal bases for the column, row, null, and left-null spaces.
4. Least squares provides a useful solution even when a system is not uniquely solvable.

## Run it

```bash
python3 matrix_space_engine.py
python3 main.py
```

The complete pipeline writes `output/determinant_area_scaling.png`, comparing expansion, area-preserving shear, and singular flattening.
