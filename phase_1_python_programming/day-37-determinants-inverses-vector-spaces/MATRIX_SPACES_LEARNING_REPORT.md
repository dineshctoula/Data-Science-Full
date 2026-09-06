# Day 37 — Determinants, Matrix Inverses & Vector Spaces

## Goal

Connect three closely related ideas: a determinant measures signed geometric scaling, a non-zero determinant permits an inverse, and rank describes how many independent directions a matrix preserves.

## What I built

- `matrix_space_engine.py` — cofactor-expansion determinants, minors, cofactors, adjugate inverses, invertibility diagnostics, SVD-based four-subspace analysis, and least-squares solving.
- `visualizer.py` — unit-square area-scaling and standard-basis transformation charts.
- `main.py` — an executable learning walkthrough that generates both visual artifacts.
- `test_matrix_space_engine.py` and `test_visualizer.py` — seven automated regression tests for numerical identities, invalid inputs, and image generation.

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
# Run the complete learning walkthrough and generate the PNG artifacts.
python3 main.py

# Run the automated regression suite before committing changes.
python3 -m unittest -v
```

The complete pipeline writes the following artifacts to `output/`:

- `determinant_area_scaling.png` — compares expansion, area-preserving shear, and singular flattening.
- `basis_transformation.png` — compares the standard basis with its image under the invertible example matrix.

## Implementation notes

- Rank detection uses a relative singular-value cutoff, so scaling every matrix entry by the same non-zero constant does not change the detected rank.
- An inverse is verified in both orders (`A @ A⁻¹` and `A⁻¹ @ A`) because matrix multiplication is not generally commutative.
- The visualizer deliberately accepts only `2 × 2` matrices: the displayed unit-square geometry is an area-based explanation rather than a misleading projection of higher-dimensional data.
