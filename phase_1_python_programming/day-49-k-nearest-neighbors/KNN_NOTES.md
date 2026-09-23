# Day 49 — k-Nearest Neighbors

## What I built

- `knn_engine.py` — store the training set, measure euclidean/manhattan distance, majority vote (with a distance tie-break), LOOCV helpers, blob + moons datasets
- `visualizer.py` — decision regions, k-sweep line chart, neighbor circle around a query, confusion counts
- `main.py` — run the demos and dump plots into `output/`
- `test_knn_engine.py` — basic regressions for the stuff that broke while I was writing it

## Notes from running it

Blobs (3 classes) with k=5:

- train accuracy ≈ 0.99 (a bit rosy — points vote with themselves nearby)
- LOOCV ≈ 0.99 as well on this easy set

Moons:

- LOOCV k=7 ≈ 0.97
- LOOCV k=1 ≈ 0.95 (a little noisier, as expected)

Takeaway I keep repeating to myself: **small k = wiggly boundary, large k = smoother but can wash out thin shapes**.

## How to run

```bash
cd phase_1_python_programming/day-49-k-nearest-neighbors
python3 -m unittest -v
python3 main.py
```

Plots land in `output/knn_regions.png`, `k_sweep.png`, `neighbor_example.png`, `knn_confusion.png`.
