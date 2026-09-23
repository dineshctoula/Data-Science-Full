# Day 50 — Decision Trees

Quick notes from running the walkthrough (seed=50).

## What I built

- CART-style binary tree: try every feature × midpoint threshold, pick the
  split that drops gini (or entropy) the most.
- Stops when a node is pure, hits `max_depth`, or has fewer than
  `min_samples_split` rows.
- Prediction: walk left on `x[j] <= threshold`, right otherwise; leaf vote
  is the majority label.

## Numbers from `main.py`

**Axis-aligned blobs** (easy case):
- One vertical cut around `x[0] <= 1.935` already gets train/test accuracy **1.0**.
- Gini and entropy both land on the same shallow tree here.

**XOR-ish corners** (needs depth):
| max_depth | train | test |
|----------:|------:|-----:|
| 1 | 0.643 | 0.333 |
| 2 | 0.726 | 0.667 |
| 3 | 0.810 | 0.722 |
| 4 | 0.976 | 1.000 |
| 6+ | 1.000 | 1.000 |

Depth 1 is basically a coin flip on XOR — makes sense, one axis cut can't
separate diagonal classes. By depth 4 the rectangles finally carve out the
corners.

## Takeaways

- Trees give axis-aligned decision regions (stair-step boundaries).
- Deeper ≠ always better on noisy data; here the XOR set is clean so train
  and test both climb.
- Gini vs entropy rarely changes much on these toys; gini is a bit cheaper.

## Files

- `tree_engine.py` — impurity helpers + `DecisionTreeClassifier`
- `visualizer.py` — regions, depth sweep, impurity bars, confusion
- `test_tree_engine.py` — 10 tests
- `main.py` — runnable lesson
