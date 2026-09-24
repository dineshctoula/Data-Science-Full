# Day 51 — Random Forests

Notes from the walkthrough run (seed=51).

## Idea in one line

Train a pile of slightly different trees (bootstrap rows + random feature
subsets at each split), then majority-vote. Individual trees overfit; the
average usually doesn't as hard.

## Numbers

**Noisy moons** (test set):
- 1 tree → **0.810**
- forest (40 trees) → **0.857**
- OOB on train ≈ **0.898** (free validation, no holdout needed)

**n_trees sweep** (same moons split):
| n | test |
|--:|-----:|
| 1 | 0.810 |
| 5 | 0.905 |
| 10 | 0.929 |
| 20 | 0.857 |
| 40 | 0.857 |

More trees helped up to ~10 here, then bounced a bit — small test set
noise, not a theorem. Still better than the lonely tree.

**Wide features** (8 cols, only x0/x1 matter):
- max_features=all → 0.762
- max_features=sqrt → 0.690

sqrt didn't win this seed. That's fine — random subspaces help more when
p is huge / noise is nastier. Point is the knob exists.

## Takeaways

- Bagging = resample rows. Feature randomness = don't always split on the
  same greedy column.
- OOB is handy when you don't want to carve out a val set.
- Forests still give axis-aligned regions, just smoother-looking once you
  vote a lot of trees.

## Files

- `forest_engine.py` — `_Tree` + `RandomForestClassifier`
- `visualizer.py` — regions, n_trees curve, miss overlay
- `test_forest_engine.py` — 8 tests
- `main.py` — runnable lesson
