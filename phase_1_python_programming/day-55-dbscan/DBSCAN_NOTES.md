# Day 55 — DBSCAN

Density clustering. No k. eps and min_samples instead.

## What counts

- **core**: at least `min_samples` points (including itself) inside eps
- **border**: reachable from a core, but not dense itself
- **noise**: label `-1`, nobody dense claimed it

## Numbers (seed=55)

**Moons**, eps=0.22, min_samples=5:
- **2 clusters**, 1 noise point, 133 cores

**eps sweep** (same moons):
| eps | clusters | noise |
|----:|---------:|------:|
| 0.08 | 1 | 134 |
| 0.15 | 14 | 11 |
| 0.22 | 2 | 1 |
| 0.35 | 1 | 0 |
| 0.60 | 1 | 0 |

Too small → almost everything is noise (or tiny fragments). Too big → the
two moons glue into one blob. 0.22 was the sweet spot this seed.

**Blobs + 12 outliers**, eps=0.5: **2 clusters, 11 noise**. One junk point
happened to land near a blob. That's the density rule, not a bug.

## Takeaways

- Good when clusters aren't round and you want outliers called out.
- eps is the fiddly knob — a small sweep beats guessing once.
- O(n²) neighborhood scan is fine for toys; a real run would want a tree.
