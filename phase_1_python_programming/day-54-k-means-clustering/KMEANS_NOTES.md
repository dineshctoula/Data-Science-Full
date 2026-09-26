# Day 54 — K-Means

Unsupervised piles. No labels at fit time.

## Loop

1. place k centers (k-means++-ish)
2. assign each point to nearest center
3. move center → mean of its members
4. repeat until centers barely move

Inertia = sum of squared distances to your own center. Lower is tighter.

## Numbers (seed=54)

**Neat 3-blobs, k=3:** inertia **58.0** in 2 iters.

**Elbow:**
| k | inertia |
|--:|--------:|
| 1 | 735 |
| 2 | 392 |
| 3 | 58 |
| 4 | 48 |
| 5 | 41 |

Clear bend at k=3 — matches how the data was made. After that you're just
carving smaller pieces of the same piles.

**Uneven blobs:** inertia ≈ 78 — still finds 3 groups, but the spread-out
cloud is an awkward spherical-assumption case.

## Takeaways

- k is a hyperparameter; elbow / silhouette help guess it.
- Multiple restarts matter — bad init can stick.
- Assumes round-ish equal-ish clusters. Weird shapes → try something else
  (GMM, DBSCAN) later.
