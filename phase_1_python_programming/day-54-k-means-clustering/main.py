"""Run the Day 54 k-means walkthrough."""

import numpy as np

from kmeans_engine import KMeans, elbow_scores, make_blobs, make_uneven_blobs
from visualizer import KMeansPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 54: K-MEANS CLUSTERING")
    print("=" * 72)

    X, y = make_blobs(k=3, n_per=45, seed=54)

    print("\n1) three neat blobs, k=3")
    km = KMeans(k=3, n_init=10, seed=54).fit(X)
    print("  ", km.result().summary())
    print("   center coords:\n", np.round(km.centers_, 3))

    print("\n2) elbow sweep (looking for the bend)")
    scores = elbow_scores(X, ks=range(1, 8), seed=54)
    for k, ine in scores.items():
        mark = " <-- ?" if k == 3 else ""
        print(f"   k={k}: inertia={ine:.1f}{mark}")

    print("\n3) uneven blobs — same algorithm, messier piles")
    Xu, yu = make_uneven_blobs(seed=54)
    km_u = KMeans(k=3, n_init=10, seed=55).fit(Xu)
    print("  ", km_u.result().summary())

    plots = KMeansPlots()
    p1 = plots.clusters(km, X, title="three blobs")
    p2 = plots.elbow(scores, title="elbow on neat blobs")
    p3 = plots.inertia_curve(km.inertia_curve_, title="best restart: inertia vs iter")
    p4 = plots.truth_vs_pred(X, y, km.labels_, title="neat blobs")
    p5 = plots.clusters(km_u, Xu, title="uneven blobs", filename="kmeans_uneven.png")

    print("\n4) plots saved")
    for p in (p1, p2, p3, p4, p5):
        print("  ", p)
    print("\nDay 54 done.")


if __name__ == "__main__":
    run_pipeline()
