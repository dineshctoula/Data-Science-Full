"""Run the Day 55 DBSCAN walkthrough."""

from dbscan_engine import DBSCAN, make_blobs_with_outliers, make_two_moons
from visualizer import DBSCANPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 55: DBSCAN")
    print("=" * 72)

    X, y = make_two_moons(n_per=70, noise=0.07, seed=55)
    print("\n1) two moons (k-means would mash these)")
    moons = DBSCAN(eps=0.22, min_samples=5).fit(X)
    print("  ", moons.result().summary())
    print("   core points:", int(moons.core_mask_.sum()))

    print("\n2) eps sweep on the same moons")
    sweep = []
    for eps in (0.08, 0.15, 0.22, 0.35, 0.6):
        m = DBSCAN(eps=eps, min_samples=5).fit(X)
        r = m.result()
        sweep.append((eps, r.n_clusters, r.n_noise))
        print(f"   eps={eps:.2f}: clusters={r.n_clusters}, noise={r.n_noise}")

    print("\n3) blobs + planted outliers")
    Xb, yb = make_blobs_with_outliers(n_per=40, n_out=12, seed=55)
    blobs = DBSCAN(eps=0.5, min_samples=5).fit(Xb)
    print("  ", blobs.result().summary())

    plots = DBSCANPlots()
    p1 = plots.clusters(moons, X, title="moons", filename="dbscan_moons.png")
    p2 = plots.clusters(blobs, Xb, title="blobs + outliers", filename="dbscan_blobs.png")
    p3 = plots.eps_sweep(sweep, title="moons: eps sweep")

    print("\n4) plots saved")
    for p in (p1, p2, p3):
        print("  ", p)
    print("\nDay 55 done.")


if __name__ == "__main__":
    run_pipeline()
