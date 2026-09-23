"""Run the Day 49 KNN walkthrough and dump a few plots."""

from knn_engine import (
    KNearestNeighbors,
    compare_k_values,
    leave_one_out_accuracy,
    make_blob_classes,
    make_two_moons_ish,
)
from visualizer import KNNPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 49: k-NEAREST NEIGHBORS")
    print("=" * 72)

    # --- blobs first (easy case) ---
    X, y = make_blob_classes(n_per_class=40, seed=49)
    print("\n1) three blobs, k=5 (euclidean)")
    model = KNearestNeighbors(k=5).fit(X, y)
    train_result = model.score(X, y)
    print("   train accuracy (optimistic):", round(train_result.accuracy, 3))
    loo = leave_one_out_accuracy(X, y, k=5)
    print("   leave-one-out accuracy:     ", round(loo, 3))

    print("\n2) same data but manhattan distance")
    man = KNearestNeighbors(k=5, distance="manhattan").fit(X, y)
    print("  ", man.score(X, y).summary())

    # --- k sweep ---
    print("\n3) LOOCV across a few k values")
    sweep = compare_k_values(X, y, ks=[1, 3, 5, 9, 15])
    for k, acc in sweep.items():
        marker = " <-- best" if acc == max(sweep.values()) else ""
        print(f"   k={k:2d}: {acc:.3f}{marker}")

    # --- moons (harder boundary) ---
    Xm, ym = make_two_moons_ish(n_per_class=60, seed=49)
    print("\n4) interlocking moons (non-linear boundary)")
    moon_model = KNearestNeighbors(k=7).fit(Xm, ym)
    print("   LOOCV k=7:", round(leave_one_out_accuracy(Xm, ym, k=7), 3))
    # k=1 often overfits noise on moons; still worth printing
    print("   LOOCV k=1:", round(leave_one_out_accuracy(Xm, ym, k=1), 3))

    # --- one concrete prediction ---
    query = Xm.mean(axis=0)  # somewhere in the middle, just for demo
    pred = moon_model.predict(query)
    print("\n5) predict a point near the data centroid → class", int(pred[0]))

    plots = KNNPlots()
    regions = plots.decision_map(moon_model, Xm, ym, title="moons decision map", filename="knn_regions.png")
    sweep_plot = plots.k_sweep_bars(sweep, title="blob data: LOOCV vs k")
    neighbors = plots.neighbor_example(moon_model, Xm, ym, query, title="neighbors around the query")
    conf = plots.confusion_simple(ym, moon_model.predict(Xm), title="moons train confusion (k=7)")

    print("\n6) plots saved")
    print("  ", regions)
    print("  ", sweep_plot)
    print("  ", neighbors)
    print("  ", conf)
    print("\nDay 49 done.")


if __name__ == "__main__":
    run_pipeline()
