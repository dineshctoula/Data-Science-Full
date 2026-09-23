"""Run the Day 50 decision tree walkthrough and dump a few plots."""

from tree_engine import (
    DecisionTreeClassifier,
    gini,
    make_axis_aligned_blobs,
    make_xor_like,
    train_test_split,
)
from visualizer import TreePlots


def run_pipeline():
    print("=" * 72)
    print("DAY 50: DECISION TREES")
    print("=" * 72)

    # --- easy blobs ---
    X, y = make_axis_aligned_blobs(n_per_class=45, seed=50)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.3, seed=50)

    print("\n1) axis-aligned blobs, gini, max_depth=3")
    model = DecisionTreeClassifier(max_depth=3, criterion="gini").fit(Xtr, ytr)
    train_r = model.score(Xtr, ytr)
    test_r = model.score(Xte, yte)
    print("  ", train_r.summary())
    print("   test accuracy:", round(test_r.accuracy, 3))
    print("\n   tree sketch:")
    for line in model.print_tree().splitlines():
        print("   ", line)

    print("\n2) same data, entropy criterion")
    ent = DecisionTreeClassifier(max_depth=3, criterion="entropy").fit(Xtr, ytr)
    print("  ", ent.score(Xte, yte).summary())

    # --- depth sweep (overfitting story) ---
    print("\n3) depth sweep on XOR-ish data (train vs test)")
    Xx, yx = make_xor_like(n_per_class=60, seed=50)
    Xtr2, Xte2, ytr2, yte2 = train_test_split(Xx, yx, test_frac=0.3, seed=51)
    depths = [1, 2, 3, 4, 6, 8]
    train_accs, test_accs = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d).fit(Xtr2, ytr2)
        train_accs.append(m.score(Xtr2, ytr2).accuracy)
        test_accs.append(m.score(Xte2, yte2).accuracy)
        print(f"   depth={d}: train={train_accs[-1]:.3f}  test={test_accs[-1]:.3f}")

    # --- one hand-wavy impurity demo ---
    print("\n4) gini before/after a rough x0 split at 2.0 on the blobs")
    left = y[X[:, 0] <= 2.0]
    right = y[X[:, 0] > 2.0]
    print(f"   parent gini={gini(y):.3f}  left={gini(left):.3f}  right={gini(right):.3f}")

    plots = TreePlots()
    xor_model = DecisionTreeClassifier(max_depth=4).fit(Xx, yx)
    regions = plots.decision_map(xor_model, Xx, yx, title="XOR-ish regions", filename="tree_xor_regions.png")
    blob_regions = plots.decision_map(model, X, y, title="blob regions", filename="tree_blob_regions.png")
    sweep = plots.depth_sweep(depths, train_accs, test_accs, title="XOR: depth vs accuracy")
    bars = plots.impurity_bars(y, left, right, title="blobs: gini after x0<=2 cut")
    conf = plots.confusion_simple(yte2, DecisionTreeClassifier(max_depth=4).fit(Xtr2, ytr2).predict(Xte2))

    print("\n5) plots saved")
    for p in (regions, blob_regions, sweep, bars, conf):
        print("  ", p)
    print("\nDay 50 done.")


if __name__ == "__main__":
    run_pipeline()
