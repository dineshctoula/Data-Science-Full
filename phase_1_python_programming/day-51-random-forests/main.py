"""Run the Day 51 random forest walkthrough and dump a few plots."""

from forest_engine import (
    RandomForestClassifier,
    compare_n_trees,
    make_noisy_moons,
    make_wide_features,
    train_test_split,
)
from visualizer import ForestPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 51: RANDOM FORESTS")
    print("=" * 72)

    # --- noisy moons ---
    X, y = make_noisy_moons(n_per_class=70, noise=0.28, seed=51)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.3, seed=51)

    print("\n1) one deep-ish tree vs a small forest (noisy moons)")
    single = RandomForestClassifier(n_trees=1, max_depth=8, seed=51).fit(Xtr, ytr)
    forest = RandomForestClassifier(n_trees=40, max_depth=8, seed=51).fit(Xtr, ytr)
    print("   single tree test:", round(single.score(Xte, yte).accuracy, 3))
    print("   forest test:     ", round(forest.score(Xte, yte).accuracy, 3))
    print("   forest OOB:      ", round(forest.oob_score(Xtr, ytr), 3))

    print("\n2) growing the forest — does more trees help?")
    sweep = compare_n_trees(X, y, n_list=(1, 5, 10, 20, 40), seed=51)
    for n, acc in sweep.items():
        print(f"   n_trees={n:2d}: test acc={acc:.3f}")

    # --- wide features (noise columns) ---
    print("\n3) wide feature set (only x0/x1 matter)")
    Xw, yw = make_wide_features(n=140, p=8, seed=51)
    Xtrw, Xtew, ytrw, ytew = train_test_split(Xw, yw, test_frac=0.3, seed=52)
    # force looking at all features vs sqrt — sqrt should ignore junk more often
    all_f = RandomForestClassifier(n_trees=30, max_features=None, seed=52).fit(Xtrw, ytrw)
    sqrt_f = RandomForestClassifier(n_trees=30, max_features="sqrt", seed=52).fit(Xtrw, ytrw)
    print("   max_features=all :", round(all_f.score(Xtew, ytew).accuracy, 3))
    print("   max_features=sqrt:", round(sqrt_f.score(Xtew, ytew).accuracy, 3))

    plots = ForestPlots()
    regions = plots.decision_map(forest, X, y, title="noisy moons — forest")
    curve = plots.n_trees_curve(sweep, title="moons: test acc vs n_trees")
    vs = plots.single_vs_forest(
        Xte, yte, single.predict(Xte), forest.predict(Xte), title="test set misses"
    )
    conf = plots.confusion_simple(yte, forest.predict(Xte), title="forest test confusion")

    print("\n4) plots saved")
    for p in (regions, curve, vs, conf):
        print("  ", p)
    print("\nDay 51 done.")


if __name__ == "__main__":
    run_pipeline()
