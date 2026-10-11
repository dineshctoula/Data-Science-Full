"""Run the Day 71 multiclass walkthrough."""

from multi_engine import evaluate, make_blobs
from visualizer import MultiPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 71: MULTICLASS")
    print("=" * 72)

    X, y = make_blobs(n_per_class=90, sep=1.4, noise=1.1, seed=71)
    result = evaluate(X, y, seed=71)

    print(f"\n1) table {X.shape}, three classes, {int((y == 0).sum())} rows each")
    print(f"   one-vs-rest accuracy {result.accuracy:.3f}")
    print(f"   majority-class accuracy {result.majority_accuracy:.3f}")

    print("\n2) confusion, rows are actual, columns are the guess")
    for row in result.confusion:
        print("  ", " ".join(f"{int(value):3d}" for value in row))

    print("\n3) recall")
    for class_id in range(result.n_classes):
        print(f"   class {class_id}: {result.recall(class_id):.3f}")

    recalls = [result.recall(class_id) for class_id in range(result.n_classes)]
    plots = MultiPlots()
    p1 = plots.confusion_heatmap(result.confusion, title="seed 71 — where the mistakes go")
    p2 = plots.recall_bars(recalls, title="seed 71 — recall by class")

    print("\n4) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 71 done.")


if __name__ == "__main__":
    run_pipeline()
