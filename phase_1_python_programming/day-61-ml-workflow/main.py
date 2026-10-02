"""Run the Day 61 train / validate / compare walkthrough."""

from workflow_engine import compare_models, make_signup_data, pick_by_validation
from visualizer import WorkflowPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 61: TRAIN, VALIDATE, COMPARE")
    print("=" * 72)

    X, y, names = make_signup_data(n=260, seed=61)
    print(f"\n1) signup data: n={len(y)}, features={names}")
    print(f"   signup rate: {y.mean():.3f}")

    print("\n2) same split for every model")
    scores = compare_models(X, y, val_frac=0.3, seed=61)
    for s in scores:
        gap = s.train_acc - s.val_acc
        print(f"   {s.summary()}   gap={gap:+.3f}")

    winner = pick_by_validation(scores)
    print(f"\n   picked by validation: {winner.name} ({winner.val_acc:.3f})")

    print("\n3) a second split, in case the first one was kind")
    again = compare_models(X, y, val_frac=0.3, seed=62)
    again_winner = pick_by_validation(again)
    for s in again:
        print(f"   {s.summary()}")
    print(f"   picked this time: {again_winner.name}")

    plots = WorkflowPlots()
    p1 = plots.accuracy_bars(scores, title="seed 61: train vs validation")
    p2 = plots.visits_scatter(X, y, names, title="visits and spend, colored by signup")

    print("\n4) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 61 done.")


if __name__ == "__main__":
    run_pipeline()
