"""Run the Day 72 early-stopping walkthrough."""

from stop_engine import fit_until, make_noisy_classes
from visualizer import StopPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 72: EARLY STOPPING")
    print("=" * 72)

    X, y = make_noisy_classes(n=220, n_noise=20, sep=1.8, seed=72)
    result = fit_until(X, y, n_train=70, n_val=60, lr=0.25, epochs=200, seed=72)

    print(f"\n1) table {X.shape}, class rate {y.mean():.3f}")
    print("   70 train, 60 validation, the rest held out")
    print("   columns 0 and 1 are real, the other 20 are noise")

    print("\n2) log loss along the way")
    print(f"   {'epoch':>6}  {'train':>8}  {'val':>8}")
    show = [1, 5, 10, 20, 50, 100, 200]
    for epoch in show:
        i = epoch - 1
        print(f"   {epoch:6d}  {result.train_loss[i]:8.3f}  {result.val_loss[i]:8.3f}")

    print(f"\n3) lowest validation loss at epoch {result.best_epoch} ({result.best_val:.3f})")
    print(f"   test log loss  stopped {result.test_loss_best:.3f}   last epoch {result.test_loss_final:.3f}")
    print(f"   test accuracy  stopped {result.test_acc_best:.3f}   last epoch {result.test_acc_final:.3f}")

    plots = StopPlots()
    p1 = plots.loss_curves(
        result.train_loss,
        result.val_loss,
        result.best_epoch,
        title="seed 72 — train keeps falling, validation does not",
    )
    p2 = plots.test_bars(
        result.test_loss_best,
        result.test_loss_final,
        title="seed 72 — what the test slice says",
    )

    print("\n4) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 72 done.")


if __name__ == "__main__":
    run_pipeline()
