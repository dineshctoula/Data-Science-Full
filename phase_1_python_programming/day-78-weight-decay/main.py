"""Run the Day 78 weight-decay walkthrough."""

from decay_engine import make_wide, pick_by_val, sweep
from visualizer import DecayPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 78: WEIGHT DECAY")
    print("=" * 72)

    X, y = make_wide(n=80, n_noise=40, seed=78)
    decays = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0]
    trials, sizes = sweep(X, y, decays, seed=78)
    chosen = pick_by_val(trials)

    print(f"\n1) table {X.shape}: y = 2 x0 - 1.2 x1 plus noise, 40 junk columns")
    print(f"   train {sizes[0]}, validation {sizes[1]}, test {sizes[2]}")
    print("   gradient descent, lr 0.1, 600 epochs, intercept never decayed")

    print(f"\n2) {'decay':>6}  {'train':>7}  {'val':>7}  {'test':>7}  {'junk':>6}  {'x0':>6}  {'x1':>6}")
    for t in trials:
        print(
            f"   {t.decay:6g}  {t.train_mse:7.3f}  {t.val_mse:7.3f}  {t.test_mse:7.3f}  "
            f"{t.noise_norm:6.3f}  {t.x0:+6.2f}  {t.x1:+6.2f}"
        )

    plain = trials[0]
    print(f"\n3) validation picks decay {chosen.decay:g}")
    print(f"   test MSE {chosen.test_mse:.3f} vs {plain.test_mse:.3f} with no decay")

    plots = DecayPlots()
    p1 = plots.mse_lines(trials, title="seed 78 — error across the decay grid")
    p2 = plots.weight_shrink(trials, title="seed 78 — what the decay does to the weights")

    print("\n4) plots saved")
    print("  ", p1)
    print("  ", p2)
    print("\nDay 78 done.")


if __name__ == "__main__":
    run_pipeline()
