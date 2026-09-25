"""Run the Day 53 linear SVM walkthrough."""

from svm_engine import LinearSVM, make_linearly_separable, make_soft_overlap, train_test_split
from visualizer import SVMPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 53: SUPPORT VECTOR MACHINES (linear)")
    print("=" * 72)

    X, y = make_linearly_separable(n_per=45, seed=53)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.3, seed=53)

    print("\n1) easy separable blobs")
    model = LinearSVM(C=1.0, lr=0.05, n_epochs=50, seed=53).fit(Xtr, ytr)
    print("  ", model.score(Xte, yte).summary())
    print("   ||w|| =", round(float((model.w_ ** 2).sum() ** 0.5), 3))

    print("\n2) C sweep on overlapping blobs")
    Xo, yo = make_soft_overlap(n_per=55, seed=53)
    Xtr_o, Xte_o, ytr_o, yte_o = train_test_split(Xo, yo, test_frac=0.3, seed=54)
    sweep = {}
    for C in (0.1, 0.5, 1.0, 5.0, 20.0):
        m = LinearSVM(C=C, lr=0.05, n_epochs=55, seed=54).fit(Xtr_o, ytr_o)
        acc = m.score(Xte_o, yte_o).accuracy
        sweep[C] = acc
        print(f"   C={C:<4}: test acc={acc:.3f}, margin pts≈{m.score(Xtr_o, ytr_o).n_sv_approx}")

    soft = LinearSVM(C=1.0, lr=0.05, n_epochs=55, seed=54).fit(Xtr_o, ytr_o)

    plots = SVMPlots()
    p1 = plots.decision_boundary(model, X, y, title="separable case", filename="svm_separable.png")
    p2 = plots.decision_boundary(soft, Xo, yo, title="soft overlap", filename="svm_overlap.png")
    p3 = plots.loss_curve(model.loss_curve_, title="separable: objective vs epoch")
    p4 = plots.c_sweep_bars(sweep, title="overlap data: accuracy vs C")

    print("\n3) plots saved")
    for p in (p1, p2, p3, p4):
        print("  ", p)
    print("\nDay 53 done.")


if __name__ == "__main__":
    run_pipeline()
