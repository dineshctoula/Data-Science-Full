"""Run the Day 58 ridge / lasso walkthrough."""

import numpy as np

from reg_engine import (
    coefficient_path,
    fit_lasso,
    fit_ridge,
    make_sparse_regression,
    mse,
    train_test_split,
)
from visualizer import RegPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 58: RIDGE AND LASSO")
    print("=" * 72)

    X, y, names = make_sparse_regression(n=180, seed=58)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.3, seed=58)

    print("\n1) ordinary least squares (ridge with λ=0)")
    ols = fit_ridge(Xtr, ytr, lam=0.0)
    print("  ", ols.summary())
    print("   coefs:", dict(zip(names, np.round(ols.coef, 3))))
    print("   test MSE:", round(mse(yte, ols.predict(Xte)), 3))

    print("\n2) ridge vs lasso at a few lambdas")
    lams = [0.5, 2.0, 8.0, 25.0, 80.0]
    ridge_mse, lasso_mse = [], []
    for lam in lams:
        ridge = fit_ridge(Xtr, ytr, lam=lam)
        lasso = fit_lasso(Xtr, ytr, lam=lam)
        r_mse = mse(yte, ridge.predict(Xte))
        l_mse = mse(yte, lasso.predict(Xte))
        ridge_mse.append(r_mse)
        lasso_mse.append(l_mse)
        print(f"   λ={lam:<5} ridge MSE={r_mse:.3f} | {lasso.summary()} test MSE={l_mse:.3f}")

    # one lambda where the contrast is easy to stare at
    show = 8.0
    ridge = fit_ridge(Xtr, ytr, lam=show)
    lasso = fit_lasso(Xtr, ytr, lam=show)
    print("\n3) coefficients at λ=8")
    print(f"   {'name':<10} {'ols':>8} {'ridge':>8} {'lasso':>8}")
    for i, name in enumerate(names):
        print(f"   {name:<10} {ols.coef[i]:8.3f} {ridge.coef[i]:8.3f} {lasso.coef[i]:8.3f}")

    lasso_path = coefficient_path(Xtr, ytr, lams, kind="lasso")
    ridge_path = coefficient_path(Xtr, ytr, lams, kind="ridge")

    plots = RegPlots()
    p1 = plots.coef_path(lams, lasso_path, names, title="lasso path", filename="lasso_path.png")
    p2 = plots.coef_path(lams, ridge_path, names, title="ridge path", filename="ridge_path.png")
    p3 = plots.mse_vs_lambda(lams, ridge_mse, lasso_mse)
    p4 = plots.coef_bars(names, ols.coef, ridge.coef, lasso.coef, title="λ=8 vs OLS")

    print("\n4) plots saved")
    for p in (p1, p2, p3, p4):
        print("  ", p)
    print("\nDay 58 done.")


if __name__ == "__main__":
    run_pipeline()
