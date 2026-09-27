"""Run the Day 56 PCA walkthrough."""

import numpy as np

from pca_engine import PCA, make_noisy_signal, make_stretched_cloud
from visualizer import PCAPlots


def run_pipeline():
    print("=" * 72)
    print("DAY 56: PRINCIPAL COMPONENT ANALYSIS")
    print("=" * 72)

    X = make_stretched_cloud(n=180, seed=56)
    print("\n1) stretched 2D cloud")
    pca2 = PCA(n_components=2).fit(X)
    print("  ", pca2.result(X).summary())
    print("   variance ratios:", np.round(pca2.explained_variance_ratio_, 3))

    print("\n2) drop to 1 component and see the blur")
    pca1 = PCA(n_components=1).fit(X)
    print("  ", pca1.result(X).summary())
    hat = pca1.inverse_transform(pca1.transform(X))

    print("\n3) 6-D data, mostly 2-D signal")
    Xw = make_noisy_signal(n=160, p=6, seed=56)
    wide = PCA(n_components=6).fit(Xw)
    print("   ratios:", np.round(wide.explained_variance_ratio_, 3))
    cum = np.cumsum(wide.explained_variance_ratio_)
    print("   cumulative:", np.round(cum, 3))
    for k in (1, 2, 3):
        m = PCA(n_components=k).fit(Xw)
        print(f"   k={k}: recon MSE={m.reconstruction_mse(Xw):.4f}")

    plots = PCAPlots()
    p1 = plots.stretched_cloud(pca2, X, title="stretched cloud + PC axes")
    p2 = plots.scree(wide.explained_variance_ratio_, title="6-D signal: scree")
    p3 = plots.recon_scatter(X, hat, title="1-PC reconstruction of the cloud")

    print("\n4) plots saved")
    for p in (p1, p2, p3):
        print("  ", p)
    print("\nDay 56 done.")


if __name__ == "__main__":
    run_pipeline()
