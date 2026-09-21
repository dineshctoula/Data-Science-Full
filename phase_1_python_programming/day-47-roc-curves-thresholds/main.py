"""Runnable learning pipeline for Day 47: ROC curves and thresholds."""

import numpy as np

from roc_engine import RocEngine
from visualizer import RocVisualizer


def run_pipeline() -> None:
    print("=" * 80)
    print("DAY 47: ROC CURVES & CLASSIFICATION THRESHOLDS")
    print("=" * 80)

    # ---- 1. Useful vs random ranking ---------------------------------------
    labels, scores = RocEngine.generate_scored_labels(n=200, seed=47, separation=1.6)
    curve = RocEngine.build_curves(labels, scores)
    poor_labels, poor_scores = RocEngine.generate_poor_ranking(n=200, seed=47)
    poor_curve = RocEngine.build_curves(poor_labels, poor_scores)
    print("\n1. Ranking quality summarized by ROC AUC")
    print(f"  Useful class-conditional scores: AUC = {curve.auc_roc:.4f}")
    print(f"  Uninformative random scores:     AUC = {poor_curve.auc_roc:.4f}")
    print("  (0.5 ≈ chance; closer to 1.0 means positives rank above negatives)")

    # ---- 2. Default threshold 0.5 ------------------------------------------
    at_half = curve.at_threshold(0.5)
    print("\n2. Operating point at the common 0.5 cutoff")
    print(f"  {at_half.summary()}")

    # ---- 3. Better thresholds ----------------------------------------------
    youden = curve.best_youden_threshold()
    best_f1 = curve.best_f1_threshold()
    print("\n3. Thresholds chosen by Youden's J and by F1")
    print(f"  Youden: {youden.summary()}")
    print(f"  F1:     {best_f1.summary()}")

    # ---- 4. Confusion counts at the Youden point ---------------------------
    print("\n4. Confusion counts at the Youden threshold")
    print(
        f"  TP = {youden.true_positive:.0f}, FP = {youden.false_positive:.0f}, "
        f"TN = {youden.true_negative:.0f}, FN = {youden.false_negative:.0f}"
    )

    # ---- 5. Hand-check AUC on a tiny perfect ranking -----------------------
    tiny_labels = np.array([0.0, 0.0, 1.0, 1.0])
    tiny_scores = np.array([0.1, 0.2, 0.8, 0.9])
    tiny_auc = RocEngine.build_curves(tiny_labels, tiny_scores).auc_roc
    print("\n5. Perfect tiny ranking should have AUC = 1")
    print(f"  AUC = {tiny_auc:.4f}")

    # ---- 6. Charts ---------------------------------------------------------
    visualizer = RocVisualizer()
    roc_image = visualizer.plot_roc_curve(curve, youden, title="ROC for useful ranking")
    pr_image = visualizer.plot_precision_recall(curve, best_f1, title="Precision–recall curve")
    tradeoff_image = visualizer.plot_threshold_tradeoff(curve)
    auc_image = visualizer.plot_auc_comparison(
        {"Useful ranking": curve.auc_roc, "Random scores": poor_curve.auc_roc}
    )
    print("\n6. Generated visual explanations")
    print(f"  ROC curve:            {roc_image}")
    print(f"  Precision–recall:     {pr_image}")
    print(f"  Threshold tradeoff:   {tradeoff_image}")
    print(f"  AUC comparison:       {auc_image}")
    print("\nDay 47 pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
