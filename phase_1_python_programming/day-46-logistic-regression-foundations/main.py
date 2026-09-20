"""Runnable learning pipeline for Day 46: logistic regression foundations."""

import numpy as np

from logistic_engine import LogisticEngine
from visualizer import LogisticVisualizer


def run_pipeline() -> None:
    print("=" * 80)
    print("DAY 46: LOGISTIC REGRESSION FOUNDATIONS")
    print("=" * 80)

    # ---- 1. Sigmoid and logit ----------------------------------------------
    scores = np.array([-2.0, 0.0, 2.0])
    probs = LogisticEngine.sigmoid(scores)
    print("\n1. Sigmoid turns linear scores into probabilities")
    for score, probability in zip(scores, probs):
        print(f"  σ({score: .1f}) = {probability:.4f}")
    print(f"  logit(σ(z)) recovers z: {np.round(LogisticEngine.logit(probs), 4)}")

    # ---- 2. One-feature pass/fail model ------------------------------------
    hours, labels_1d = LogisticEngine.generate_one_feature_pass_data()
    one_feature = LogisticEngine.fit(
        hours, labels_1d, ("study_hours",), learning_rate=0.3, iterations=2_500
    )
    print("\n2. One-feature logistic model (study hours → pass)")
    print(f"  {one_feature.summary()}")
    print(f"  Loss: {one_feature.loss_history[0]:.4f} → {one_feature.final_loss:.4f}")

    # ---- 3. Two-feature exam model -----------------------------------------
    features, labels, names = LogisticEngine.generate_exam_pass_data()
    model = LogisticEngine.fit(features, labels, names, learning_rate=0.2, iterations=3_000)
    print("\n3. Two-feature model (study hours + GPA)")
    print(f"  n = {model.n_observations}")
    print(f"  {model.summary()}")
    for name, slope in zip(names, model.slopes):
        print(f"  slope({name}) = {slope:.4f}")

    # ---- 4. Confusion matrix -----------------------------------------------
    matrix = LogisticEngine.confusion_matrix(labels, model.predictions)
    print("\n4. Confusion matrix [[TN, FP], [FN, TP]]")
    print(f"  {matrix.astype(int).tolist()}")
    print(
        f"  accuracy = {model.accuracy:.3f}, precision = {model.precision:.3f}, "
        f"recall = {model.recall:.3f}"
    )

    # ---- 5. Probability predictions ----------------------------------------
    new_students = np.array([[2.0, 2.0], [6.0, 3.2], [10.0, 3.8]])
    new_probs = LogisticEngine.predict_proba(new_students, model.coefficients)
    new_preds = LogisticEngine.predict(new_students, model.coefficients)
    print("\n5. Predictions for three new students")
    for row, probability, prediction in zip(new_students, new_probs, new_preds):
        decision = "pass" if prediction == 1.0 else "fail"
        print(
            f"  {row[0]:.0f} hours, GPA {row[1]:.1f} → "
            f"P(pass) = {probability:.3f} → {decision}"
        )

    # ---- 6. Charts ---------------------------------------------------------
    visualizer = LogisticVisualizer()
    sigmoid_image = visualizer.plot_sigmoid()
    curve_image = visualizer.plot_probability_curve(
        hours, labels_1d, one_feature, title="Pass probability vs study hours"
    )
    loss_image = visualizer.plot_loss_history(model, title="Two-feature training loss")
    confusion_image = visualizer.plot_confusion_matrix(matrix, title="Exam-pass confusion matrix")
    print("\n6. Generated visual explanations")
    print(f"  Sigmoid curve:      {sigmoid_image}")
    print(f"  Probability curve:  {curve_image}")
    print(f"  Loss history:       {loss_image}")
    print(f"  Confusion matrix:   {confusion_image}")
    print("\nDay 46 pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
