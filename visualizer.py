"""
visualizer.py
Draws a chart showing which customer factors matter most
when predicting churn, using a trained Random Forest model.
"""

import matplotlib.pyplot as plt
import joblib
from pathlib import Path


def plot_feature_importance(model, feature_names, save_path="outputs/feature_importance.png"):
    """
    Creates a bar chart showing how important each feature (column)
    was to the Random Forest model's decisions, then saves it as an image.

    Only works with models that have a 'feature_importances_' attribute,
    which Random Forest provides (Logistic Regression does not).
    """
    if not hasattr(model, "feature_importances_"):
        print("This model does not support feature importance charts.")
        print("(Feature importance is only available for Random Forest.)")
        return

    importances = model.feature_importances_
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Pair each feature name with its importance score, then sort
    # from most important to least important, so the chart reads
    # top-to-bottom like a leaderboard.
    feature_importance_pairs = sorted(
        zip(feature_names, importances),
        key=lambda pair: pair[1],
        reverse=True,
    )

    sorted_names = [pair[0] for pair in feature_importance_pairs]
    sorted_scores = [pair[1] for pair in feature_importance_pairs]

    # Draw a horizontal bar chart. Horizontal works better than vertical
    # here because feature names can be long and hard to fit upright.
    plt.figure(figsize=(8, 6))
    plt.barh(sorted_names, sorted_scores, color="steelblue")
    plt.xlabel("Importance Score")
    plt.title("Which Factors Matter Most for Customer Churn")
    plt.gca().invert_yaxis()  # Puts the most important feature at the top
    plt.tight_layout()

    # Save the chart as an image file instead of (or in addition to)
    # just popping up on screen, so it's kept for later reference.
    plt.savefig(save_path)
    print(f"\nChart saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    model = joblib.load("random_forest.joblib")
    feature_names = list(model.feature_names_in_)
    plot_feature_importance(model, feature_names)