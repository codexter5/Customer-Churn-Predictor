
"""
visualizer.py
Draws a chart showing which customer factors matter most when
predicting churn, using a trained Random Forest model.
"""

import matplotlib.pyplot as plt


def plot_feature_importance(model, feature_names, save_path="outputs/feature_importance.png"):
    """
    Creates a bar chart showing how important each feature (column)
    was to the Random Forest model's decisions, then saves it.

    Only works with models that have 'feature_importances_', which
    Random Forest provides (Logistic Regression does not).
    """
    if not hasattr(model, "feature_importances_"):
        print("This model does not support feature importance charts.")
        print("(Feature importance is only available for Random Forest.)")
        return

    importances = model.feature_importances_

    feature_importance_pairs = sorted(
        zip(feature_names, importances),
        key=lambda pair: pair[1],
        reverse=True,
    )
    sorted_names = [pair[0] for pair in feature_importance_pairs]
    sorted_scores = [pair[1] for pair in feature_importance_pairs]

    plt.figure(figsize=(9, 7))
    plt.barh(sorted_names, sorted_scores, color="steelblue")
    plt.xlabel("Importance Score")
    plt.title("Which Factors Matter Most for Customer Churn")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"\nChart saved to: {save_path}")
    plt.show()