
"""
model_trainer.py
Trains Logistic Regression and Random Forest models on the cleaned
churn data, and evaluates how well each one predicts churn.
"""

import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def split_data(data, target_column="Churn"):
    """
    Splits the cleaned data into inputs (X) and the target (y), then
    into an 70% training portion and a 30% testing portion - so we can
    fairly check the model on data it has never seen before (Phase 8).
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """
    Trains Logistic Regression - our simple, interpretable baseline
    model (Phase 8), now benefiting from one-hot encoded categories.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Trains Random Forest - our stronger "production" model (Phase 8),
    built from 100 decision trees voting together.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Tests a trained model against unseen data and returns accuracy,
    precision, recall, and a confusion matrix. We prioritize recall
    (Phase 1) since missing a real churner is costlier than wrongly
    flagging a loyal customer.
    """
    predictions = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }

    return results


def save_model(model, filepath):
    """Saves a trained model to disk so we don't retrain every run."""
    joblib.dump(model, filepath)


def load_model(filepath):
    """Loads a previously saved model back from disk."""
    model = joblib.load(filepath)
    return model