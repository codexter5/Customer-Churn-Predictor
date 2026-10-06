"""
model_trainer.py
Trains machine learning models on customer data and evaluates
how well they predict churn.
"""


import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def split_data(data, target_column="Churn"):
    """
    Splits the cleaned data into:
    - X: the input features (everything except the churn column)
    - y: the target (the churn column itself, what we want to predict)

    Then splits both into a training set (used to teach the model)
    and a testing set (used to check how well it learned).
    """
    X = data.drop(columns=[target_column])
    y = data[target_column].astype(int)

    # 80% of the data trains the model, 20% is held back to test it fairly,
    # like studying with 80% of flashcards and quizzing yourself on the rest.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model.
    This is a simple model that estimates the probability of churn
    based on a weighted combination of the input features.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest model.
    This model builds many small decision trees and combines their
    votes into one final answer, usually giving stronger accuracy
    than a single simple model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Tests a trained model against data it has never seen before,
    and returns a dictionary of performance scores.
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
    """
    Saves a trained model to a file, so we don't have to retrain
    it every single time we run the program.
    """
    joblib.dump(model, filepath)


def load_model(filepath):
    """
    Loads a previously saved model back from a file.
    """
    model = joblib.load(filepath)
    return model


if __name__ == "__main__":
    from data_handler import clean_data, load_data

    data = clean_data(load_data("data/customer_data.csv"))
    X_train, X_test, y_train, y_test = split_data(data)

    models = {
        "logistic_regression": train_logistic_regression(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train),
    }

    for name, model in models.items():
        results = evaluate_model(model, X_test, y_test)
        print(f"{name}: accuracy={results['accuracy']:.3f}, "
              f"precision={results['precision']:.3f}, "
              f"recall={results['recall']:.3f}")
        save_model(model, f"models/{name}.joblib")

# """
# model_trainer.py
# Trains Logistic Regression and Random Forest models on the cleaned
# churn data, and evaluates how well each one predicts churn.
# """

# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# def split_data(data, target_column="Churn"):
#     """
#     Splits the cleaned data into inputs (X) and the target (y), then
#     into an 80% training portion and a 20% testing portion - so we can
#     fairly check the model on data it has never seen before (Phase 8).
#     """
#     X = data.drop(columns=[target_column])
#     y = data[target_column]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     return X_train, X_test, y_train, y_test


# def train_logistic_regression(X_train, y_train):
#     """
#     Trains Logistic Regression - our simple, interpretable baseline
#     model (Phase 8), now benefiting from one-hot encoded categories.
#     """
#     model = LogisticRegression(max_iter=1000)
#     model.fit(X_train, y_train)
#     return model


# def train_random_forest(X_train, y_train):
#     """
#     Trains Random Forest - our stronger "production" model (Phase 8),
#     built from 100 decision trees voting together.
#     """
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     return model


# def evaluate_model(model, X_test, y_test):
#     """
#     Tests a trained model against unseen data and returns accuracy,
#     precision, recall, and a confusion matrix. We prioritize recall
#     (Phase 1) since missing a real churner is costlier than wrongly
#     flagging a loyal customer.
#     """
#     predictions = model.predict(X_test)

#     results = {
#         "accuracy": accuracy_score(y_test, predictions),
#         "precision": precision_score(y_test, predictions),
#         "recall": recall_score(y_test, predictions),
#         "confusion_matrix": confusion_matrix(y_test, predictions),
#     }

#     return results


# def save_model(model, filepath):
#     """Saves a trained model to disk so we don't retrain every run."""
#     joblib.dump(model, filepath)


# def load_model(filepath):
#     """Loads a previously saved model back from disk."""
#     model = joblib.load(filepath)
#     return model