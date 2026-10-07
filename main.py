"""
main.py
The main menu for the Customer Churn Predictor.
"""

import os

import joblib
import pandas as pd

import data_handler
import model_trainer
import neural_trainer
import predictor
import visualizer

DATA_PATH = "data/customer_data.csv"
DATA_PATH_2 = "data/customer_data2.csv"
RF_MODEL_PATH = "models/random_forest.joblib"
LR_MODEL_PATH = "models/logistic_regression.joblib"
NN_MODEL_PATH = "models/neural_network.pt"
FEATURE_COLUMNS_PATH = "models/feature_columns.joblib"
TEST_DATA_PATH = "models/test_data.joblib"
CHART_PATH = "outputs/feature_importance.png"
TARGET_COLUMN = "Churn"

cleaned_data = None
X_train = X_test = y_train = y_test = None
rf_model = None
lr_model = None
nn_model = None
nn_scaler = None
# The exact list of input columns the models were trained on, in order.
# Needed by predictor.py and visualizer.py even when a user loads saved
# models without rerunning the full data pipeline (so X_train itself
# may not exist in memory).
feature_columns = None


def ensure_folders_exist():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


def view_data_summary():
    global cleaned_data
    if not os.path.exists(DATA_PATH) or not os.path.exists(DATA_PATH_2):
        print(f"\nCould not find the data files.")
        print(f"Expected to find:")
        print(f"  {DATA_PATH}")
        print(f"  {DATA_PATH_2}")
        print("Make sure you're running this program from inside the")
        print("project folder, and that both CSV files are in 'data/'.")
        return
    raw_data = data_handler.load_and_combine(DATA_PATH, DATA_PATH_2)
    cleaned_data = data_handler.clean_data(raw_data, target_column=TARGET_COLUMN)
    print("\n--- Data Summary ---")
    print(data_handler.get_summary(cleaned_data, target_column=TARGET_COLUMN))


def saved_models_exist():
    """
    Checks whether every file needed to skip retraining is present:
    all three models, the feature column list, and the saved test set.
    All five must exist together - a partial set (e.g. models but no
    saved test set) is treated as "not available", since evaluation
    would otherwise silently use a different test set than the one
    the models were originally scored against.
    """
    return (
        os.path.exists(LR_MODEL_PATH)
        and os.path.exists(RF_MODEL_PATH)
        and os.path.exists(NN_MODEL_PATH)
        and os.path.exists(FEATURE_COLUMNS_PATH)
        and os.path.exists(TEST_DATA_PATH)
    )


def load_saved_models():
    """
    Loads all three models, the feature column list, and the saved
    test set from disk, without retraining anything. This is what
    makes menu options 3-5 usable without ever running option 1 or
    waiting through training again.
    """
    global X_test, y_test, rf_model, lr_model, nn_model, nn_scaler, feature_columns

    lr_model = model_trainer.load_model(LR_MODEL_PATH)
    rf_model = model_trainer.load_model(RF_MODEL_PATH)
    nn_model, nn_scaler = neural_trainer.load_neural_network(NN_MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    X_test, y_test = joblib.load(TEST_DATA_PATH)

    print("Loaded previously saved models - training skipped.")


def train_models():
    global X_train, X_test, y_train, y_test, rf_model, lr_model, nn_model, nn_scaler, feature_columns

    if saved_models_exist():
        answer = input(
            "\nFound previously saved models. Load them instead of "
            "retraining? (y/n): "
        ).strip().lower()
        if answer == "y":
            load_saved_models()
            return
        print("Retraining from scratch - this will overwrite the saved models.")

    if cleaned_data is None:
        print("\nPlease load the data first (Menu option 1).")
        return

    X_train, X_test, y_train, y_test = model_trainer.split_data(cleaned_data, target_column=TARGET_COLUMN)
    feature_columns = X_train.columns.tolist()

    print("\nTraining Logistic Regression...")
    lr_model = model_trainer.train_logistic_regression(X_train, y_train)
    print("Training Random Forest...")
    rf_model = model_trainer.train_random_forest(X_train, y_train)
    print("Training Neural Network (shows progress per epoch)...")
    nn_model, nn_scaler = neural_trainer.train_neural_network(X_train, y_train)

    ensure_folders_exist()
    model_trainer.save_model(lr_model, LR_MODEL_PATH)
    model_trainer.save_model(rf_model, RF_MODEL_PATH)
    neural_trainer.save_neural_network(nn_model, nn_scaler, NN_MODEL_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)
    joblib.dump((X_test, y_test), TEST_DATA_PATH)
    print("All three models trained and saved successfully.")


def format_confusion_matrix(cm):
    """
    Turns the raw confusion matrix (a plain 2x2 grid of numbers) into
    a labeled table, so it's clear which number means what without
    having to remember the row/column order yourself.
    """
    labels = ["Actual: Stayed", "Actual: Churned"]
    columns = ["Predicted: Stayed", "Predicted: Churned"]
    return pd.DataFrame(cm, index=labels, columns=columns)


def evaluate_models():
    if rf_model is None or lr_model is None or nn_model is None:
        print("\nPlease train the models first (Menu option 2).")
        return

    # neural_trainer.evaluate_neural_network() returns the exact same
    # dict shape (accuracy, precision, recall, confusion_matrix) as
    # model_trainer.evaluate_model(), so all three models can be
    # printed through one shared loop.
    all_results = [
        ("Logistic Regression", model_trainer.evaluate_model(lr_model, X_test, y_test)),
        ("Random Forest", model_trainer.evaluate_model(rf_model, X_test, y_test)),
        ("Neural Network (PyTorch)", neural_trainer.evaluate_neural_network(nn_model, nn_scaler, X_test, y_test)),
    ]

    for name, results in all_results:
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {results['accuracy']:.2%}")
        print(f"Precision: {results['precision']:.2%}")
        print(f"Recall:    {results['recall']:.2%}")
        print("Confusion Matrix:")
        print(format_confusion_matrix(results["confusion_matrix"]))


def predict_new_customer():
    if rf_model is None or lr_model is None or nn_model is None:
        print("\nPlease train the models first (Menu option 2).")
        return

    print("\nWhich model would you like to use for this prediction?")
    print("  1. Logistic Regression")
    print("  2. Random Forest")
    print("  3. Neural Network (PyTorch)")
    model_choice = input("Choose an option (1-3): ").strip()

    if model_choice not in ("1", "2", "3"):
        print("\nInvalid choice. Returning to menu.")
        return

    customer_df = predictor.get_customer_input(feature_columns)

    if model_choice == "1":
        prediction, probability = predictor.predict_churn(lr_model, customer_df)
    elif model_choice == "2":
        prediction, probability = predictor.predict_churn(rf_model, customer_df)
    else:
        # The neural network isn't a scikit-learn model, so it doesn't
        # have .predict()/.predict_proba() - we use neural_trainer's
        # own prediction function instead, which also applies the same
        # scaling the model was trained with.
        probability = neural_trainer.predict_proba(nn_model, nn_scaler, customer_df)[0]
        prediction = int(probability >= 0.5)

    predictor.display_prediction(prediction, probability)


def show_chart():
    if rf_model is None:
        print("\nPlease train the models first (Menu option 2).")
        return
    ensure_folders_exist()
    visualizer.plot_feature_importance(rf_model, feature_columns, CHART_PATH)


def show_menu():
    print("\n=== Customer Churn Predictor ===")
    print("1. View data summary")
    print("2. Train models")
    print("3. Evaluate models")
    print("4. Predict for a new customer")
    print("5. Show chart")
    print("6. Exit")


def main():
    while True:
        show_menu()
        choice = input("Choose an option (1-6): ")
        if choice == "1":
            view_data_summary()
        elif choice == "2":
            train_models()
        elif choice == "3":
            evaluate_models()
        elif choice == "4":
            predict_new_customer()
        elif choice == "5":
            show_chart()
        elif choice == "6":
            print("Goodbye!")
            break
        else:
            print("\nInvalid choice. Please enter a number from 1 to 6.")


if __name__ == "__main__":
    main()