"""
main.py
The main menu for the Customer Churn Predictor.
This file connects data_handler.py, model_trainer.py,
predictor.py, and visualizer.py into one working program.
"""

import os
import data_handler
import model_trainer
import predictor
import visualizer

# File paths used throughout the program.
DATA_PATH = "data/customer_data.csv"
DATA_PATH_2 = "data/customer_data2.csv"
RF_MODEL_PATH = "models/random_forest.joblib"
LR_MODEL_PATH = "models/logistic_regression.joblib"
CHART_PATH = "outputs/feature_importance.png"
TARGET_COLUMN = "Churn"

# These variables hold the loaded/trained items in memory while
# the program runs, so different menu options can share them.
cleaned_data = None
X_train = X_test = y_train = y_test = None
rf_model = None
lr_model = None


def ensure_folders_exist():
    """
    Makes sure the 'models' and 'outputs' folders exist before we
    try to save anything into them. Creates them if missing.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


def view_data_summary():
    """Menu option 1: Load the CSV and show a quick summary."""
    global cleaned_data

    raw_data = data_handler.load_and_combine(DATA_PATH, DATA_PATH_2)
    cleaned_data = data_handler.clean_data(raw_data)

    print("\n--- Data Summary ---")
    print(data_handler.get_summary(cleaned_data))


def train_models():
    """Menu option 2: Train Logistic Regression and Random Forest."""
    global cleaned_data, X_train, X_test, y_train, y_test, rf_model, lr_model

    if cleaned_data is None:
        print("\nPlease load the data first (Menu option 1).")
        return

    X_train, X_test, y_train, y_test = model_trainer.split_data(
        cleaned_data, target_column=TARGET_COLUMN
    )

    print("\nTraining Logistic Regression...")
    lr_model = model_trainer.train_logistic_regression(X_train, y_train)

    print("Training Random Forest...")
    rf_model = model_trainer.train_random_forest(X_train, y_train)

    ensure_folders_exist()
    model_trainer.save_model(lr_model, LR_MODEL_PATH)
    model_trainer.save_model(rf_model, RF_MODEL_PATH)

    print("Both models trained and saved successfully.")


def evaluate_models():
    """Menu option 3: Show accuracy, precision, recall, and confusion matrix."""
    if rf_model is None or lr_model is None:
        print("\nPlease train the models first (Menu option 2).")
        return

    for name, model in [("Logistic Regression", lr_model), ("Random Forest", rf_model)]:
        results = model_trainer.evaluate_model(model, X_test, y_test)
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {results['accuracy']:.2%}")
        print(f"Precision: {results['precision']:.2%}")
        print(f"Recall:    {results['recall']:.2%}")
        print(f"Confusion Matrix:\n{results['confusion_matrix']}")


def predict_new_customer():
    """Menu option 4: Predict churn for a new customer typed in by the user."""
    if rf_model is None:
        print("\nPlease train the models first (Menu option 2).")
        return

    feature_columns = X_train.columns
    customer_df = predictor.get_customer_input(feature_columns)
    prediction, probability = predictor.predict_churn(rf_model, customer_df)
    predictor.display_prediction(prediction, probability)


def show_chart():
    """Menu option 5: Show the feature importance chart."""
    if rf_model is None:
        print("\nPlease train the models first (Menu option 2).")
        return

    ensure_folders_exist()
    visualizer.plot_feature_importance(rf_model, X_train.columns, CHART_PATH)


def show_menu():
    """Prints the main menu options."""
    print("\n=== Customer Churn Predictor ===")
    print("1. View data summary")
    print("2. Train models")
    print("3. Evaluate models")
    print("4. Predict for a new customer")
    print("5. Show chart")
    print("6. Exit")


def main():
    """Runs the menu loop until the user chooses to exit."""
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