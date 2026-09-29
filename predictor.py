"""
predictor.py
Takes details about a single new customer, typed in by the user,
and predicts whether that customer is likely to churn.
"""

import pandas as pd
import joblib


def get_customer_input(columns):
    """
    Asks the user to type in a value for each column the model needs
    (except the churn column itself, since that's what we're predicting).

    Returns a pandas DataFrame with exactly one row: the new customer.
    """
    customer_data = {}

    print("\nPlease enter the following customer details:")

    for column in columns:
        while True:
            value = input(f"  {column}: ")
            try:
                # We convert the typed text into a number,
                # since our model only understands numbers.
                customer_data[column] = float(value)
                break
            except ValueError:
                # If the user types something that isn't a number
                # (like letters), we ask again instead of crashing.
                print("  Please enter a valid number.")

    # Wrap the single customer's answers into a one-row table,
    # matching the format the model expects.
    customer_df = pd.DataFrame([customer_data])
    return customer_df


def predict_churn(model, customer_df):
    """
    Uses the trained model to predict churn for the new customer.
    Returns both the prediction (0 or 1) and the probability (0-100%).
    """
    prediction = model.predict(customer_df)[0]

    # predict_proba gives the probability of each class (stay vs churn).
    # [0][1] grabs the probability of "churn" (class 1) for our one customer.
    probability = model.predict_proba(customer_df)[0][1]

    return prediction, probability


def display_prediction(prediction, probability):
    """
    Prints the prediction result in a friendly, readable way.
    """
    probability_percent = probability * 100

    if prediction == 1:
        print(f"\nResult: This customer is LIKELY TO CHURN.")
    else:
        print(f"\nResult: This customer is LIKELY TO STAY.")

    print(f"Estimated churn probability: {probability_percent:.1f}%")


if __name__ == "__main__":
    model = joblib.load("random_forest.joblib")
    columns = list(model.feature_names_in_)
    customer_data = get_customer_input(columns)
    prediction, probability = predict_churn(model, customer_data)
    display_prediction(prediction, probability)