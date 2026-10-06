"""
data_handler.py
Loads and cleans the combined customer churn dataset, and generates
a quick summary. Every cleaning decision here was tested and verified
in notebooks/eda.ipynb (Phases 6-8) before being written here.
"""

import pandas as pd


def load_and_combine(filepath_1, filepath_2):
    """
    Reads both customer CSV files and stacks them into one larger
    DataFrame. CustomerID is NOT unique across the combined data (each
    file numbers its own customers independently), but this is handled
    safely since clean_data() drops CustomerID entirely before training.
    """
    data_1 = pd.read_csv(filepath_1)
    data_2 = pd.read_csv(filepath_2)
    combined = pd.concat([data_1, data_2], ignore_index=True)
    return combined


def clean_data(data, target_column="Churn"):
    """
    Cleans the raw combined data so it's ready for machine learning.

    Steps, each justified by something found during EDA:
    1. Drop CustomerID - confirmed in Phase 7 to have a strong
       accidental correlation with Churn (-0.65) caused by how the
       dataset file was assembled, not real customer behavior.
    2. Drop rows missing the target - filling a missing Churn value
       with the column average would fabricate a fake outcome.
    3. One-hot encode text columns - confirmed in Phase 8 to improve
       Logistic Regression by +2.7 accuracy points, with no real cost
       to Random Forest, by removing a misleading alphabetical
       ordering (e.g. Contract Length: Annual=0, Monthly=1, Quarterly=2,
       even though Monthly is the riskiest category).
    """
    data = data.drop(columns=["CustomerID"], errors="ignore")

    if target_column in data.columns:
        data = data.dropna(subset=[target_column])

    # Select text columns using both "object" and "str", since pandas
    # 3.0+ stores text as a dedicated "str" dtype rather than "object".
    text_columns = data.select_dtypes(include=["object", "str"]).columns.tolist()

    # Don't one-hot encode the target column itself, even if it were
    # somehow read in as text.
    text_columns = [col for col in text_columns if col != target_column]

    data = pd.get_dummies(data, columns=text_columns)

    return data


def get_summary(data, target_column="Churn"):
    """
    Returns a simple text summary of the dataset: row/column counts
    and the churn balance.
    """
    total_rows = len(data)
    total_columns = len(data.columns)

    if target_column in data.columns:
        churned = data[target_column].sum()
        churn_rate = (churned / total_rows) * 100
        summary = (
            f"Total customers: {total_rows}\n"
            f"Total columns: {total_columns}\n"
            f"Customers who churned: {int(churned)} ({churn_rate:.1f}%)"
        )
    else:
        summary = f"Total customers: {total_rows}\nTotal columns: {total_columns}"

    return summary