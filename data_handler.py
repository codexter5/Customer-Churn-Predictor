"""
data_handler.py
Loads customer data from a CSV file and cleans it up
so it's ready to be used by machine learning models.
"""

import pandas as pd


def load_data(filepath):
    """
    Reads the CSV file from disk and returns it as a pandas DataFrame.
    A DataFrame is just a table, like an Excel sheet, that Python can work with.
    """
    data = pd.read_csv(filepath)
    return data


def clean_data(data):
    """
    Cleans the raw data so it's usable for machine learning:
    - Removes columns that don't help prediction (IDs, names)
    - Converts text columns (like Gender) into numbers
    - Fills in any missing values
    """
    # Drop columns that don't help predict churn.
    # We use errors="ignore" so it won't crash if a column is already missing.
    columns_to_drop = ["CustomerID", "CustomerId", "Surname", "RowNumber"]
    data = data.drop(columns=columns_to_drop, errors="ignore")

    # Convert text columns into numeric codes.
    # Machine learning models only understand numbers, not words.
    text_columns = data.select_dtypes(include="object").columns
    for column in text_columns:
        data[column] = data[column].astype("category").cat.codes

    # Fill any missing (blank) values with the column's average.
    # This avoids errors during training, which requires complete data.
    data = data.fillna(data.mean(numeric_only=True))

    return data


def get_summary(data):
    """
    Returns a simple text summary of the dataset:
    number of rows, columns, and how many customers churned.
    """
    total_rows = len(data)
    total_columns = len(data.columns)

    # The provided customer data uses "Churn" for the target column.
    if "Churn" in data.columns:
        churned = data["Churn"].sum()
        churn_rate = (churned / total_rows) * 100
        summary = (
            f"Total customers: {total_rows}\n"
            f"Total columns: {total_columns}\n"
            f"Customers who churned: {churned} ({churn_rate:.1f}%)"
        )
    else:
        summary = f"Total customers: {total_rows}\nTotal columns: {total_columns}"

    return summary


if __name__ == "__main__":
    filepath = "churn data/customer_data2.csv"
    customer_data = load_data(filepath)
    print(get_summary(customer_data))