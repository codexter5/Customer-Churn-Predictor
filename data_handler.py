"""
data_handler.py
Loads customer data from a CSV file and cleans it up
so it's ready to be used by machine learning models.
"""

import pandas as pd
print("pandas version:", pd.__version__)  # For debugging, to check which pandas version is running
print("numpy version:", pd.np.__version__)  # For debugging, to check which numpy version is running

def load_data(filepath):
    """
    Reads the CSV file from disk and returns it as a pandas DataFrame.
    A DataFrame is just a table, like an Excel sheet, that Python can work with.
    """
    data = pd.read_csv(filepath)
    return data


def load_and_combine(filepath_1, filepath_2):
    """
    Reads two customer CSV files and stacks them into one larger DataFrame.
    Used to combine customer_data.csv and customer_data2.csv into a single,
    bigger dataset before cleaning and training.

    Note: CustomerID is NOT unique across the combined data (each file
    numbers its own customers starting from a small number), so CustomerID
    should never be used to look up a specific real customer after this.
    It gets dropped anyway during clean_data(), so this doesn't affect training.
    """
    data_1 = pd.read_csv(filepath_1)
    data_2 = pd.read_csv(filepath_2)

    # ignore_index=True renumbers the rows 0, 1, 2... in the combined table,
    # instead of keeping two separate sets of row numbers that would clash.
    combined = pd.concat([data_1, data_2], ignore_index=True)
    return combined


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

    # Drop any row that is missing the target column (Churn).
    # We must do this BEFORE filling missing values below, otherwise a
    # blank Churn cell would get replaced with the column's average
    # (e.g. 0.57), which is not a valid churn label - churn must be 0 or 1.
    if "Churn" in data.columns:
        data = data.dropna(subset=["Churn"])

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
    filepath = "data/customer_data.csv"
    customer_data = load_data(filepath)
    print(get_summary(customer_data))