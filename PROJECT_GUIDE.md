# Customer Churn Predictor

## 1. Project Overview

This project predicts whether a customer is likely to churn. It is a small Python machine-learning application with:

- CSV data loading and cleaning
- Logistic Regression and Random Forest model training
- Model evaluation using accuracy, precision, recall, and a confusion matrix
- Interactive churn prediction for one new customer
- Random Forest feature-importance visualization
- A menu-driven entry point that connects the complete workflow

The current application uses `data/customer_data.csv` as its default dataset and predicts the `Churn` column.

## 2. Current Project Structure

```text
Customer Churn Predictor/
|
|-- data/
|   |-- customer_data.csv        # Default dataset used by main.py and model_trainer.py
|   |-- customer_data2.csv       # Alternate dataset for summaries/manual experiments
|
|-- data_handler.py              # CSV loading, cleaning, and summaries
|-- model_trainer.py             # Splitting, training, evaluation, and model serialization
|-- predictor.py                 # Interactive single-customer prediction
|-- visualizer.py                # Random Forest feature-importance chart
|-- main.py                      # Complete interactive menu application
|-- requirements.txt             # Pinned Python dependencies
|-- models/
|   |-- random_forest.joblib     # Saved Random Forest model
|   |-- logistic_regression.joblib # Saved Logistic Regression model
|-- outputs/
|   |-- feature_importance.png   # Generated chart
|-- venv/                        # Existing Python virtual environment
|-- README.md                    # Currently empty
|-- PROJECT_GUIDE.md             # This document
```

Generated files such as `.joblib` models, `outputs/`, `models/`, `__pycache__/`, and `venv/` should generally not be committed to source control unless the project requirements explicitly call for them.

## 3. Environment Setup on Windows

Open PowerShell in the project directory:

```powershell
cd "C:\Users\ACER\Documents\SoftwareEngineering\Customer Churn Predictor"
```

The repository currently contains a virtual environment. Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```powershell
python -m pip install -r .\requirements.txt
```

If PowerShell does not allow activation, run commands through the environment directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

## 4. How to Run the Application

The recommended entry point is `main.py`:

```powershell
.\venv\Scripts\python.exe .\main.py
```

The menu options are:

```text
1. View data summary
2. Train models
3. Evaluate models
4. Predict for a new customer
5. Show chart
6. Exit
```

Recommended workflow:

1. Select `1` to load and clean the dataset.
2. Select `2` to train both models and save them.
3. Select `3` to evaluate both models.
4. Select `4` to enter a new customer's feature values and predict churn.
5. Select `5` to save and display the feature-importance chart.
6. Select `6` to exit.

Options `2` through `5` depend on the in-memory models created during the current run. Start with option `1` after launching the program.

## 5. Running Individual Modules

### Data summary

```powershell
.\venv\Scripts\python.exe .\data_handler.py
```

The standalone data handler currently summarizes `customer_data2.csv`. Change the filepath in its `__main__` block if another file should be summarized.

### Model training

```powershell
.\venv\Scripts\python.exe .\model_trainer.py
```

This loads `customer_data.csv`, cleans it, trains both models, prints evaluation metrics, and writes:

- `logistic_regression.joblib`
- `random_forest.joblib`

### Single-customer prediction

```powershell
.\venv\Scripts\python.exe .\predictor.py
```

This loads `random_forest.joblib` and prompts for one numeric value for every model feature.

### Feature-importance chart

```powershell
.\venv\Scripts\python.exe .\visualizer.py
```

This loads `random_forest.joblib` and writes `outputs/feature_importance.png`. Matplotlib may also open a chart window depending on the Python environment.

## 6. Data Contract

The current CSV files have these columns:

```text
CustomerID, Age, Gender, Tenure, Usage Frequency, Support Calls,
Payment Delay, Subscription Type, Contract Length, Total Spend,
Last Interaction, Churn
```

Meaning of the columns:

| Column | Role | Notes |
|---|---|---|
| `CustomerID` | Identifier | Dropped before training |
| `Age` | Feature | Numeric |
| `Gender` | Feature | Text, converted to category codes |
| `Tenure` | Feature | Numeric |
| `Usage Frequency` | Feature | Numeric |
| `Support Calls` | Feature | Numeric |
| `Payment Delay` | Feature | Numeric |
| `Subscription Type` | Feature | Text, converted to category codes |
| `Contract Length` | Feature | Text, converted to category codes |
| `Total Spend` | Feature | Numeric |
| `Last Interaction` | Feature | Numeric |
| `Churn` | Target | Binary target: `0` means stay and `1` means churn |

After cleaning, the model has 10 input features. `CustomerID` is removed and `Churn` is separated as the target.

## 7. Data Preparation Behavior

`data_handler.clean_data()` currently performs these operations:

1. Drops `CustomerID`, `CustomerId`, `Surname`, and `RowNumber` when present.
2. Converts every remaining object/text column to pandas categorical integer codes.
3. Fills missing numeric values with each column's mean.

Important implications:

- New prediction values must be numeric because the models do not accept raw text.
- The categorical encoding is created directly from the DataFrame. If category labels or category ordering change, the numeric meaning can change.
- For production-quality predictions, replace the manual category conversion with a persisted preprocessing pipeline, such as `sklearn.compose.ColumnTransformer` and `OneHotEncoder`.
- The target is converted to integers in `model_trainer.split_data()` so classifiers receive discrete labels.
- Missing target values can make the summary display a fractional churn count because `clean_data()` fills every numeric column, including `Churn`, with a mean. A future cleanup should handle the target separately and avoid imputing missing labels.

## 8. Model Training and Evaluation

`model_trainer.py` provides these functions:

- `split_data(data, target_column="Churn")`: creates an 80/20 train/test split with `random_state=42`.
- `train_logistic_regression(X_train, y_train)`: trains Logistic Regression with `max_iter=1000`.
- `train_random_forest(X_train, y_train)`: trains Random Forest with 100 trees and `random_state=42`.
- `evaluate_model(model, X_test, y_test)`: returns accuracy, precision, recall, and a confusion matrix.
- `save_model(model, filepath)`: serializes a model with `joblib`.
- `load_model(filepath)`: loads a serialized model.

A previous successful run on the current dataset produced approximately:

```text
Logistic Regression: accuracy=0.853, precision=0.883, recall=0.854
Random Forest:       accuracy=1.000, precision=1.000, recall=0.999
```

These are reference results, not guaranteed benchmarks. The unusually high Random Forest result should be investigated before treating it as evidence of production performance. Use a separate validation set, cross-validation, and checks for data leakage.

## 9. Saved Model Contracts

The current model files are:

```text
random_forest.joblib
logistic_regression.joblib
```

They are saved by `model_trainer.py` in the `models/` directory. `main.py` uses the same paths.

The saved Random Forest stores the feature names in `model.feature_names_in_`. `predictor.py` uses those names to determine which fields to request. If the training features change, retrain the models before running the predictor.

Do not load a model trained with a different feature order or preprocessing scheme. A model file and its preprocessing assumptions must be updated together.

## 10. Module Responsibilities

### `data_handler.py`

Owns CSV loading, cleaning, and summary generation. It should remain the single place where raw customer data is converted into model-ready data.

### `model_trainer.py`

Owns train/test splitting, model construction, evaluation, and serialization. It can be run directly for a quick training pass.

### `predictor.py`

Owns interactive input, prediction, probability display, and direct loading of the saved Random Forest when run by itself.

### `visualizer.py`

Owns the Random Forest feature-importance chart. It requires a model with `feature_importances_`.

### `main.py`

Owns the interactive workflow and in-memory state. It imports the other modules and coordinates their functions. It is the preferred entry point for a user.

## 11. Important Configuration Values

The active paths and target are defined near the top of `main.py`:

```python
DATA_PATH = "data/customer_data.csv"
RF_MODEL_PATH = "models/random_forest.joblib"
LR_MODEL_PATH = "models/logistic_regression.joblib"
CHART_PATH = "outputs/feature_importance.png"
TARGET_COLUMN = "Churn"
```

If the dataset, target column, model location, or output location changes, update all related standalone entry points as well:

- `main.py`
- `model_trainer.py`
- `predictor.py`
- `visualizer.py`
- `data_handler.py` if its summary input should change

## 12. Dependencies

The dependency file is named `requirements.txt` and contains pinned versions for pandas, NumPy, scikit-learn, joblib, Matplotlib, SciPy, and their supporting packages.

The most important runtime packages are:

- `pandas`
- `scikit-learn`
- `joblib`
- `matplotlib`

Use the existing `venv` interpreter when running the project. The system Python previously failed to import `joblib`, while the project virtual environment worked.

## 13. Common Problems and Fixes

### `ModuleNotFoundError`

Use the project interpreter instead of the system interpreter:

```powershell
.\venv\Scripts\python.exe .\main.py
```

If the package is not installed:

```powershell
.\venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

### Model file not found

Run training first:

```powershell
.\venv\Scripts\python.exe .\model_trainer.py
```

### Prediction input rejected

Enter numeric values only. Text categories currently need to be entered as their encoded numeric values, not as words such as `Male`, `Basic`, or `Monthly`.

### Menu options say that models are not trained

Within `main.py`, select option `1` first, then option `2`. The menu stores the loaded data and trained models only for the current process.

### Wrong file or target column

Check `DATA_PATH` and `TARGET_COLUMN` in `main.py`, and check the standalone file paths in the `__main__` blocks of the other modules.

## 14. Recommended Future Improvements

1. Replace manual categorical codes with a persisted preprocessing pipeline.
2. Handle the target column separately so missing labels are not mean-imputed.
3. Move all configuration into one file or command-line arguments.
4. Make `main.py` load existing models when available instead of requiring retraining every launch.
5. Add automated tests for loading, cleaning, splitting, prediction, and evaluation.
6. Add a real validation strategy and investigate the near-perfect Random Forest result.
9. Add `.gitignore` entries for `venv/`, `__pycache__/`, generated model files, and generated charts.
10. Add command-line arguments for selecting the dataset, model, and output paths.

## 15. Quick Reference

From the project directory:

```powershell
# Launch the full menu application
.\venv\Scripts\python.exe .\main.py

# Train models directly
.\venv\Scripts\python.exe .\model_trainer.py

# Predict one customer directly
.\venv\Scripts\python.exe .\predictor.py

# Generate the feature-importance chart directly
.\venv\Scripts\python.exe .\visualizer.py
```
