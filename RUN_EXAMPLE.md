# Customer Churn Predictor: Example Run

This example shows how to run the project on Windows with PowerShell. Run every command from the project root:

```text
C:\Users\ACER\Documents\SoftwareEngineering\Customer Churn Predictor
```

## 1. Prepare the Environment

If the virtual environment already exists, activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If it does not exist yet, create it first:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r .\requirements.txt
```

You can confirm that the project interpreter is being used:

```powershell
python --version
python -c "import pandas, sklearn, joblib, matplotlib; print('Dependencies are available.')"
```

If PowerShell activation is unavailable, use the interpreter directly in every command:

```powershell
.\venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

## 2. Start the Menu Application

Run the recommended entry point:

```powershell
python .\main.py
```

The application displays:

```text
=== Customer Churn Predictor ===
1. View data summary
2. Train models
3. Evaluate models
4. Predict for a new customer
5. Show chart
6. Exit
Choose an option (1-6):
```

## 3. Recommended Example Session

### Step 1: Load and summarize the data

Enter `1`:

```text
Choose an option (1-6): 1

--- Data Summary ---
Total customers: <number of combined rows>
Total columns: 11
Customers who churned: <number> (<percentage>%)
```

The menu workflow combines these files before cleaning:

```text
data/customer_data.csv
data/customer_data2.csv
```

### Step 2: Train both models

Enter `2`:

```text
Choose an option (1-6): 2

Training Logistic Regression...
Training Random Forest...
Both models trained and saved successfully.
```

This creates or replaces:

```text
models/logistic_regression.joblib
models/random_forest.joblib
```

### Step 3: Evaluate the models

Enter `3`:

```text
Choose an option (1-6): 3

--- Logistic Regression ---
Accuracy:  <percentage>
Precision: <percentage>
Recall:    <percentage>
Confusion Matrix:
[[... ...]
 [... ...]]

--- Random Forest ---
Accuracy:  <percentage>
Precision: <percentage>
Recall:    <percentage>
Confusion Matrix:
[[... ...]
 [... ...]]
```

The exact values depend on the current CSV contents and installed package versions. The evaluation uses the held-out 20% test split created with `random_state=42`.

### Step 4: Predict one customer

Enter `4`:

```text
Choose an option (1-6): 4

Please enter the following customer details:
  Age: 42
  Gender: 1
  Tenure: 18
  Usage Frequency: 12
  Support Calls: 3
  Payment Delay: 4
  Subscription Type: 0
  Contract Length: 1
  Total Spend: 1250
  Last Interaction: 8

Result: This customer is LIKELY TO STAY.
Estimated churn probability: <percentage>%
```

Important: enter numbers only. `Gender`, `Subscription Type`, and `Contract Length` currently use numeric category codes created during cleaning. Text such as `Male`, `Basic`, or `Monthly` will be rejected. The current application does not print the category-code mapping, so use this feature only with values whose encoding you understand.

### Step 5: Create the feature-importance chart

Enter `5`:

```text
Choose an option (1-6): 5

Chart saved to: outputs/feature_importance.png
```

The chart uses the Random Forest model and may also open in a Matplotlib window. If no window appears, the PNG should still be available in `outputs/`.

### Step 6: Exit

Enter `6`:

```text
Choose an option (1-6): 6
Goodbye!
```

## 4. Check the Generated Files

After the run, use PowerShell to inspect the artifacts:

```powershell
Get-ChildItem .\models, .\outputs
```

Expected files:

```text
models\logistic_regression.joblib
models\random_forest.joblib
outputs\feature_importance.png
```

## 5. Run Individual Modules

These commands are useful when testing one part of the project instead of the complete menu:

```powershell
# Print a summary for data/customer_data.csv
python .\data_handler.py

# Train, evaluate, and save both models using customer_data.csv
python .\model_trainer.py

# Load the saved Random Forest and predict one customer
python .\predictor.py

# Generate the Random Forest feature-importance chart
python .\visualizer.py
```

The standalone scripts use their own paths and do not all use the combined two-file workflow from `main.py`.

## 6. Validate the Python Files

Run this check after making code changes:

```powershell
python -m py_compile .\data_handler.py .\model_trainer.py .\predictor.py .\visualizer.py .\main.py
```

No output means compilation succeeded.

## 7. Regenerate the Developer Manual

The project includes a generator for the comprehensive Word manual:

```powershell
python .\generate_project_manual.py
```

This updates:

```text
PROJECT_DEVELOPER_MANUAL.docx
```

## 8. Common Ordering Mistakes

- Starting with option `2` before option `1`: load and clean the data first.
- Starting with option `3`, `4`, or `5` before training: run option `2` first.
- Running commands outside the project root: relative paths such as `data/customer_data.csv` will fail.
- Using system Python instead of the project interpreter: packages may appear to be missing.
- Entering category words during prediction: current prediction input accepts numeric values only.
- Reusing old model files after changing features or preprocessing: retrain the models first.
