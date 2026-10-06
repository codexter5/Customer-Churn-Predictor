# Customer Churn Predictor

A menu-driven Python machine-learning project that predicts customer churn with Logistic Regression and Random Forest models.

## Project structure

```text
customer_churn_predictor/
|-- data/
|   |-- customer_data.csv
|   |-- customer_data2.csv
|-- notebooks/
|   |-- eda.ipynb
|-- models/
|   |-- (trained .joblib models)
|-- outputs/
|   |-- (generated charts)
|-- data_handler.py
|-- model_trainer.py
|-- predictor.py
|-- visualizer.py
|-- main.py
|-- requirements.txt
|-- README.md
```

## Setup and run

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

Use the menu to load the data, train and evaluate models, make a prediction, and save a feature-importance chart. Trained models are saved in `models/`; charts are saved in `outputs/`. The exploratory analysis is available in `notebooks/eda.ipynb` and expects to be opened with the notebook working directory set to `notebooks/`.

# Customer Churn Predictor

A terminal-based Python application that predicts whether a customer is likely to churn (stop using a service), using historical customer data and two machine learning models: Logistic Regression and Random Forest.

## Problem Statement

Churn is when a customer stops using a company's product or service. It's significantly cheaper to retain an existing customer than acquire a new one, so businesses benefit from identifying at-risk customers *before* they leave, not after. This project builds a model that estimates a customer's churn probability from their usage, billing, and support-contact history, and explains which factors drive that risk.

Because missing an actual churner is costlier than wrongly flagging a loyal customer (a false "stay" means a lost customer forever; a false "churn" only costs a wasted retention offer), this project prioritizes **recall** over raw accuracy when judging model performance.

## Dataset

Two combined CSV files, `data/customer_data.csv` (440,833 rows) and `data/customer_data2.csv` (64,374 rows), for a combined **505,206 customer records** after cleaning. Each row represents one customer:

| Column | Description |
|---|---|
| `CustomerID` | Row identifier (dropped before training — not predictive) |
| `Age` | Customer age (18–65) |
| `Gender` | Male / Female |
| `Tenure` | Months as a customer (1–60) |
| `Usage Frequency` | How often the service is used (1–30) |
| `Support Calls` | Number of support contacts (0–10) |
| `Payment Delay` | Days late on payment (0–30) |
| `Subscription Type` | Basic / Standard / Premium |
| `Contract Length` | Monthly / Quarterly / Annual |
| `Total Spend` | Total amount spent (100–1000) |
| `Last Interaction` | Days since last contact (1–30) |
| `Churn` | Target: 1 = churned, 0 = stayed |

Overall churn rate: **55.5%** (fairly balanced, not the heavily skewed ~5% churn rate seen at many real companies — see Limitations).

## Project Structure

```
customer_churn_predictor/
├── data/
│   ├── customer_data.csv
│   └── customer_data2.csv
├── notebooks/
│   └── eda.ipynb              # Exploratory analysis - see Key Findings below
├── models/                    # Trained models saved here (not committed to git)
├── outputs/                   # Generated charts saved here (not committed to git)
├── data_handler.py            # Loads, combines, and cleans the data
├── model_trainer.py           # Trains and evaluates both models
├── predictor.py                # Predicts churn for a single new customer
├── visualizer.py               # Draws the feature-importance chart
├── main.py                     # Terminal menu tying everything together
├── requirements.txt
└── README.md
```

## Setup (Windows)

```powershell
cd "path\to\customer_churn_predictor"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the Application

```powershell
.\venv\Scripts\python.exe .\main.py
```

Menu options:
1. View data summary
2. Train models
3. Evaluate models
4. Predict for a new customer
5. Show feature-importance chart
6. Exit

Run options in order (1 → 2 → 3/4/5) — options 3 onward depend on data/models loaded earlier in the same run.

When option 2 is selected, the application reports the exact number and percentage of cleaned rows used for training and testing. The current split uses 80% of the rows for training and 20% for testing.

## Exploratory Data Analysis — Key Findings

Full analysis in `notebooks/eda.ipynb`. Highlights:

- **Contract Length is the single strongest churn signal found**: Monthly-contract customers churn at **90.2%**, versus ~46% for Annual/Quarterly. This pattern is invisible on a standard correlation heatmap (see note below) but was confirmed directly via churn-rate comparison.
- **Support Calls** more than doubles for churners (2.0 → 5.3 average calls), the strongest numeric correlation found (+0.52).
- **Total Spend** is notably lower for churners (721 → 539 average), correlation −0.37.
- **Payment Delay** is higher for churners (10.4 → 16.0 days), correlation +0.33.
- **Tenure, Usage Frequency, and Subscription Type** showed negligible individual relationships with churn in every test performed.

**A documented pitfall from this analysis**: converting category columns (like Contract Length) into single numeric codes for a correlation heatmap assigns codes alphabetically, which can completely hide a real pattern — Contract Length's correlation showed as ≈0.00 despite being the dataset's strongest signal, because "Monthly" (the risky category) numerically sits *between* the two safe categories. Always cross-check a correlation heatmap against direct group comparisons for categorical data.

## Model Choices

**Logistic Regression** and **Random Forest** are both trained, for a deliberate reason: Logistic Regression is simple and interpretable (good for sanity-checking), while Random Forest captures more complex, non-linear patterns.

**One-hot encoding** is used for all three category columns rather than simple numeric codes. This was empirically tested, not assumed:

| | Label-encoded | One-hot encoded |
|---|---|---|
| Logistic Regression accuracy | 81.9% | **84.6%** |
| Random Forest accuracy | 93.5% | 93.4% (no meaningful change) |

One-hot encoding meaningfully helped Logistic Regression (which assumes a straight-line relationship per feature, and is misled by arbitrary category ordering) while making no real difference to Random Forest (which can isolate categories using multiple threshold splits regardless of their numeric order).

## Results

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Logistic Regression | 84.6% | 87.2% | 84.7% |
| **Random Forest** | **93.4%** | 89.6% | **99.6%** |

Random Forest is used as the production model for live predictions (`main.py` option 4), missing only ~204 actual churners out of over 56,000 in the test set.

**Baseline for comparison**: since 55.5% of customers churn, a model that always guesses "churn" would score 55.5% accuracy for free. Both models substantially beat this baseline.

## Known Limitations

- **This dataset shows signs of being synthetic**, not real-world customer data: Total Spend has an unnatural distribution jump around 500, and Tenure/Subscription Type show almost no relationship with churn at all — real customer data is rarely this clean. Results and feature importance should be understood as a demonstration of the pipeline, not a claim about real-world churn drivers.
- **`CustomerID` was found to correlate at −0.65 with Churn** purely due to how the dataset file was assembled (sequential numbering by batch), not real customer behavior. It is dropped before training to avoid the model learning this artifact as if it were a real signal.
- **Random Forest's default feature-importance metric is biased toward high-cardinality numeric columns** (like Age, with 48 possible values) over simple binary one-hot columns (like `Contract Length_Monthly`, with only 2). This caused Contract Length to rank only 5th in the importance chart despite having the single largest real churn-rate gap in the dataset. A fairer comparison would use permutation importance instead of the default impurity-based importance.
- **The saved Random Forest model file is ~535 MB**, since trees are allowed to grow without a depth limit. This is excluded from version control via `.gitignore`. Limiting `max_depth` would shrink this substantially with minimal accuracy cost.

## Future Improvements

- Use permutation importance for a fairer feature-importance ranking
- Limit `max_depth` on the Random Forest to reduce model file size
- Validate findings against real (non-synthetic) customer data
- Add cross-validation instead of a single train/test split
- Hyperparameter tuning (e.g. `GridSearchCV`) for both models