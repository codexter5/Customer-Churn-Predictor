# Customer Churn Predictor

A menu-driven Python machine-learning project that predicts customer churn with Logistic Regression and Random Forest models.

## Project structure

```text
customer_churn_predictor/
|-- data/customer_data.csv
|-- models/
|-- outputs/
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

Use the menu to load the data, train and evaluate models, make a prediction, and save a feature-importance chart. Trained models are saved in `models/`; charts are saved in `outputs/`.