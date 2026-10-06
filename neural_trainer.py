"""
neural_trainer.py
Builds, trains, evaluates, and saves/loads a PyTorch neural network
for churn prediction - a third model alongside Logistic Regression
and Random Forest (model_trainer.py).

Unlike Random Forest, a neural network is sensitive to feature scale
(confirmed empirically: unscaled training produced an unstable,
bouncing loss, while scaled training converged smoothly - see
notebooks/eda.ipynb or the project README for the comparison). This
file is responsible for scaling its own inputs with StandardScaler,
kept separate from model_trainer.py since Logistic Regression and
Random Forest don't need this step.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


class ChurnNet(nn.Module):
    """
    A small feedforward neural network (a "Multi-Layer Perceptron"):
    input features -> 32 neurons -> 16 neurons -> 1 output.

    Each hidden layer is followed by ReLU, a simple function that
    passes positive values through unchanged and blocks negative
    values (sets them to 0). This non-linearity is what lets the
    network learn curved, complex patterns - without it, stacking
    multiple linear layers would mathematically collapse into just
    one big linear layer, no more powerful than Logistic Regression.

    The final layer outputs one raw number (a "logit") per customer,
    not yet a 0-1 probability - converting it to a probability is
    handled separately (see predict_proba below), paired with
    BCEWithLogitsLoss during training for better numerical stability.
    """

    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def train_neural_network(X_train, y_train, epochs=15, batch_size=512, learning_rate=0.001):
    """
    Trains a ChurnNet on the given training data. Scales the features
    first (fit only on training data, to avoid leaking test-set
    information into the scaler - the same train/test separation
    principle behind train_test_split itself).

    Returns the trained model AND the fitted scaler, since the exact
    same scaler must be reused later on any new data (test set, or a
    single new customer in predictor.py) for predictions to be valid.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype("float32"))

    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.astype("float32").values).reshape(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    torch.manual_seed(42)
    model = ChurnNet(input_size=X_tensor.shape[1])
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(loader)
        print(f"  Epoch {epoch + 1}/{epochs} - loss: {average_loss:.4f}")

    return model, scaler


def predict_proba(model, scaler, X):
    """
    Returns churn probabilities (0 to 1) for each row in X, using the
    same scaler that was fit during training.
    """
    X_scaled = scaler.transform(X.astype("float32"))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits).numpy().flatten()

    return probabilities


def evaluate_neural_network(model, scaler, X_test, y_test, threshold=0.5):
    """
    Evaluates the network the same way model_trainer.evaluate_model()
    does for the other two models, so results are directly comparable:
    accuracy, precision, recall, and a confusion matrix.
    """
    probabilities = predict_proba(model, scaler, X_test)
    predictions = (probabilities >= threshold).astype(int)

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }

    return results


def save_neural_network(model, scaler, filepath):
    """
    Saves the model's learned weights AND the fitted scaler together.
    Both are required to make valid predictions later - the scaler
    alone or the weights alone are not enough.
    """
    torch.save({
        "model_state": model.state_dict(),
        "input_size": model.layers[0].in_features,
        "scaler": scaler,
    }, filepath)


def load_neural_network(filepath):
    """Loads a previously saved ChurnNet and its matching scaler."""
    checkpoint = torch.load(filepath, weights_only=False)

    model = ChurnNet(input_size=checkpoint["input_size"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    scaler = checkpoint["scaler"]

    return model, scaler