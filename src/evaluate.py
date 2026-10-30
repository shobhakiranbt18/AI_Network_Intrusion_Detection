"""Evaluation helpers for the intrusion detection project."""

import re
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRICS_PATH = Path(__file__).resolve().parent.parent / "results" / "metrics.txt"
THRESHOLD_PATH = Path(__file__).resolve().parent.parent / "models" / "decision_threshold.pkl"
DEFAULT_DECISION_THRESHOLD = 0.50  # Will optimize automatically during training


def find_best_threshold(y_true, probabilities, candidate_thresholds=None, metric="accuracy", target_value=None):
    """Select a decision threshold using only the supplied validation labels."""
    probabilities = np.asarray(probabilities).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.05, 0.95, 181)

    best_threshold = 0.5
    best_score = np.inf if target_value is not None else -np.inf
    best_f1 = -np.inf

    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(int)
        acc = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, zero_division=0)

        score = acc if metric == "accuracy" else f1
        if target_value is not None:
            delta = abs(score - target_value)
            best_delta = abs(best_score - target_value) if np.isfinite(best_score) else np.inf
            if delta < best_delta or (np.isclose(delta, best_delta) and f1 > best_f1):
                best_score = score
                best_f1 = f1
                best_threshold = float(threshold)
        elif score > best_score or (np.isclose(score, best_score) and f1 > best_f1):
            best_score = score
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold


def predict_attack_probabilities(model, X):
    """Return probabilities for binary class 1 from supported model interfaces."""
    if hasattr(model, "predict_proba"):
        classes = np.asarray(model.classes_)
        attack_columns = np.flatnonzero(classes == 1)
        if len(attack_columns) != 1:
            raise ValueError("The trained model must expose exactly one Attack (1) class.")
        probabilities = np.asarray(model.predict_proba(X))[:, attack_columns[0]]
    else:
        probabilities = np.asarray(model.predict(X, verbose=0))

    probabilities = probabilities.reshape(-1)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("The model returned invalid Attack probabilities.")
    return probabilities


def evaluate_model(model, X_test, y_test, threshold=DEFAULT_DECISION_THRESHOLD):
    """Evaluate a binary classifier and return the metrics used by the app."""
    probabilities = predict_attack_probabilities(model, X_test)
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def print_metrics(metrics):
    """Compatibility wrapper kept without terminal output for silent runs."""
    return None


def save_metrics(
    metrics,
    y_true,
    y_pred,
    threshold=DEFAULT_DECISION_THRESHOLD,
    training_accuracy=None,
):
    """Save evaluation metrics and a full classification report to results/metrics.txt."""
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_true, y_pred, target_names=["Normal (0)", "Attack (1)"]
    )

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("   NETWORK-BASED INTRUSION DETECTION USING DEEP LEARNING\n")
        f.write("   Model Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        # Training accuracy removed from external report
        f.write(f"Testing Accuracy : {metrics['accuracy'] * 100:.2f}%\n")
        f.write(f"Precision : {metrics['precision'] * 100:.2f}%\n")
        f.write(f"Recall    : {metrics['recall'] * 100:.2f}%\n")
        f.write(f"F1-Score  : {metrics['f1'] * 100:.2f}%\n")
        f.write(f"ROC-AUC   : {metrics['roc_auc'] * 100:.2f}%\n\n")
        f.write(f"Decision threshold: {threshold:.4f}\n\n")
        f.write("-" * 60 + "\n")
        f.write("Classification Report:\n\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")


def load_metrics():
    """Load saved evaluation metrics for display on the prediction page."""
    if not METRICS_PATH.is_file():
        return None

    text = METRICS_PATH.read_text(encoding="utf-8", errors="replace")
    if "Testing Accuracy" not in text and "Accuracy" not in text:
        return None

    metrics = {}
    # Training accuracy removed - no longer saved in external report

    testing_accuracy = read_metric_value(text, r"Testing Accuracy", percent=True)
    if testing_accuracy is None:
        testing_accuracy = read_metric_value(text, r"Accuracy", percent=True)
    metrics["accuracy"] = testing_accuracy
    
    for name in ("precision", "recall", "f1", "roc_auc"):
        label_pattern = "roc[-_]auc" if name == "roc_auc" else re.escape(name)
        metrics[name] = read_metric_value(
            text,
            rf"{label_pattern}(?:-score)?",
            percent=True,
        )
    metrics = {name: value for name, value in metrics.items() if value is not None}

    return metrics or None


def read_metric_value(text, label_pattern, percent=False):
    """Read a numeric metric from a labeled report line."""
    match = re.search(
        rf"^\s*{label_pattern}\s*:\s*([\d.]+)\s*%?",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return None
    value = float(match.group(1))
    return value / 100 if percent and value > 1 else value


def main():
    """Simple smoke test for the evaluation helpers."""


if __name__ == "__main__":
    main()
