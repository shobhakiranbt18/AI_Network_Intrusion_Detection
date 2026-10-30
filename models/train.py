"""Train the DNN intrusion detector using Extra Trees for feature selection.

Evaluation protocol (paper style):
  UNSW_NB15_training-set.csv is split 70% / 30%.
    - 70% -> further split into 85% fit / 15% validation
    - 30% -> held-out TEST set (never used for fitting, scaling or threshold tuning)
"""

import json
import os
import sys
from datetime import datetime, timezone

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf

from models.model import build_dnn, get_callbacks
from src.evaluate import (
    DEFAULT_DECISION_THRESHOLD,
    evaluate_model,
    find_best_threshold,
    save_metrics,
)
from src.feature_selection import FEATURE_NAMES_PATH, run_feature_selection
from src.preprocess import preprocess_single_csv_split
from src.visualization_reporting import generate_visualization_reports

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DATASET_CSV = os.path.join(DATASET_DIR, "UNSW_NB15_training-set.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "intrusion_model.keras")
FEATURE_PATH = FEATURE_NAMES_PATH
THRESHOLD_PATH = os.path.join(PROJECT_ROOT, "models", "decision_threshold.pkl")
TRAINING_METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "training_metadata.json")

EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TEST_SPLIT = 0.30        # 30% of the training-set CSV is the held-out test set
VAL_SPLIT = 0.15         # 15% of the remaining 70% is used for validation
RANDOM_SEED = 42
# Max number of (one-hot expanded) features kept by Extra Trees.
K_FEATURES = 43
FEATURE_IMPORTANCE_THRESHOLD = 0.021
FEATURE_SELECTOR_ESTIMATORS = 350
VARIANCE_THRESHOLD = 0.0   # only drop constant columns (0.01 removed useful features)

MIN_ACCURACY_TARGET = 0.90


def cleanup_stale_model_artifacts() -> None:
    """Remove duplicate model files that are not used by the app."""
    stale_files = [
        os.path.join(PROJECT_ROOT, "models", "intrusion_model.h5"),
        os.path.join(PROJECT_ROOT, "models", "intrusion_model.joblib"),
    ]
    for stale_file in stale_files:
        try:
            if os.path.exists(stale_file):
                os.remove(stale_file)
        except OSError:
            pass


def main():
    """Split the training CSV 70/30, fit on 70%, evaluate on the 30% hold-out."""
    if not os.path.isfile(TRAIN_DATASET_CSV):
        raise FileNotFoundError(
            f"Required UNSW-NB15 CSV not found: {TRAIN_DATASET_CSV}. "
            "Place UNSW_NB15_training-set.csv in the dataset folder."
        )

    cleanup_stale_model_artifacts()
    os.makedirs(os.path.dirname(TRAINING_METADATA_PATH), exist_ok=True)
    with open(TRAINING_METADATA_PATH, "w", encoding="utf-8") as metadata_file:
        json.dump({"status": "training"}, metadata_file)

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Split raw rows 70/30 first; preprocessing is fitted on the 70% only.
    X_trainval, X_test, y_trainval, y_test, feature_names = preprocess_single_csv_split(
        TRAIN_DATASET_CSV,
        test_size=TEST_SPLIT,
        seed=RANDOM_SEED,
    )

    # Validation split (early stopping + threshold selection)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=VAL_SPLIT,
        stratify=y_trainval,
        random_state=RANDOM_SEED,
    )

    X_train, X_val, selected_features = run_feature_selection(
        X_train,
        X_val,
        y_train,
        feature_names,
        k=K_FEATURES,
        variance_threshold=VARIANCE_THRESHOLD,
        use_extra_trees=True,
        importance_threshold=FEATURE_IMPORTANCE_THRESHOLD,
    )
    selected_features = [name for name in selected_features if str(name).lower() != "id"]
    selected_features = [
        name for name in selected_features if str(name).lower() not in {"attack_cat", "label"}
    ]
    selected_indices = [feature_names.index(name) for name in selected_features]
    feature_names = selected_features
    X_test = X_test[:, selected_indices]

    model = build_dnn(input_dim=len(feature_names), learning_rate=LEARNING_RATE)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(patience=10),
        verbose=0,
    )

    training_probabilities = np.asarray(
        model.predict(X_train, batch_size=BATCH_SIZE, verbose=0)
    ).reshape(-1)
    training_predictions = (
        training_probabilities >= DEFAULT_DECISION_THRESHOLD
    ).astype(int)
    training_accuracy = accuracy_score(y_train, training_predictions)

    validation_probabilities = np.asarray(
        model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    ).reshape(-1)
    validation_accuracy = accuracy_score(
        y_val,
        (validation_probabilities >= DEFAULT_DECISION_THRESHOLD).astype(int),
    )

    # Threshold is chosen on the validation set only; the test set stays untouched.
    best_threshold = find_best_threshold(
        y_val,
        validation_probabilities,
        candidate_thresholds=np.linspace(0.20, 0.95, 151),
        metric="accuracy",
    )

    y_prob = np.asarray(
        model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    ).reshape(-1)
    metrics = evaluate_model(model, X_test, y_test, threshold=best_threshold)
    y_pred = (y_prob >= best_threshold).astype(int)

    model.save(MODEL_PATH)
    joblib.dump(feature_names, FEATURE_PATH)
    joblib.dump(best_threshold, THRESHOLD_PATH)
    print(f"Saved DNN model: {MODEL_PATH}")

    selector = ExtraTreesClassifier(
        n_estimators=FEATURE_SELECTOR_ESTIMATORS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    selector.fit(X_train, y_train)
    selected_feature_importances = selector.feature_importances_

    generate_visualization_reports(
        model,
        X_test,
        y_test,
        feature_names,
        threshold=best_threshold,
        history=history,
        feature_importances=selected_feature_importances,
    )
    save_metrics(
        metrics,
        y_test,
        y_pred,
        threshold=best_threshold,
        training_accuracy=training_accuracy,
    )

    report_path = os.path.join(PROJECT_ROOT, "results", "classification_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal (0)", "Attack (1)"],
        digits=4,
    )
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("UNSW-NB15 Test Set Classification Report (30% hold-out)\n")
        report_file.write(
            f"Model: Deep Neural Network; Extra Trees selected {len(feature_names)} features\n"
        )
        report_file.write(f"Training accuracy: {training_accuracy * 100:.2f}%\n")
        report_file.write(f"Validation accuracy: {validation_accuracy * 100:.2f}%\n")
        report_file.write(f"Testing accuracy: {metrics['accuracy'] * 100:.2f}%\n")
        report_file.write(f"Decision threshold: {best_threshold:.3f} (Optimized)\n")
        report_file.write(f"Epochs completed: {len(history.history['loss'])}/{EPOCHS}\n\n")
        report_file.write(report)
        report_file.write(
            "\nConfusion matrix (rows=true, columns=predicted; 0=Normal, 1=Attack):\n"
        )
        report_file.write(f"{confusion_matrix(y_test, y_pred, labels=[0, 1])}\n")

    with open(TRAINING_METADATA_PATH, "w", encoding="utf-8") as metadata_file:
        json.dump(
            {
                "status": "complete",
                "model_type": "DNN",
                "feature_selector": "ExtraTreesClassifier",
                "feature_selector_estimators": FEATURE_SELECTOR_ESTIMATORS,
                "selected_features": feature_names,
                "training_dataset": os.path.basename(TRAIN_DATASET_CSV),
                "evaluation_dataset": "30% hold-out split of UNSW_NB15_training-set.csv",
                "training_rows": int(len(y_trainval)),
                "fit_rows": int(len(y_train)),
                "validation_rows": int(len(y_val)),
                "testing_rows": int(len(y_test)),
                "epochs_max": EPOCHS,
                "epochs_completed": len(history.history["loss"]),
                "batch_size": BATCH_SIZE,
                "validation_accuracy": float(validation_accuracy),
                "training_accuracy": float(training_accuracy),
                "testing_accuracy": float(metrics["accuracy"]),
                "decision_threshold": float(best_threshold),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            metadata_file,
            indent=2,
        )


if __name__ == "__main__":
    main()