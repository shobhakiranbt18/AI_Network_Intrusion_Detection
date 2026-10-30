"""
models/predict.py
-----------------
Prediction logic for the trained DNN intrusion detection model.

Responsibilities
----------------
- Load the saved model (intrusion_model.keras) on first call (cached)
- Preprocess an uploaded CSV using the same pipeline as training
- Return per-row predictions with confidence scores
- Provide a summary dict for the result page
"""

import os
import sys

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# Allow imports from project root regardless of working directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess        import preprocess_uploaded_file
from src.feature_selection import load_feature_names
from src.evaluate import DEFAULT_DECISION_THRESHOLD, predict_attack_probabilities

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "intrusion_model.keras")
MODEL_LEGACY_PATH = os.path.join(PROJECT_ROOT, "models", "intrusion_model.h5")
ALL_FEATURES_PATH = os.path.join(PROJECT_ROOT, "models", "all_feature_names.pkl")

# Module-level cache so the model is loaded only once per process
_model_cache: tf.keras.Model | None = None


def load_model() -> tf.keras.Model:
    """
    Load the trained DNN model from disk.
    Uses module-level caching to avoid reloading on every request.

    Returns
    -------
    tf.keras.Model

    Raises
    ------
    FileNotFoundError if intrusion_model.keras does not exist.
    """
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not os.path.isfile(MODEL_PATH):
        if os.path.isfile(MODEL_LEGACY_PATH):
            model_path = MODEL_LEGACY_PATH
        else:
            raise FileNotFoundError(
                f"Trained model not found at: {MODEL_PATH}\n"
                "Please run  python models/train.py  to train the model first."
            )
    else:
        model_path = MODEL_PATH

    try:
        _model_cache = tf.keras.models.load_model(model_path)
    except Exception as exc:  # pragma: no cover - surfaced to callers as runtime issue
        raise RuntimeError(f"Failed to load trained model from {model_path}: {exc}") from exc
    return _model_cache


def load_decision_threshold() -> float:
    """Return the fixed Attack probability threshold used by the application."""
    return DEFAULT_DECISION_THRESHOLD


def predict_from_file(filepath: str,
                      threshold: float | None = None,
                      batch_size: int = 1000) -> dict:
    """
    Run the full prediction pipeline on an uploaded CSV file.

    Steps
    -----
    1. Load model (cached)
    2. Load selected feature names from training
    3. Preprocess the uploaded CSV (with chunked processing)
    4. Run inference in batches
    5. Build and return a results dictionary

    Parameters
    ----------
    filepath   : str   – absolute path to the uploaded CSV
    threshold  : float – classification threshold (default 0.50)
    batch_size : int   – batch size for predictions (default 1000)

    Returns
    -------
    dict with keys:
        predictions   : list[dict]  – per-row results
        total_rows    : int
        attack_count  : int
        normal_count  : int
        overall_label : str         – Attack if at least one row is flagged
        overall_conf  : float       – average confidence among flagged rows
    """
    if not filepath or not os.path.isfile(filepath):
        raise FileNotFoundError(f"Input CSV not found: {filepath}")

    if batch_size is None or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer.")
    batch_size = int(batch_size)

    if threshold is None:
        threshold = load_decision_threshold()
    else:
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be a finite float between 0 and 1.")

    # 1. Load model
    model = load_model()

    # 2. Load feature names persisted during training
    feature_names = load_feature_names()
    if not feature_names:
        raise RuntimeError(
            "Feature names not found. Please re-train the model first."
        )
    expected_features = model.input_shape[-1]
    if expected_features != len(feature_names):
        raise RuntimeError(
            f"Model expects {expected_features} input features, but "
            f"feature_names.pkl contains {len(feature_names)}. "
            "The training run may have been interrupted; complete training "
            "before starting predictions."
        )

    # 3. Preprocess uploaded CSV (with chunked processing)
    X = preprocess_uploaded_file(filepath, feature_names)

    if X.shape[0] == 0:
        raise ValueError("The uploaded CSV contains no valid data rows after preprocessing.")

    # 4. Run inference in batches to avoid memory issues
    y_prob_list = []

    for i in range(0, X.shape[0], batch_size):
        batch_end = min(i + batch_size, X.shape[0])
        X_batch = X[i:batch_end]
        y_prob_batch = predict_attack_probabilities(model, X_batch).astype(np.float64)
        if y_prob_batch.size == 0:
            continue
        y_prob_list.append(y_prob_batch)

    if not y_prob_list:
        raise ValueError("The model produced no predictions for the uploaded data.")

    y_prob = np.concatenate(y_prob_list).astype(np.float64)
    if not np.isfinite(y_prob).all() or np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("The model returned invalid probabilities; expected values between 0 and 1.")

    # Binary classification at the validation-selected cutoff:
    # Model outputs probability that sample is class 1 (Attack)
    # - If prob >= threshold → Predict Attack (1)
    # - Otherwise            → Predict Normal (0)
    y_pred = (y_prob >= threshold).astype(int)

    # 5. Build per-row result list
    predictions = []
    for i, (prob, pred) in enumerate(zip(y_prob, y_pred)):
        # prob = probability of Attack class
        attack_probability = float(prob)
        
        # Determine label based on prediction
        if pred == 1:  # Attack
            label = "Attack (1)"
            confidence = attack_probability  # Confidence is the probability of predicted class
        else:  # Normal
            label = "Normal (0)"
            confidence = 1.0 - attack_probability  # Confidence is probability of Normal
            
        predictions.append({
            "row":        i + 1,
            "label":      label,
            "confidence": round(confidence * 100, 2),
            "raw_prob":   round(attack_probability, 4),
        })

    # 6. Summary statistics
    attack_count  = int(np.sum(y_pred == 1))
    normal_count  = int(np.sum(y_pred == 0))
    total_rows    = len(y_pred)
    normal_mask = y_pred == 0
    attack_mask = y_pred == 1
    avg_confidence_normal = round(
        float(np.mean(1.0 - y_prob[normal_mask]) * 100), 2
    ) if normal_count else 0.0
    avg_confidence_attack = round(
        float(np.mean(y_prob[attack_mask]) * 100), 2
    ) if attack_count else 0.0

    # A security summary should flag any predicted attack, even if it is a minority.
    if attack_count:
        overall_label = "Attack (1)"
        overall_conf = avg_confidence_attack
    else:
        overall_label = "Normal (0)"
        overall_conf = avg_confidence_normal

    result = {
        "predictions":          predictions,
        "total_rows":           total_rows,
        "attack_count":         attack_count,
        "normal_count":         normal_count,
        "overall_label":        overall_label,
        "overall_conf":         overall_conf,
        "threshold_used":       round(threshold, 4),
    }

    return result


def predict_single_row(row_data: dict,
                        feature_names: list[str],
                        threshold: float | None = None) -> dict:
    """
    Predict a single network traffic record supplied as a dict.

    Parameters
    ----------
    row_data      : dict        – feature name → value mapping
    feature_names : list[str]   – ordered features expected by the model
    threshold     : float

    Returns
    -------
    dict with label, confidence, raw_prob
    """
    from src.preprocess import SCALER_PATH

    if not isinstance(row_data, dict):
        raise TypeError("row_data must be a dict mapping feature names to values.")
    if not feature_names:
        raise ValueError("feature_names cannot be empty.")

    model = load_model()
    expected_features = model.input_shape[-1]
    if expected_features != len(feature_names):
        raise RuntimeError(
            f"Model expects {expected_features} input features, but "
            f"{len(feature_names)} feature names were provided. "
            "Re-train the model before making predictions."
        )
    if threshold is None:
        threshold = load_decision_threshold()
    else:
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be a finite float between 0 and 1.")

    if not os.path.exists(ALL_FEATURES_PATH):
        raise RuntimeError(
            "all_feature_names.pkl not found. Please re-train the model first."
        )
    all_feature_names = joblib.load(ALL_FEATURES_PATH)
    missing_features = [name for name in feature_names if name not in all_feature_names]
    if missing_features:
        raise ValueError(
            f"Selected features are missing from all_feature_names.pkl: {missing_features}"
        )

    # Scale the complete input vector before passing it to the classifier.
    df = pd.DataFrame([row_data])
    for feat in all_feature_names:
        if feat not in df.columns:
            df[feat] = 0
    df = df[all_feature_names].copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Scale
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        X_all = scaler.transform(df.values)
    else:
        X_all = df.values

    selected_indices = [all_feature_names.index(name) for name in feature_names]
    X = X_all[:, selected_indices]

    prob = float(predict_attack_probabilities(model, X)[0])
    pred = int(prob >= threshold)
    label = "Attack (1)" if pred == 1 else "Normal (0)"
    confidence = prob if pred == 1 else (1.0 - prob)

    return {
        "label":      label,
        "confidence": round(confidence * 100, 2),
        "raw_prob":   round(prob, 4),
    }
