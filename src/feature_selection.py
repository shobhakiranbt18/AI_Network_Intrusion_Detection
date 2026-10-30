"""
src/feature_selection.py
------------------------
Feature selection for the UNSW-NB15 dataset using:
  - Variance Threshold  (removes near-zero variance columns)
  - ExtraTreesClassifier (as mentioned in base paper)
  - SelectKBest / f_classif (ANOVA F-test for numeric features)
  - Persists selected feature names to disk for use during prediction
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier

FEATURE_NAMES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "feature_names.pkl"
)
NON_FEATURE_COLUMNS = {"id", "attack_cat", "label"}


def drop_non_feature_columns(feature_names: list[str]) -> list[str]:
    """Remove identifier and label columns so they are never selected as model inputs."""
    return [name for name in feature_names if str(name).lower() not in NON_FEATURE_COLUMNS]


def valid_feature_indices(feature_names: list[str]) -> list[int]:
    """Return positional indices for columns that are valid model inputs."""
    return [idx for idx, name in enumerate(feature_names) if str(name).lower() not in NON_FEATURE_COLUMNS]


def remove_low_variance(X: np.ndarray,
                         feature_names: list[str],
                         threshold: float = 0.01
                         ) -> tuple[np.ndarray, list[str]]:
    """
    Remove features whose variance falls below *threshold*.

    Parameters
    ----------
    X            : np.ndarray  – feature matrix (already scaled)
    feature_names: list[str]   – ordered column names matching X
    threshold    : float       – variance threshold (default 0.01)

    Returns
    -------
    (X_reduced, kept_feature_names)
    """
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)
    mask = selector.get_support()
    kept = [name for name, keep in zip(feature_names, mask) if keep]
    print(f"[feature_selection] Variance threshold: {len(feature_names)} -> {len(kept)} features")
    return X_reduced, kept


def select_k_best_extra_trees(X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_test:  np.ndarray,
                               feature_names: list[str],
                               k: int = 30,
                               importance_threshold: float | None = 0.021
                               ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Select features using ExtraTreesClassifier importance scores.
    If a threshold is provided, keep every feature whose importance is greater than or equal to it,
    otherwise keep the top-k features as a fallback.
    """
    valid_indices = valid_feature_indices(feature_names)
    if valid_indices:
        X_train = X_train[:, valid_indices]
        X_test = X_test[:, valid_indices]
    feature_names = [feature_names[idx] for idx in valid_indices]

    k = min(k, X_train.shape[1])

    model = ExtraTreesClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    ranked_indices = np.argsort(importances)[::-1]

    selected_indices = ranked_indices[:k]

    if importance_threshold is not None:
        threshold_mask = importances[ranked_indices] >= importance_threshold
        threshold_indices = ranked_indices[threshold_mask]
        if threshold_indices.size >= k:
            selected_indices = threshold_indices[:k]
        else:
            # Keep the top-k features even if fewer pass the threshold. This avoids
            # dropping the model down to too few inputs while still removing ID and
            # non-feature columns before ranking.
            selected_indices = ranked_indices[:k]

    selected_indices = selected_indices[:k]
    selected_names = [feature_names[i] for i in selected_indices]

    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]
    return X_train_sel, X_test_sel, selected_names


def select_k_best(X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_test:  np.ndarray,
                   feature_names: list[str],
                   k: int = 30
                   ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Select the top *k* features using ANOVA F-test (f_classif).
    Fit on training data, transform both train and test sets.

    Parameters
    ----------
    X_train, y_train : training features and labels
    X_test           : test features
    feature_names    : ordered list of feature names
    k                : number of top features to select (default 30)

    Returns
    -------
    (X_train_sel, X_test_sel, selected_feature_names)
    """
    k = min(k, X_train.shape[1])   # cannot request more than available
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)

    mask     = selector.get_support()
    selected = [name for name, keep in zip(feature_names, mask) if keep]
    return X_train_sel, X_test_sel, selected


def run_feature_selection(X_train: np.ndarray,
                           X_test:  np.ndarray,
                           y_train: np.ndarray,
                           feature_names: list[str],
                           k: int = 30,
                           variance_threshold: float = 0.01,
                           use_extra_trees: bool = True,
                           importance_threshold: float | None = 0.021
                           ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Complete feature selection pipeline:
      1. Remove low-variance features
      2. ExtraTreesClassifier feature importance (if use_extra_trees=True)
         OR SelectKBest with ANOVA F-test (if use_extra_trees=False)
      3. Persist selected feature names for prediction-time use
    """
    # Guard against accidental selection of identifier columns.
    valid_indices = valid_feature_indices(feature_names)
    if valid_indices:
        X_train = X_train[:, valid_indices]
        X_test = X_test[:, valid_indices]
    feature_names = [feature_names[idx] for idx in valid_indices]

    # Step 1 – variance filter
    X_train_vt, vt_names = remove_low_variance(X_train, feature_names, variance_threshold)
    # Apply same columns to test set
    col_indices = [feature_names.index(n) for n in vt_names]
    X_test_vt   = X_test[:, col_indices]

    # Step 2 – Feature selection (ExtraTreesClassifier or SelectKBest)
    if use_extra_trees:
        X_train_sel, X_test_sel, selected_names = select_k_best_extra_trees(
            X_train_vt, y_train, X_test_vt, vt_names, k=k, importance_threshold=importance_threshold
        )
    else:
        X_train_sel, X_test_sel, selected_names = select_k_best(
            X_train_vt, y_train, X_test_vt, vt_names, k=k
        )

    # Step 3 – Persist
    os.makedirs(os.path.dirname(FEATURE_NAMES_PATH), exist_ok=True)
    joblib.dump(selected_names, FEATURE_NAMES_PATH)
    print(f"[feature_selection] Saved {len(selected_names)} feature names -> {FEATURE_NAMES_PATH}")

    return X_train_sel, X_test_sel, selected_names


def load_feature_names() -> list[str]:
    """
    Load the persisted feature names list produced during training.

    Returns
    -------
    list[str] of selected feature names, or empty list if not found.
    """
    if os.path.exists(FEATURE_NAMES_PATH):
        names = joblib.load(FEATURE_NAMES_PATH)
        print(f"[feature_selection] Loaded {len(names)} feature names from disk.")
        return names
    print("[feature_selection] WARNING: feature_names.pkl not found.")
    return []


def get_feature_importances(feature_names: list[str],
                             X_train: np.ndarray,
                             y_train: np.ndarray) -> pd.DataFrame:
    """
    Compute ANOVA F-scores for all features and return as a sorted DataFrame.
    Useful for reporting / visualisation.

    Parameters
    ----------
    feature_names : list[str]
    X_train       : np.ndarray
    y_train       : np.ndarray

    Returns
    -------
    pd.DataFrame with columns ['feature', 'f_score', 'p_value'] sorted by f_score desc
    """
    from sklearn.feature_selection import f_classif
    f_scores, p_values = f_classif(X_train, y_train)
    df = pd.DataFrame({
        "feature": feature_names,
        "f_score": f_scores,
        "p_value": p_values
    }).sort_values("f_score", ascending=False).reset_index(drop=True)
    return df
