"""
src/preprocess.py
-----------------
Data preprocessing for the UNSW-NB15 dataset.

  - Loading / validating CSV files
  - Dropping identifier columns (id, attack_cat)
  - Filling missing values using TRAIN statistics only
  - log1p for heavy-tailed numeric columns, then MinMax scaling
  - One-hot encoding of categorical columns
  - 70/30 split of a single CSV (preprocessor is fitted on the 70% only)
  - Preprocessing of uploaded CSVs for prediction (same pipeline as training)
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

# -- Columns that are identifiers or ground-truth labels -----------------------
DROP_COLS = ["id", "attack_cat"]   # drop id and multi-class attack category
LABEL_COL = "label"                # binary label: 0 = Normal, 1 = Attack

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
SCALER_PATH = os.path.join(_MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(_MODELS_DIR, "encoders.pkl")
PREPROCESSOR_PATH = os.path.join(_MODELS_DIR, "preprocessor.pkl")
ALL_FEATURES_PATH = os.path.join(_MODELS_DIR, "all_feature_names.pkl")


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def safe_log1p(x):
    """log1p that never produces NaN (negative values are clipped to 0).

    Defined at module level so the fitted preprocessor can be pickled.
    """
    return np.log1p(np.clip(x, 0, None))


def _read_labeled_csv(path: str, name: str) -> pd.DataFrame:
    """Read a CSV, normalise column names and validate the binary label."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"UNSW-NB15 {name} CSV not found: {path}")

    frame = pd.read_csv(path, low_memory=False)
    frame.columns = frame.columns.str.strip().str.lower()
    frame = _normalise_missing_values(frame)

    if LABEL_COL not in frame.columns:
        raise ValueError(f"The {name} CSV is missing the required '{LABEL_COL}' column.")
    try:
        labels = frame[LABEL_COL].astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"The {name} CSV label column must contain binary 0/1 values.") from exc
    if not labels.isin([0, 1]).all():
        raise ValueError(f"The {name} CSV label column must contain only 0/1 values.")
    frame[LABEL_COL] = labels
    return frame


def _normalise_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable missing markers to NumPy NaN without coercing numeric columns to object."""
    normalised = frame.copy()
    for column in normalised.columns:
        series = normalised[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            normalised[column] = pd.to_numeric(series, errors="coerce")
        else:
            normalised[column] = series.astype(object).where(pd.notna(series), np.nan)
    return normalised


def _is_categorical_series(series: pd.Series) -> bool:
    """Return True for columns stored as strings/categories rather than numeric types."""
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
    )


def _compute_fill_values(frame: pd.DataFrame) -> dict:
    """Median (numeric) / mode (categorical) computed on the TRAIN frame only."""
    fill_values = {}
    for column in frame.columns:
        series = frame[column]
        if _is_categorical_series(series):
            modes = series.mode()
            fill_values[column] = modes.iloc[0] if not modes.empty else "unknown"
        else:
            fill_values[column] = series.median()
    return fill_values


def _apply_fill_values(frame: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    frame = frame.drop(columns=DROP_COLS, errors="ignore").copy()
    for column in frame.columns:
        if column in fill_values:
            fill_value = fill_values[column]
            if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
                frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(fill_value)
            else:
                frame[column] = frame[column].astype(object).where(pd.notna(frame[column]), fill_value)
    return frame


def fit_preprocessor(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Fit the full preprocessor on TRAIN data only and transform both frames.

    Numeric columns:
        * heavy-tailed (skew > 2, non-negative) -> median impute -> log1p -> MinMax
        * others                                 -> median impute -> MinMax
    Categorical columns: most-frequent impute -> one-hot
    """
    train_features = _normalise_missing_values(
        train_df.drop(columns=[LABEL_COL], errors="ignore")
    )
    test_features = _normalise_missing_values(
        test_df.drop(columns=[LABEL_COL], errors="ignore")
    )

    cat_cols = [
        col
        for col in train_features.columns
        if not pd.api.types.is_numeric_dtype(train_features[col])
        and not pd.api.types.is_bool_dtype(train_features[col])
    ]
    num_cols = [col for col in train_features.columns if col not in cat_cols]

    skewed_cols = [
        c for c in num_cols
        if train_features[c].min() >= 0 and train_features[c].skew() > 2
    ]
    normal_num_cols = [c for c in num_cols if c not in skewed_cols]

    skewed_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(safe_log1p, feature_names_out="one-to-one")),
        ("scaler", MinMaxScaler()),
    ])
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=0.01,
                max_categories=50,
                sparse_output=False,
            ),
        ),
    ])

    transformers = []
    if skewed_cols:
        transformers.append(("skew", skewed_transformer, skewed_cols))
    if normal_num_cols:
        transformers.append(("num", numeric_transformer, normal_num_cols))
    if cat_cols:
        transformers.append(("cat", categorical_transformer, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    train_array = preprocessor.fit_transform(train_features)
    test_array = preprocessor.transform(test_features)
    feature_names = preprocessor.get_feature_names_out().tolist()

    os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    if "cat" in preprocessor.named_transformers_:
        joblib.dump(
            preprocessor.named_transformers_["cat"].named_steps["onehot"], ENCODER_PATH
        )
    # Legacy scaler kept only because predict.py::predict_single_row imports it.
    legacy_scaler = MinMaxScaler()
    legacy_scaler.fit(train_array)
    joblib.dump(legacy_scaler, SCALER_PATH)
    joblib.dump(feature_names, ALL_FEATURES_PATH)

    return train_array, test_array, feature_names, preprocessor


# ------------------------------------------------------------------------------
# Training-time entry points
# ------------------------------------------------------------------------------
def preprocess_single_csv_split(
    csv_path: str,
    test_size: float = 0.30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Paper-style evaluation: split ONE csv (training-set) into 70% train / 30% test.

    The split happens on the RAW rows first, and imputation values, log1p scaling,
    MinMax scaling and one-hot encoding are all fitted on the 70% part only,
    so the 30% test part never influences preprocessing (no leakage).
    """
    raw = _read_labeled_csv(csv_path, "training")
    raw = raw.drop(columns=DROP_COLS, errors="ignore")

    train_raw, test_raw = train_test_split(
        raw,
        test_size=test_size,
        stratify=raw[LABEL_COL],
        random_state=seed,
    )
    train_raw = train_raw.reset_index(drop=True)
    test_raw = test_raw.reset_index(drop=True)

    fill_values = _compute_fill_values(train_raw)
    train_df = _apply_fill_values(train_raw, fill_values)
    test_df = _apply_fill_values(test_raw, fill_values)

    X_train, X_test, feature_names, _ = fit_preprocessor(train_df, test_df)
    y_train = train_df[LABEL_COL].astype(int).to_numpy()
    y_test = test_df[LABEL_COL].astype(int).to_numpy()

    print(
        f"[preprocess] Single-CSV split: {len(y_train):,} train rows, "
        f"{len(y_test):,} test rows ({test_size:.0%}); "
        f"{len(feature_names)} transformed features."
    )
    return X_train, X_test, y_train, y_test, feature_names


def preprocess_official_train_test(
    train_path: str,
    test_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Preprocess the official train and test CSVs without fitting on test data."""
    train_raw = _read_labeled_csv(train_path, "training")
    test_raw = _read_labeled_csv(test_path, "testing")

    train_features = train_raw.drop(columns=DROP_COLS, errors="ignore")
    test_features = test_raw.drop(columns=DROP_COLS, errors="ignore")
    missing_features = sorted(
        set(train_features.columns) - set(test_features.columns)
    )
    if missing_features:
        raise ValueError(
            f"The testing CSV is missing training feature columns: {missing_features}"
        )

    fill_values = _compute_fill_values(train_features)
    train_df = _apply_fill_values(train_raw, fill_values)
    test_df = _apply_fill_values(test_raw, fill_values)

    X_train, X_test, feature_names, _ = fit_preprocessor(train_df, test_df)
    y_train = train_df[LABEL_COL].astype(int).to_numpy()
    y_test = test_df[LABEL_COL].astype(int).to_numpy()

    print(
        f"[preprocess] Official UNSW-NB15 split: {len(y_train):,} training rows, "
        f"{len(y_test):,} testing rows; {len(feature_names)} transformed features."
    )
    return X_train, X_test, y_train, y_test, feature_names


# ------------------------------------------------------------------------------
# Prediction-time entry point
# ------------------------------------------------------------------------------
def preprocess_uploaded_file(filepath: str,
                             feature_names: list[str],
                             chunksize: int = 10000) -> np.ndarray:
    """
    Preprocess a user-uploaded CSV for prediction (chunked).

    Uses the fitted ColumnTransformer saved during training, so cleaning, log1p,
    scaling and one-hot encoding are identical to training. Missing values are
    left as NaN and handled by the imputers fitted on the training data.
    Finally the columns are selected IN THE SAME ORDER as `feature_names`
    (the order the DNN was trained with).
    """
    if not os.path.exists(PREPROCESSOR_PATH):
        raise RuntimeError(
            "No trained preprocessing pipeline was found. "
            "Please re-train the model using: python models/train.py"
        )

    print(f"[preprocess] Processing uploaded file in chunks of {chunksize} rows...", flush=True)

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    raw_feature_names = list(preprocessor.feature_names_in_)
    all_feature_cols = preprocessor.get_feature_names_out().tolist()
    print(f"[preprocess] Loaded preprocessor with {len(all_feature_cols)} transformed features.", flush=True)

    missing = [name for name in feature_names if name not in all_feature_cols]
    if missing:
        raise RuntimeError(
            f"Selected features not produced by the saved preprocessor: {missing}. "
            "Please re-train the model."
        )
    # Order MUST match the training order of the selected features.
    selected_indices = [all_feature_cols.index(name) for name in feature_names]
    print(f"[preprocess] Model expects {len(feature_names)} selected features", flush=True)

    chunks_processed = []
    total_rows = 0

    try:
        for chunk_num, df_chunk in enumerate(
            pd.read_csv(filepath, chunksize=chunksize, low_memory=False), 1
        ):
            df_chunk.columns = df_chunk.columns.str.strip().str.lower()
            df_chunk = _normalise_missing_values(df_chunk)
            cols_to_drop = [c for c in DROP_COLS + [LABEL_COL] if c in df_chunk.columns]
            df_chunk = df_chunk.drop(columns=cols_to_drop, errors="ignore")

            if df_chunk.empty:
                continue

            for col in raw_feature_names:
                if col not in df_chunk.columns:
                    df_chunk[col] = np.nan      # imputer fills it with train stats
            df_chunk = df_chunk[raw_feature_names]

            X_chunk = preprocessor.transform(df_chunk)
            X_chunk_final = X_chunk[:, selected_indices]

            chunks_processed.append(X_chunk_final)
            total_rows += X_chunk_final.shape[0]
            print(
                f"[preprocess] Processed chunk {chunk_num}: "
                f"{X_chunk_final.shape[0]} rows (Total: {total_rows})",
                flush=True,
            )
            del df_chunk

        if not chunks_processed:
            raise ValueError("No valid data found in the uploaded file after preprocessing.")

        X = np.vstack(chunks_processed)
        print(f"[preprocess] Successfully processed {X.shape[0]} rows with {X.shape[1]} features", flush=True)
        return X

    except pd.errors.ParserError as e:
        raise ValueError(f"CSV parsing error: {str(e)}. Please ensure the file is a valid CSV.")
    except MemoryError:
        raise RuntimeError(
            "Out of memory error. The file is too large to process. "
            "Please try uploading a smaller file (< 50,000 rows recommended)."
        )