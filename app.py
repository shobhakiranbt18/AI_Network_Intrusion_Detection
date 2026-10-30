"""
app.py
------
Flask web application for a network-based intrusion detection system using deep
learning. Runs exclusively on localhost (127.0.0.1:5000).

Routes
------
GET  /            → Home page (index.html)
GET  /upload      → Upload page (upload.html)
POST /predict     → Process uploaded CSV, run inference → result.html
GET  /result      → Redirect back to upload if accessed directly
"""

import json
import os
import subprocess
import sys
import traceback
import uuid

# ── Project root on sys.path so src/ and models/ are importable ───────────────
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _bootstrap_project_venv() -> None:
    """Re-run this script under the bundled project venv when Flask is missing."""
    if os.environ.get("ARSS_APP_BOOTSTRAPPED") == "1":
        return

    try:
        import flask  # noqa: F401
        return
    except ModuleNotFoundError:
        venv_candidates = [
            os.path.join(PROJECT_ROOT, ".venv311", "Scripts", "python.exe"),
            os.path.join(PROJECT_ROOT, ".venv_run", "Scripts", "python.exe"),
            os.path.join(PROJECT_ROOT, "venv_py311", "Scripts", "python.exe"),
            os.path.join(PROJECT_ROOT, ".venv311", "bin", "python"),
            os.path.join(PROJECT_ROOT, ".venv_run", "bin", "python"),
            os.path.join(PROJECT_ROOT, "venv_py311", "bin", "python"),
        ]
        for candidate in venv_candidates:
            if os.path.isfile(candidate):
                os.environ["ARSS_APP_BOOTSTRAPPED"] = "1"
                result = subprocess.run(
                    [candidate, os.path.join(PROJECT_ROOT, "app.py"), *sys.argv[1:]],
                    check=False,
                )
                raise SystemExit(result.returncode)
        print(
            "Flask is not installed in the active Python environment. "
            "Please run the project with its bundled venv: "
            "./.venv311/Scripts/python.exe app.py",
            file=sys.stderr,
        )
        raise SystemExit(1)


_bootstrap_project_venv()

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from models.predict import predict_from_file
from src.evaluate   import DEFAULT_DECISION_THRESHOLD, load_metrics

# ── App configuration ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(32)          # required for flash messages

APP_USERNAME = "admin"
APP_PASSWORD = "admin123"

UPLOAD_FOLDER   = os.path.join(PROJECT_ROOT, "uploads")
ALLOWED_EXTENSIONS = {"csv"}
MAX_CONTENT_LENGTH  = 150 * 1024 * 1024   # 150 MB upload limit (for UNSW-NB15 datasets)

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Helper functions ──────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    """Return True only if the filename has a .csv extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def model_is_trained() -> bool:
    """Check whether the trained model and required artefacts exist.

    Accept both the older metadata format (official testing-file name) and the
    current training metadata format that stores the 30% hold-out evaluation.
    """
    model_path   = os.path.join(PROJECT_ROOT, "models", "intrusion_model.keras")
    scaler_path  = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
    encoders_path = os.path.join(PROJECT_ROOT, "models", "encoders.pkl")
    preprocessor_path = os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl")
    feature_path = os.path.join(PROJECT_ROOT, "models", "feature_names.pkl")
    all_features_path = os.path.join(PROJECT_ROOT, "models", "all_feature_names.pkl")
    threshold_path = os.path.join(PROJECT_ROOT, "models", "decision_threshold.pkl")
    metadata_path = os.path.join(PROJECT_ROOT, "models", "training_metadata.json")
    required_paths = (
        model_path,
        scaler_path,
        encoders_path,
        preprocessor_path,
        feature_path,
        all_features_path,
        threshold_path,
        metadata_path,
    )
    if not all(os.path.isfile(path) for path in required_paths):
        return False
    try:
        with open(metadata_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError) as exc:
        app.logger.warning("Training metadata is unavailable or invalid: %s", exc)
        return False

    status = metadata.get("status")
    if status != "complete":
        return False

    model_type = metadata.get("model_type")
    feature_selector = metadata.get("feature_selector")
    evaluation_dataset = metadata.get("evaluation_dataset", "")
    evaluation_dataset_lower = str(evaluation_dataset).lower()
    is_compatible_dataset = (
        evaluation_dataset_lower == "unsw_nb15_testing-set.csv"
        or "testing-set" in evaluation_dataset_lower
        or "hold-out" in evaluation_dataset_lower
        or "unsw_nb15_training-set.csv" in evaluation_dataset_lower
    )
    return (
        model_type == "DNN"
        and feature_selector == "ExtraTreesClassifier"
        and is_compatible_dataset
    )


def get_plot_status() -> dict[str, bool]:
    """Return availability flags for dashboard and evaluation plot images."""
    image_dir = os.path.join(PROJECT_ROOT, "static", "images")
    filenames = {
        "classification_metrics": "classification_metrics.png",
        "confusion_matrix": "confusion_matrix.png",
        "roc_curve": "roc_curve.png",
        "feature_importance_selected": "feature_importance_selected.png",
        "training_accuracy": "training_accuracy.png",
        "training_loss": "training_loss.png",
    }
    return {name: os.path.exists(os.path.join(image_dir, filename)) for name, filename in filenames.items()}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Login-first flow: logged in users go straight to upload."""
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return redirect(url_for("upload"))


@app.route("/login", methods=["GET", "POST"])
def login():
    """Simple demo login screen for the application."""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username == APP_USERNAME and password == APP_PASSWORD:
            session["logged_in"] = True
            session["username"] = username
            flash("Login successful.", "success")
            return redirect(url_for("upload"))

        flash("Invalid username or password.", "danger")

    return render_template("login.html", model_trained=model_is_trained())


@app.route("/upload")
def upload():
    """Upload / Prediction page behind a simple login gate."""
    if not session.get("logged_in"):
        flash("Please log in to continue.", "warning")
        return redirect(url_for("login"))

    if not model_is_trained():
        flash(
            "The model has not been trained yet. "
            "Please run  python models/train.py  to train the model first.",
            "warning",
        )
        return render_template("index.html", model_trained=False)
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle CSV upload, run inference, and render result page.

    Validations
    -----------
    - File must be present in the request
    - File must have a .csv extension
    - File must not be empty

    On success  → render result.html with full prediction data
    On failure  → flash error message and redirect to upload page
    """
    # ── Validate file presence ────────────────────────────────────────────
    if "file" not in request.files:
        flash("No file part in the request. Please select a CSV file.", "danger")
        return redirect(url_for("upload"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected. Please choose a CSV file to upload.", "danger")
        return redirect(url_for("upload"))

    if not allowed_file(file.filename):
        flash(
            "Invalid file type. Only CSV files (.csv) are accepted.",
            "danger",
        )
        return redirect(url_for("upload"))

    # ── Save uploaded file with a unique name ─────────────────────────────
    original_name = secure_filename(file.filename)
    unique_name   = f"{uuid.uuid4().hex}_{original_name}"
    save_path     = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)

    try:
        file.save(save_path)

        # Quick sanity check: file must not be empty
        if os.path.getsize(save_path) == 0:
            os.remove(save_path)
            flash("Uploaded file is empty. Please upload a valid CSV file.", "danger")
            return redirect(url_for("upload"))

        # ── Run prediction pipeline ───────────────────────────────────────
        result = predict_from_file(save_path)

        # ── Limit predictions sent to browser (prevent crash) ─────────────
        # Only send first 1000 rows to browser, but keep full statistics
        all_predictions = result["predictions"]
        limited_predictions = all_predictions[:1000]  # Only first 1000 rows
        displayed_attack_count = sum(
            prediction["label"] == "Attack" for prediction in limited_predictions
        )
        displayed_normal_count = len(limited_predictions) - displayed_attack_count
        
        # Add warning if data was truncated
        data_truncated = len(all_predictions) > 1000

        # ── Load training evaluation metrics ─────────────────────────────
        metrics = load_metrics()

        # ── Plot availability ─────────────────────────────────────────────
        plots = get_plot_status()
        # Show plots even if metrics file doesn't exist (graphs may still be available)

        # ── Clean up temporary upload ─────────────────────────────────────
        try:
            os.remove(save_path)
        except OSError:
            pass

        return render_template(
            "result.html",
            filename      = original_name,
            predictions   = limited_predictions,  # Limited to 1000 rows
            displayed_attack_count= displayed_attack_count,
            displayed_normal_count= displayed_normal_count,
            total_rows    = result["total_rows"],
            attack_count  = result["attack_count"],
            normal_count  = result["normal_count"],
            overall_label = result["overall_label"],
            overall_conf  = result["overall_conf"],
            threshold_used= result.get("threshold_used", DEFAULT_DECISION_THRESHOLD),
            metrics       = metrics,
            metrics_stale = metrics is None,
            plots         = plots,
            data_truncated= data_truncated,  # Flag for display message
            total_predictions= len(all_predictions)  # Actual total
        )

    except FileNotFoundError as e:
        flash(str(e), "danger")
        return redirect(url_for("upload"))

    except ValueError as e:
        flash(f"Data error: {e}", "danger")
        return redirect(url_for("upload"))

    except RuntimeError as e:
        flash(f"Runtime error: {e}", "danger")
        return redirect(url_for("upload"))

    except Exception:
        app.logger.error(traceback.format_exc())
        flash(
            "An unexpected error occurred while processing your file. "
            "Please ensure it follows the UNSW-NB15 feature format.",
            "danger",
        )
        # Clean up if still exists
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass
        return redirect(url_for("upload"))


@app.route("/result")
def result():
    """Redirect direct GET requests to the upload page."""
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return redirect(url_for("upload"))


@app.route("/logout")
def logout():
    """Clear any session state and return to the login page."""
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def page_not_found(e):
    return render_template("index.html",
                           model_trained=model_is_trained(),
                           error="Page not found (404)."), 404


@app.errorhandler(413)
def request_entity_too_large(e):
    flash("Uploaded file exceeds the 150 MB limit. Please upload a smaller file.", "danger")
    return redirect(url_for("upload"))


@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(traceback.format_exc())
    flash("Internal server error. Please try again.", "danger")
    return redirect(url_for("index"))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60, flush=True)
    print("  Network-Based Intrusion Detection Using Deep Learning", flush=True)
    print("  Running on http://127.0.0.1:5000", flush=True)
    print("=" * 60 + "\n", flush=True)

    if not model_is_trained():
        print("WARNING: Trained model not found.", flush=True)
        print("   Run  python models/train.py  before making predictions.\n", flush=True)

    # Localhost only - keep debug off to avoid duplicate reloader processes during local testing
    app.run(host="127.0.0.1", port=5000, debug=False)
