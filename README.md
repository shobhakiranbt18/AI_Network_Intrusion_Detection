# Network-Based Intrusion Detection Using Deep Learning

A network-based intrusion detection system that classifies network traffic as **Normal** or **Attack** using a Deep Neural Network (DNN) trained on the **UNSW-NB15** dataset. Extra Trees selects the DNN's 8 input features.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Technologies Used](#technologies-used)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Dataset Setup](#dataset-setup)
7. [Training the Model](#training-the-model)
8. [Running the Web Application](#running-the-web-application)
9. [Using the Application](#using-the-application)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Generated Artefacts](#generated-artefacts)
12. [Model Architecture](#model-architecture)
13. [Troubleshooting](#troubleshooting)

---

## Project Overview

This system performs binary classification of network traffic records:

| Label  | Meaning                              |
|--------|--------------------------------------|
| **0**  | Normal – legitimate network traffic  |
| **1**  | Attack – malicious / intrusion event |

The pipeline trains on the official UNSW-NB15 training CSV and evaluates once on the separate official testing CSV. A stratified validation split is taken only from the training file. Encoders, scaling, and feature selection are fitted on training data only.

---

## Project Structure

```
AI_Network_Intrusion_Detection/
│
├── app.py                        # Flask routes and web logic
├── requirements.txt              # Python dependencies (pinned versions)
├── README.md                     # This file
│
├── dataset/
│   ├── UNSW_NB15_training-set.csv # Official training file
│   └── UNSW_NB15_testing-set.csv  # Separate official final evaluation file
│
├── models/
│   ├── model.py                  # DNN architecture and training callbacks
│   ├── train.py                  # Full training pipeline (run this first)
│   ├── predict.py                # Prediction logic for the web app
│   ├── intrusion_model.keras     # Trained DNN model (generated)
│   ├── best.weights.h5           # Best validation checkpoint (generated)
│   ├── scaler.pkl                # Fitted MinMaxScaler (generated)
│   ├── encoders.pkl              # Categorical encoders (generated)
│   ├── feature_names.pkl         # Ordered 8 DNN input feature names (generated)
│   ├── all_feature_names.pkl     # Ordered preprocessing feature names (generated)
│   └── training_metadata.json    # Confirms official-test evaluation completed
│
├── src/
│   ├── preprocess.py             # Data loading, cleaning, encoding, scaling
│   ├── feature_selection.py      # Optional variance/Extra Trees/ANOVA selection
│   ├── evaluate.py               # Metric computation and persistence
│   └── visualization_reporting.py # Training and evaluation graph generation
│
├── templates/
│   ├── index.html                # Home page
│   ├── upload.html               # CSV upload page
│   └── result.html               # Prediction results page
│
├── static/
│   ├── css/
│   │   └── style.css             # Dark-theme custom styles
│   └── images/
│       ├── feature_importance_selected.png # Top 8 feature rankings among all model inputs
│       ├── classification_metrics.png  # Per-class precision, recall, and F1
│       ├── confusion_matrix.png
│       └── roc_curve.png
│
├── results/
│   ├── metrics.txt               # Human-readable evaluation report
│   └── classification_report.txt # Per-class evaluation report
│
└── uploads/                      # Temporary folder for uploaded CSVs (auto-created)
```

---

## Technologies Used

| Category          | Library / Tool         | Version  |
|-------------------|------------------------|----------|
| Language          | Python                 | 3.11+    |
| Web Framework     | Flask + Werkzeug       | 3.0.3    |
| Deep Learning     | TensorFlow / Keras     | 2.16.1   |
| Data Processing   | Pandas                 | 2.2.2    |
| Numerical         | NumPy                  | 1.26.4   |
| ML Utilities      | Scikit-learn           | 1.5.1    |
| Serialisation     | Joblib                 | 1.4.2    |
| Visualisation     | Matplotlib             | 3.8.4    |
| Frontend          | Bootstrap 5, HTML5, CSS3, JavaScript | CDN |

---

## Prerequisites

- **Python 3.11 or 3.12** — [python.org/downloads](https://www.python.org/downloads/)
- **pip** (bundled with Python)
- Internet access is only required to download the dataset and Bootstrap 5 CDN assets on first page load.

Verify the active Python version. Use Python 3.11 or 3.12; Python 3.14 is not supported by the pinned TensorFlow version:

```powershell
python --version
```

---

## Installation

### Step 1: Open the project folder

From the workspace folder in PowerShell:

```powershell
cd "C:\Users\user\Downloads\arss\real time\AI_Network_Intrusion_Detection"
```

Run the following commands from the project folder, where `requirements.txt` is located.

### Step 2 — Create a Python 3.11 virtual environment

```powershell
# Create the environment with Python 3.11, even if another Python version is the default
py -3.11 -m venv .venv_run
.\.venv311\Scripts\Activate.ps1                           
python --version
```

On macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```powershell
python -m pip install -r requirements.txt
```

This installs Flask, TensorFlow, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and Joblib with pinned versions.

TensorFlow 2.16.1 runs on CPU in this Windows setup. GPU use requires a separately configured supported environment; installing this requirements file alone does not guarantee GPU acceleration.

---

## Dataset Setup

The UNSW-NB15 dataset is publicly available from the University of New South Wales.

### Download

1. Visit the official dataset page:  
   **https://research.unsw.edu.au/projects/unsw-nb15-dataset**

2. Download both official files:
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`

3. Place both in the project's `dataset/` folder:

```
dataset/
├── UNSW_NB15_training-set.csv
└── UNSW_NB15_testing-set.csv
```

Both CSV files must contain the standard UNSW-NB15 features and binary `label` column. The test file is kept separate; it is not used to fit encoders, the scaler, feature selection, or model weights.

---

## Training the Model

Once the dataset files are in place, run the training script from the **project root**:

```powershell
python models\train.py
```

### What happens during training

| Step | Action |
|------|--------|
| 1 | Load the official training CSV and separate official testing CSV |
| 2 | Fit categorical encoders on training rows, then apply them to test rows |
| 3 | Fit MinMax scaling on training features only and save `models/scaler.pkl` |
| 4 | Select the top 8 features from the 42 available using a 250-tree selector fitted on the training portion |
| 5 | Create a stratified validation split from training rows and keep the decision threshold fixed at 0.50 |
| 6 | Train the four-hidden-layer DNN for up to 20 epochs with early stopping |
| 7 | Evaluate the trained DNN once on the separate testing file |
| 8 | Generate classification-metric, confusion-matrix, and ROC charts from every row in the test file |
| 9 | Save full official-test evaluation reports and charts under `models/`, `results/`, and `static/images/` |

Training accuracy is calculated on the 85% training portion used to fit the DNN. Testing accuracy and the evaluation charts are calculated separately on all rows in the test file. The confusion matrix and ROC curve do not use a sample.

### Current Training Configuration

| Setting | Value |
|---------|-------|
| Input | 8 features selected from 42 using the training portion only |
| Model | DNN with Dense layers: 256 → 128 → 64 → 32 |
| Feature selection | Extra Trees selects 8 features from the training portion |
| Maximum epochs | 20; early stopping may end training sooner |
| Batch size | 128 |
| Optimizer | Adam, learning rate 0.001 |
| Validation | 15% of training file, stratified, random seed 42 |
| Decision threshold | Fixed at 0.50 |

### Expected training time

Training time depends on the computer. The separate testing file is not used for feature selection or DNN training.

Training scores depend on the dataset and run. Use the metrics printed by your run and saved in `results/metrics.txt`; validation performance is not the official test result. A 98% score is not guaranteed. The dashboard only marks a model ready after training completes with the official testing file.

---

## Running the Web Application

After training is complete, start the Flask server:

```powershell
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

The application binds to **127.0.0.1**. The HTML references Bootstrap and Bootstrap Icons from CDNs, so those assets need internet access unless cached by the browser.

To stop the server press `Ctrl + C` in the terminal.

---

## Using the Application

### Home Page (`/`)

- Displays the project title, description, features overview, and a "How It Works" walkthrough.
- A status badge in the navbar shows whether the model is trained and ready.
- Click **Start Prediction** to go to the upload page.

### Upload Page (`/upload`)

- Drag and drop a CSV file onto the drop zone, or click to browse.
- Only `.csv` files up to **150 MB** are accepted (increased to handle UNSW-NB15 datasets).
- Click **Predict** to submit the file for analysis.
- The button shows a spinner while the model processes the file.
- **Watch the terminal window** for progress messages during processing.

### Result Page (`/result`)

Displays a full analysis including:

| Section | Content |
|---------|---------|
| Overall Verdict | Security alert: Attack if at least one row is predicted Attack; otherwise Normal |
| Traffic Counts | Number of Attack vs Normal rows |
| Model metrics | Training accuracy and separate testing accuracy, plus Precision, Recall, F1-Score, and ROC-AUC |
| Evaluation graphs | Feature rankings plus class metrics, confusion matrix, and ROC curve using every test row |
| Prediction Table | First 1,000 per-row predictions, with confidence and Attack probability; totals above include all uploaded rows |

After training, you may upload `dataset/UNSW_NB15_testing-set.csv` to view its row-level predictions. Ground-truth labels are excluded from model inputs; dashboard evaluation scores and graphs come from comparing predictions with the official test labels during training.

The table supports three filter buttons: **All**, **Attack Only**, **Normal Only**.

Click **Predict Another File** to analyse a new CSV.

### How the app decides “Attack” or “Normal”

Each network record is sent through the trained DNN, which outputs an Attack probability value between 0 and 1. The application uses a fixed decision threshold of `0.50`:

- Attack probability greater than or equal to `0.50` → **Attack**
- Attack probability below `0.50` → **Normal**

In code, the decision is effectively:

```python
attack_probability = model.predict_proba(X)[0, 1]
pred = 1 if attack_probability >= decision_threshold else 0
# 1 = Attack, 0 = Normal
```

The file-level status is a security alert: **Attack** if at least one row crosses the threshold, even when Attack rows are a minority. The separate Attack and Normal counts and rates show how many rows were flagged. This rule is intentionally conservative; a model alert is not proof of an incident, and false positives are possible.

The confidence shown in the results is based on that probability:

- For an **Attack** prediction, confidence is approximately `prob * 100%`
- For a **Normal** prediction, confidence is approximately `(1 - prob) * 100%`

This is the exact rule used by the project in `models/predict.py`.

---

## Evaluation Metrics

Training accuracy is measured on the 85% training portion used to fit the DNN. Testing accuracy is measured separately on the test file. Both use the fixed 0.50 decision threshold and are saved to `results/metrics.txt`. Training accuracy may be higher because the model has already seen those rows. In the classification report, `Recall` in the metrics summary is the Attack-class recall; per-class and macro/weighted results are in `results/classification_report.txt`.

| Metric     | Description |
|------------|-------------|
| Accuracy   | Fraction of correctly classified samples |
| Precision  | TP / (TP + FP) — how precise attack detections are |
| Recall     | TP / (TP + FN) — what fraction of actual attacks are caught |
| F1-Score   | Harmonic mean of Precision and Recall |
| ROC-AUC    | Area Under the ROC Curve — overall discriminative ability |

Use the latest run's `results/metrics.txt` and `results/classification_report.txt` for your project report. Historical scores are not results for the current official-split model. Benchmark results do not guarantee performance on live network traffic.

---

## Generated Artefacts

After running `python models/train.py` the following files are created automatically:

```
models/
├── intrusion_model.keras   # Trained DNN
├── scaler.pkl              # MinMaxScaler fitted on training data
├── encoders.pkl            # LabelEncoders for categorical columns
├── decision_threshold.pkl  # Fixed 0.50 Attack probability threshold
├── feature_names.pkl       # Ordered 8 selected DNN input features
└── all_feature_names.pkl  # Full ordered feature list used by preprocessing

static/images/
├── feature_importance_selected.png   # Extra Trees rankings for the 8 selected inputs
├── training_accuracy.png             # DNN training and validation accuracy by epoch
├── training_loss.png                 # DNN training and validation loss by epoch
├── classification_metrics.png        # Per-class test precision, recall, and F1
├── confusion_matrix.png              # Test-set confusion matrix
└── roc_curve.png                     # Test-set ROC curve

results/
├── metrics.txt                       # Full test evaluation report
└── classification_report.txt         # Full test class report
```

---

## Model

```
8 selected features from the 42 available traffic features
    │
    ▼
Deep Neural Network (256 → 128 → 64 → 32)
    │
    ▼
Attack probability → fixed 0.50 threshold → Normal or Attack
```

Extra Trees selects the 8 features using training rows. The DNN trains for up to
20 epochs, with early stopping based on validation loss. The threshold remains
fixed at 0.50. The separate test file is reserved for final evaluation and is
not used for feature selection or DNN training.

---

## Troubleshooting

### ⚠️ **IMPORTANT: Python Version Requirement**

Use Python 3.11 or 3.12 for the pinned requirements. Python 3.14 is not supported by the pinned TensorFlow package.

✅ **Solution:** Use a virtual environment created with Python 3.11 or 3.12. Do not use `venv_py311` if it reports Python 3.14.

```powershell
# From the project folder, create the Python 3.11 environment if needed
py -3.11 -m venv .venv_run
.\.venv_run\Scripts\Activate.ps1
python --version  # Must show Python 3.11 or 3.12

# Install dependencies into this environment if not already installed
python -m pip install -r requirements.txt

# Then run the app
python app.py
```

---

### Common Issues & Solutions

#### 1. **Out of Memory Error**
**Problem:** Training or prediction runs out of memory on a large dataset/upload.

CSV input is read in chunks of 10,000 rows to reduce parsing memory. Model inputs and predictions still require memory, so chunking cannot guarantee that very large files will fit.

**If still occurring:**
- Try a smaller CSV and close other memory-intensive applications.
- Reduce the upload preprocessing chunk size in `src/preprocess.py`:
  ```python
  chunksize: int = 5000  # Reduce from 10000
  ```

---

#### 2. **Python Version Issues**
**Error:** `Could not find a version that satisfies the requirement tensorflow`

**Cause:** Using Python 3.14+ which is not supported by TensorFlow 2.16.1

✅ **Solution:** Use a Python 3.11 or 3.12 virtual environment.
```bash
# Check if Python 3.11 is available
py -0

# Create new virtual environment with Python 3.11
py -3.11 -m venv .venv_run

# Activate and install packages
.\.venv_run\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
```

---

#### 3. **Missing Packages**
**Error:** `ModuleNotFoundError: No module named 'flask'`

✅ **Solution:**
```bash
# Activate virtual environment first
.\.venv_run\Scripts\Activate.ps1

# Install all packages
python -m pip install -r requirements.txt
```

---

#### 4. **Output Not Showing**

**Problem:** Upload works but no progress in terminal or no results in browser

Progress messages are printed by the preprocessing and prediction code. Look for output similar to:
  ```
  [preprocess] Processing uploaded file in chunks of 10000 rows...
  [preprocess] Processed chunk 1: 10000 rows (Total: 10000)
  [predict] Running inference on 50000 rows in batches of 1000...
  ```

**Where to look:**
- **Terminal window** - Progress messages and status
- **Browser** - Final results page

---

#### 5. **Model Not Trained**
**Error:** `FileNotFoundError: Trained model not found`

✅ **Solution:** Train the model first
```bash
.\.venv_run\Scripts\Activate.ps1
python models/train.py
```

---

#### 6. **Official Dataset File Missing**
**Error:** An official UNSW-NB15 training or testing CSV was not found.

✅ **Solution:** Place both labeled official files in `dataset/`:
- `dataset/UNSW_NB15_training-set.csv`
- `dataset/UNSW_NB15_testing-set.csv`

Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
The official test file is used only for final evaluation; keep it out of training.

---

#### 7. **File Size Limit**
**Error:** `Uploaded file exceeds the 150 MB limit`

✅ **Current Limit:** 150 MB (enough for UNSW-NB15 datasets)

To increase further, edit `app.py`:
```python
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB
```

---

#### 8. **Port Already in Use**
**Error:** `Address already in use`

✅ **Solution:** Change port in `app.py`:
```python
app.run(host="127.0.0.1", port=5001, debug=True)
```

---

#### 9. **CSV Format Errors**
**Error:** `Data error: ...` or `No valid data rows after preprocessing`

✅ **Requirements:**
- CSV must have **header row** with UNSW-NB15 column names
- File must not be empty
- UTF-8 encoding recommended

### Performance Notes

Processing time depends on file size, available memory, and hardware. The upload limit is 150 MB; chunked reading does not eliminate the memory needed to hold predictions and model inputs.

---

## Quick-Start Summary

### ✅ If Everything is Already Set Up (Model Trained)

**Windows PowerShell** (run these commands from the project folder):
```powershell
.\.venv_run\Scripts\Activate.ps1
python --version
python app.py
```

Then open: **http://127.0.0.1:5000**

---

### 🆕 First Time Setup (Fresh Installation)

```powershell
# 1. Create Python 3.11 virtual environment
py -3.11 -m venv .venv_run

# 2. Activate it
.\.venv_run\Scripts\Activate.ps1

# 3. Install dependencies
python --version
python -m pip install -r requirements.txt

# 4. Download and place the UNSW-NB15 training CSV:
#    dataset/UNSW_NB15_training-set.csv
#    From: https://research.unsw.edu.au/projects/unsw-nb15-dataset

# 5. Train the model (time depends on hardware)
python models/train.py

# 6. Start the web application
python app.py

# 7. Open browser
#    http://127.0.0.1:5000
```

---

## Recent Code Notes

- Preprocessing preserves repeated rows; deleting them had changed the class distribution and removed valid records from evaluation.
- The results table uses the same class names as the model: **Normal** and **Attack**.
- Confusion-matrix and ROC plotting are implemented in `src/visualization_reporting.py`; the dashboard displays the generated PNG files.
- The latest reported score is a benchmark result on the supplied test CSV, not a guarantee for live traffic.

---

**Run commands from the project root directory. In PowerShell, quote the project path when changing directories because it contains a space:**

```powershell
Set-Location -LiteralPath "C:\Users\user\Downloads\arss\real time\AI_Network_Intrusion_Detection"
```
cd "C:\Users\DELL\Documents\arss\arss\real time\AI_Network_Intrusion_Detection"
$env:TF_ENABLE_ONEDNN_OPTS = "0"
.\.venv311\Scripts\python.exe .\models\train.py