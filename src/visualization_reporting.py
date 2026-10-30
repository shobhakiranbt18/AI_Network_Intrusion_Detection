"""Generate concise model evaluation charts for the NIDS dashboard."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

from src.evaluate import predict_attack_probabilities

IMAGE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "static", "images")
)
CLASS_NAMES = ["Normal (0)", "Attack (1)"]


def _validate_evaluation_rows(y_true, probabilities):
    """Validate and return all evaluation labels and probabilities."""
    y_true = np.asarray(y_true).reshape(-1)
    probabilities = np.asarray(probabilities).reshape(-1)
    if len(y_true) != len(probabilities):
        raise ValueError("Labels and prediction probabilities must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Cannot generate evaluation charts from an empty test set.")
    return y_true, probabilities


def _evaluation_scope(test_count):
    return f"All {test_count:,} test records"


def plot_classification_metrics(y_true, y_pred, output_path, scope_label):
    """Save class-wise precision, recall, and F1 scores for binary classification."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    positions = np.arange(len(CLASS_NAMES))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5.5))
    groups = (
        ax.bar(positions - width, precision, width, label="Precision", color="#58b4d1"),
        ax.bar(positions, recall, width, label="Recall", color="#58b368"),
        ax.bar(positions + width, f1, width, label="F1-score", color="#e87868"),
    )
    for group in groups:
        ax.bar_label(group, fmt="%.2f", padding=3, fontsize=9)

    # X-axis labels with sample counts below class names
    ax.set_xticks(positions, 
                  [f"{name}\n({count:,} samples)" for name, count in zip(CLASS_NAMES, support)],
                  fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(f"Binary Classification Metrics\n{scope_label}", fontsize=11, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, output_path, scope_label):
    """Save a labeled confusion matrix with exact sample counts."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_NAMES)
    # Use light blue-white colormap for better readability
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.grid(False)
    
    # Enhanced title with Binary Classification emphasis
    ax.set_title(f"Binary Classification - Confusion Matrix\nNormal (0) vs Attack (1)\n{scope_label}", 
                 fontsize=16, fontweight="bold", color="black", pad=20)
    ax.set_xlabel("Predicted Label", fontsize=13, color="black", fontweight="bold")
    ax.set_ylabel("True Label", fontsize=13, color="black", fontweight="bold")
    ax.tick_params(colors="black")
    
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)
    
    # Force text colors to be black for better readability on white background
    for text in display.text_.ravel():
        text.set_color("black")
        text.set_fontsize(16)
        text.set_fontweight("bold")
    
    # Accuracy annotation removed as requested
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white", edgecolor='black')
    plt.close(fig)


def plot_roc_curve(y_true, probabilities, output_path, scope_label):
    """Save ROC curve and AUC using every test row."""
    if np.unique(y_true).size != 2:
        raise ValueError("ROC curve requires both classes in the evaluation rows.")
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    score = auc(false_positive_rate, true_positive_rate)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    # ROC curve with blue theme on white background
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        color="#2563eb",  # Darker blue for visibility
        linewidth=3.5,
        label=f"ROC Curve (AUC = {score:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="#6b7280", linewidth=2.0, linestyle="--",
            label="Random Classifier")
    
    # Light blue fill
    ax.fill_between(false_positive_rate, true_positive_rate, alpha=0.15, color="#3b82f6")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=13, color="black", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=13, color="black", fontweight="bold")
    
    # Enhanced title with Binary Classification
    ax.set_title(f"Binary Classification - ROC Curve\nNormal (0) vs Attack (1)\n{scope_label}",
                 fontsize=16, fontweight="bold", color="black", pad=20)
    
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)
    
    legend = ax.legend(loc="lower right", fontsize=12, framealpha=0.9, labelcolor="black")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    ax.grid(True, color="#d1d5db", linestyle="--", linewidth=0.8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_visualization_reports(
    model,
    X_test,
    y_test,
    feature_names,
    threshold=0.5,
    output_dir=None,
    history=None,
    feature_importances=None,
):
    """Generate held-out test charts and feature importance from the fitted model."""
    image_dir = os.path.abspath(output_dir or IMAGE_DIR)
    os.makedirs(image_dir, exist_ok=True)
    y_test = np.asarray(y_test).reshape(-1)
    probabilities = predict_attack_probabilities(model, X_test)
    if len(y_test) != len(probabilities):
        raise ValueError("Test labels and model predictions must have the same length.")

    test_labels, test_probabilities = _validate_evaluation_rows(y_test, probabilities)
    test_predictions = (test_probabilities >= threshold).astype(int)
    scope = _evaluation_scope(len(test_labels))
    print(f"[visualization] {scope}")

    plot_classification_metrics(
        test_labels,
        test_predictions,
        os.path.join(image_dir, "classification_metrics.png"),
        scope,
    )
    plot_confusion_matrix(
        test_labels,
        test_predictions,
        os.path.join(image_dir, "confusion_matrix.png"),
        scope,
    )
    plot_roc_curve(
        test_labels,
        test_probabilities,
        os.path.join(image_dir, "roc_curve.png"),
        scope,
    )

    if history is not None:
        for metric, validation_metric, title, filename in (
            ("accuracy", "val_accuracy", "Training and Validation Accuracy", "training_accuracy.png"),
            ("loss", "val_loss", "Training and Validation Loss", "training_loss.png"),
        ):
            values = history.history
            if metric not in values or validation_metric not in values:
                continue
            
            # Limit display to first 30 epochs
            epochs_to_show = min(30, len(values[metric]))
            epochs = range(1, epochs_to_show + 1)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(epochs, values[metric][:epochs_to_show], label="Training", 
                    color="#1f77b4", linewidth=2.5, marker='o', markersize=4)
            ax.plot(epochs, values[validation_metric][:epochs_to_show], label="Validation", 
                    color="#ff7f0e", linewidth=2.5, marker='s', markersize=4)
            
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
            ax.set_ylabel(metric.title(), fontsize=11, fontweight="bold")
            ax.set_xlim(1, epochs_to_show)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(frameon=True, loc='best', fontsize=10, shadow=True)
            
            # Add note if more epochs were trained
            if len(values[metric]) > 30:
                ax.text(0.98, 0.02, f'Showing first 30 of {len(values[metric])} epochs', 
                       transform=ax.transAxes, fontsize=9, ha='right', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            fig.tight_layout()
            fig.savefig(os.path.join(image_dir, filename), dpi=160, bbox_inches="tight")
            plt.close(fig)

    if feature_importances is None:
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
        else:
            raise ValueError("Feature importances are required for this model type.")
    importances = np.asarray(feature_importances)
    if len(importances) != len(feature_names):
        raise ValueError("Feature names do not match the training matrix columns.")

    order = np.argsort(importances)[-min(8, len(feature_names)):]
    fig, ax = plt.subplots(figsize=(8, max(4, len(order) * 0.42)))
    ax.barh(np.asarray(feature_names)[order], importances[order], color="#3977a8")
    ax.set(title="Most Important Model Features", xlabel="Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        os.path.join(image_dir, "feature_importance_selected.png"),
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)
