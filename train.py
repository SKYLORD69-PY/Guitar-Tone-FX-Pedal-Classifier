"""
train.py

--------------------------------------------------------------------------
Loads the feature CSV produced by extract_features.py, trains and
compares a Random Forest and an SVM classifier, evaluates both on a
held-out test set (confusion matrix + precision/recall/F1), and persists
the better-performing model 

Usage
-----
    python train.py \
        --features-csv data/features/guitar_tone_features.csv \
        --model-dir models \
        --report-dir reports
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Held-out fraction for the test split -- large enough for a stable
# confusion matrix, small enough to leave most clips for training.
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 5  # below this, a stratified split isn't meaningful

NON_FEATURE_COLUMNS = {"filename", "label"}


@dataclass
class ModelResult:
    """Bundles one trained model with everything needed to judge it and,
    if it wins, persist it."""

    name: str
    estimator: object
    accuracy: float
    macro_f1: float
    report_text: str
    y_pred: np.ndarray


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the feature CSV and split it into X (features) and
    y (string class labels), and sanity-check class sizes."""
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    if not feature_cols:
        raise ValueError(f"No feature columns found in {csv_path} -- is this the right file?")

    X = df[feature_cols]
    y = df["label"]

    counts = y.value_counts()
    too_small = counts[counts < MIN_SAMPLES_PER_CLASS]
    if not too_small.empty:
        raise ValueError(
            "These classes have too few clips for a reliable train/test split "
            f"(need >= {MIN_SAMPLES_PER_CLASS} each): "
            f"{too_small.to_dict()}. Gather a few more clips, or merge the class "
            "into a related one, before training."
        )

    logger.info("Loaded %d clips x %d features across %d classes", *X.shape, y.nunique())
    logger.info("Class distribution:\n%s", counts.to_string())
    return X, y


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], title: str, out_path: Path) -> None:
    """Render and save a confusion matrix as a PNG.

    Rows are the true class, columns are what the model predicted. A
    strong diagonal means the model is doing well; off-diagonal mass
    shows exactly which classes get confused for which -- e.g. it's
    normal and expected for chorus/flanger/phaser to bleed into each
    other here, since they're all subtle modulation effects and none of
    our four features explicitly captures modulation rate.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, xticks_rotation=45, ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train_and_evaluate(
    name: str,
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> ModelResult:
    """Fit one model and score it on the held-out test set."""
    logger.info("Training %s...", name)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    accuracy = float(np.mean(y_pred == y_test))
    # Macro-F1 (unweighted average across classes), not plain accuracy, to
    # pick the "better" model -- merging pedals into classes (e.g. 3
    # distortion pedals -> 1 class) can leave some classes with more clips
    # than others, and macro-F1 stops a big class from hiding how badly a
    # small one is doing.
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    report_text = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    logger.info("%s -- accuracy: %.3f | macro F1: %.3f", name, accuracy, macro_f1)
    return ModelResult(name, estimator, accuracy, macro_f1, report_text, y_pred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train & compare Random Forest and SVM tone classifiers.")
    parser.add_argument("--features-csv", type=Path, default=Path("data/features/guitar_tone_features.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    if not args.features_csv.exists():
        raise FileNotFoundError(
            f"{args.features_csv} not found -- run extract_features.py first."
        )

    # ---- Load + preprocess ---------------------------------------------- #
    X, y = load_dataset(args.features_csv)
    feature_names = list(X.columns)

    # Models need numeric labels; the encoder also lets predict.py turn a
    # prediction back into a human-readable class name.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y_encoded,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_encoded,  # keep each class's proportion consistent across the split
    )

    # SVM (especially with an RBF kernel) is sensitive to feature scale;
    # Random Forest doesn't need scaling but isn't hurt by it either, so one
    # shared scaler keeps a single preprocessing path for both models.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Train + evaluate both candidates -------------------------------- #
    candidates = [
        train_and_evaluate(
            "random_forest",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=args.random_state,
            ),
            X_train_scaled, y_train, X_test_scaled, y_test, class_names,
        ),
        train_and_evaluate(
            "svm",
            SVC(
                kernel="rbf", C=5, gamma="scale", probability=True,
                class_weight="balanced", random_state=args.random_state,
            ),
            X_train_scaled, y_train, X_test_scaled, y_test, class_names,
        ),
    ]

    for result in candidates:
        print(f"\n=== {result.name} ===\n{result.report_text}")
        plot_confusion_matrix(
            y_test, result.y_pred, class_names,
            title=f"Confusion Matrix -- {result.name.replace('_', ' ').title()}",
            out_path=args.report_dir / f"confusion_matrix_{result.name}.png",
        )

    # ---- Pick + persist the winner ---------------------------------------#
    best = max(candidates, key=lambda r: r.macro_f1)
    logger.info("Best model: %s (macro F1 = %.3f)", best.name, best.macro_f1)

    joblib.dump(best.estimator, args.model_dir / "best_model.joblib")
    joblib.dump(scaler, args.model_dir / "scaler.joblib")
    joblib.dump(label_encoder, args.model_dir / "label_encoder.joblib")

    metadata = {
        "best_model": best.name,
        "accuracy": best.accuracy,
        "macro_f1": best.macro_f1,
        "classes": class_names,
        "feature_names": feature_names,
        "all_candidates": {r.name: {"accuracy": r.accuracy, "macro_f1": r.macro_f1} for r in candidates},
    }
    with open(args.model_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved best_model.joblib, scaler.joblib, label_encoder.joblib -> %s", args.model_dir)
    logger.info("Saved confusion matrices -> %s", args.report_dir)


if __name__ == "__main__":
    main()
