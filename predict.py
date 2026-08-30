"""
predict.py

Guitar Tone & FX Pedal Classifier
--------------------------------------------------------------------------
Loads the model/scaler/label-encoder saved by train.py and predicts the
 tone/effect class of a single new .wav clip.

Usage
-----
    python predict.py --input path/to/some_clip.wav
    python predict.py --input path/to/some_clip.wav --top-k 3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np

# Reuse the exact feature-extraction logic -- never reimplement it here.
# Two copies of "how to turn a clip into numbers" will eventually drift
# apart, and a silent mismatch here wouldn't crash, it would just quietly
# produce wrong predictions.
from extract_features import DEFAULT_SAMPLE_RATE, FeatureConfig, extract_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TonePredictor:
    """Wraps the three artifacts train.py saved (model, scaler, label
    encoder) so a caller just hands it a file path and gets a class name
    back.

    Deliberately a class, not just a script: a future Streamlit app can
    import this directly and instantiate it once per session, instead of
    shelling out to the CLI and reloading the model from disk on every
    single upload.
    """

    def __init__(self, model_dir: Path = Path("models"), sample_rate: int = DEFAULT_SAMPLE_RATE):
        required = ["best_model.joblib", "scaler.joblib", "label_encoder.joblib", "model_metadata.json"]
        missing = [f for f in required if not (model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing {missing} in {model_dir}/ -- run train.py first."
            )

        self.model = joblib.load(model_dir / "best_model.joblib")
        self.scaler = joblib.load(model_dir / "scaler.joblib")
        self.label_encoder = joblib.load(model_dir / "label_encoder.joblib")

        with open(model_dir / "model_metadata.json") as f:
            metadata = json.load(f)
        # The exact column order the scaler/model were fit on -- every new
        # clip gets reindexed to match THIS, not whatever key order
        # extract_features() happens to build its dict in.
        self.feature_names: list[str] = metadata["feature_names"]
        self.model_name: str = metadata["best_model"]
        self.metadata: dict = metadata  # full record (incl. all_candidates) for callers like app.py

        # n_mfcc is fully recoverable from the saved column names (count the
        # mfcc_N_mean entries); sample_rate isn't -- it only changes the
        # computed *values*, not the column names, so there's nothing in
        # metadata to recover it from. If feature extraction used a
        # non-default --sample-rate, pass that same value here (or via
        # --sample-rate on the CLI below).
        n_mfcc = sum(1 for name in self.feature_names if name.startswith("mfcc_") and name.endswith("_mean"))
        self.feature_config = FeatureConfig(sample_rate=sample_rate, n_mfcc=n_mfcc)

    def predict(self, wav_path: Path) -> dict:
        """Extract features from one clip and return a prediction with
        per-class confidence, ranked highest first."""
        features = extract_features(wav_path, self.feature_config)
        if features is None:
            raise ValueError(f"Could not read audio from {wav_path}")

        missing = [name for name in self.feature_names if name not in features]
        if missing:
            raise RuntimeError(
                f"Extracted features don't match what the model was trained on "
                f"(missing: {missing}). If feature extraction used a non-default --n-mfcc or "
                f"--sample-rate, pass the same value here."
            )

        # Reindex to the training-time column order -- the step a silent
        # feature-order mismatch would otherwise skip right past.
        vector = np.array([[features[name] for name in self.feature_names]])
        vector_scaled = self.scaler.transform(vector)

        probabilities = self.model.predict_proba(vector_scaled)[0]
        class_names = self.label_encoder.classes_
        ranked = sorted(zip(class_names, probabilities), key=lambda p: p[1], reverse=True)

        return {
            "predicted_class": ranked[0][0],
            "confidence": float(ranked[0][1]),
            "all_probabilities": {name: float(p) for name, p in ranked},
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a guitar clip's tone/effect class.")
    parser.add_argument("--input", type=Path, required=True, help="Path to a .wav clip to classify")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
        help="Must match the --sample-rate used during feature extraction (default: same default)",
    )
    parser.add_argument("--top-k", type=int, default=3, help="How many ranked classes to display")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"--input not found: {args.input}")

    predictor = TonePredictor(args.model_dir, sample_rate=args.sample_rate)
    result = predictor.predict(args.input)

    print(f"\nModel: {predictor.model_name}")
    print(f"File:  {args.input.name}")
    print(f"\nPrediction: {result['predicted_class']}  ({result['confidence']:.1%} confidence)\n")

    print(f"Top {args.top_k}:")
    for name, prob in list(result["all_probabilities"].items())[: args.top_k]:
        bar = "#" * int(round(prob * 30))
        print(f"  {name:<12} {prob:>6.1%}  {bar}")


if __name__ == "__main__":
    main()
