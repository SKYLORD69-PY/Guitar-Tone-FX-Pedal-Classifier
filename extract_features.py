"""
extract_features.py

Walks a directory of labeled guitar audio clips (one sub-folder per
tone/effect class), extracts a set of hand-crafted audio features with
Librosa, and writes a single tabular CSV for scikit-learn (Random Forest /
SVM).

Expected folder layout
-----------------------
    data/raw_audio/
        clean/
            take_001.wav
            take_002.wav
        distortion/
            take_001.wav
            ...
        chorus/
            ...

Each sub-folder name under --data-dir becomes the class label. The
script does not care how many classes there are or what they're called --
it just reads the directory tree.

Usage
-----
    python extract_features.py \
        --data-dir data/raw_audio \
        --output data/features/guitar_tone_features.csv
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Every clip is resampled to this rate before feature extraction. Guitar
# tone/FX differences live almost entirely below ~10 kHz, so 22050 Hz
# (Librosa's own default) is plenty of resolution and keeps extraction
# fast without losing anything perceptually relevant.
DEFAULT_SAMPLE_RATE = 22_050

# Number of MFCC coefficients to compute. 13 is the classic MIR/speech
# default: it captures the coarse shape of the spectral envelope (i.e.
# timbre) without chasing the fine pitch-level detail that would tie the
# features to *what note was played* rather than *how it sounds*.
DEFAULT_N_MFCC = 13

AUDIO_EXTENSIONS = {".wav"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureConfig:
    """Groups the knobs that affect feature extraction so they travel
    together instead of as loose function arguments scattered everywhere."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    n_mfcc: int = DEFAULT_N_MFCC


DEFAULT_CONFIG = FeatureConfig()


def extract_features(file_path: Path, config: FeatureConfig = DEFAULT_CONFIG) -> dict[str, float] | None:
    """Extract a fixed-length feature vector from one audio clip.

    Every feature below is computed frame-by-frame across the clip, then
    summarised with its mean and standard deviation:
      - the MEAN captures the "average" character of the tone
      - the STD captures how much that character *moves* over the clip
        (a modulation effect like chorus/tremolo sweeps over time and
        will show higher variance than a static clean tone)

    Parameters
    ----------
    file_path : Path
        Path to a .wav file (mono or stereo).
    config : FeatureConfig
        Sample rate / MFCC count to use for this extraction.

    Returns
    -------
    dict[str, float] | None
        Flat dict of feature_name -> value, or None if the file couldn't
        be read (corrupt/empty clip), so the caller can skip it and keep
        going instead of crashing the whole batch.
    """
    try:
        # mono=True downmixes stereo to mono. Effect *colour* doesn't live
        # in stereo placement, and mono guarantees every clip produces a
        # feature vector of the same shape regardless of source channel
        # count -- important since scikit-learn needs a fixed-width table.
        y, sr = librosa.load(file_path, sr=config.sample_rate, mono=True)
    except Exception as exc:  # librosa/soundfile can raise several types
        logger.warning("Skipping unreadable file %s (%s)", file_path, exc)
        return None

    if y.size == 0:
        logger.warning("Skipping empty audio file %s", file_path)
        return None

    features: dict[str, float] = {}

    # ---- MFCCs (Mel-Frequency Cepstral Coefficients) --------------------- #
    # MFCCs describe the *shape* of the spectral envelope on a scale that
    # approximates human pitch perception. In practice they're the
    # standard "timbre fingerprint": two notes at the same pitch and
    # loudness but different tone colour (clean vs. fuzzy vs. chorused)
    # will show visibly different MFCC profiles. We keep the first 13
    # coefficients -- the coarse envelope shape; higher coefficients
    # mostly encode fine pitch detail we don't want here.
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.n_mfcc)
    for i in range(config.n_mfcc):
        features[f"mfcc_{i + 1}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i + 1}_std"] = float(np.std(mfccs[i]))

    # ---- Spectral Centroid ------------------------------------------------ #
    # The "centre of mass" of the frequency spectrum, in Hz. Musically this
    # tracks perceived *brightness*. A clean, bassy tone sits lower; an
    # overdriven or fuzzed tone -- which adds odd/even harmonics well above
    # the fundamental -- pulls the centroid upward.
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    # ---- Zero-Crossing Rate (ZCR) ------------------------------------------#
    # How often the waveform crosses zero amplitude, per frame. Roughly
    # tracks noisiness / harmonic density: clipping-based effects
    # (distortion, fuzz, overdrive) generate lots of extra high-frequency
    # harmonics and push ZCR up; a clean sustained note has a lower, more
    # regular ZCR.
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # ---- RMS Energy --------------------------------------------------------#
    # Frame-wise loudness (root-mean-square amplitude). Effects with
    # built-in compression/sustain (distortion, overdrive) tend to flatten
    # and sustain the RMS envelope; time-based effects like reverb and
    # delay leave a characteristic decaying "tail" that shows up as
    # elevated RMS *variance* after the note has technically stopped ringing.
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    return features


def build_dataset(data_dir: Path, config: FeatureConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Walk `data_dir`, extract features for every clip, and return a tidy
    DataFrame: one row per clip, one column per feature, plus `filename`
    and `label` columns.

    The label is simply the name of each clip's immediate parent folder,
    so the on-disk layout *is* the labelling mechanism -- no separate
    annotation file needed:

        data_dir/clean/clip1.wav       -> label = "clean"
        data_dir/distortion/clip1.wav  -> label = "distortion"
    """
    class_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not class_dirs:
        raise FileNotFoundError(
            f"No class sub-folders found under {data_dir}. Expected one "
            f"folder per tone/effect label, e.g. {data_dir}/clean/*.wav"
        )

    rows: list[dict[str, float | str]] = []
    for class_dir in class_dirs:
        label = class_dir.name
        audio_files = sorted(
            f for f in class_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not audio_files:
            logger.warning("No .wav files found in %s -- skipping class '%s'", class_dir, label)
            continue

        for file_path in tqdm(audio_files, desc=f"Extracting [{label}]"):
            features = extract_features(file_path, config)
            if features is None:
                continue
            features["filename"] = file_path.name
            features["label"] = label
            rows.append(features)

    if not rows:
        raise RuntimeError("No features were extracted -- check your --data-dir contents.")

    df = pd.DataFrame(rows)
    # Put filename/label first for readability; feature columns trail
    # after in a stable, deterministic order.
    ordered_cols = ["filename", "label"] + [c for c in df.columns if c not in ("filename", "label")]
    return df[ordered_cols]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MFCC / spectral-centroid / ZCR / RMS features from labeled guitar audio clips."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw_audio"),
        help="Root folder containing one sub-folder per class label (default: data/raw_audio)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/features/guitar_tone_features.csv"),
        help="Path to write the output CSV (default: data/features/guitar_tone_features.csv)",
    )
    parser.add_argument(
        "--n-mfcc",
        type=int,
        default=DEFAULT_N_MFCC,
        help=f"Number of MFCC coefficients to extract (default: {DEFAULT_N_MFCC})",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Sample rate every clip is resampled to before extraction (default: {DEFAULT_SAMPLE_RATE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"--data-dir does not exist: {args.data_dir}")

    config = FeatureConfig(sample_rate=args.sample_rate, n_mfcc=args.n_mfcc)

    logger.info("Scanning %s for class folders...", args.data_dir)
    df = build_dataset(args.data_dir, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    logger.info("Wrote %d rows x %d columns to %s", df.shape[0], df.shape[1], args.output)
    logger.info("Classes found: %s", sorted(df["label"].unique()))


if __name__ == "__main__":
    main()
