# Guitar Tone & FX Pedal Classifier

A lightweight audio classification project that identifies guitar tones and effects from short audio clips using handcrafted audio features and classical machine learning.

This project is built around Librosa and scikit-learn, with a simple pipeline for organizing labeled recordings, extracting timbral features, and preparing data for classification.

## What this project does

The goal is to classify guitar audio into categories such as:

- clean
- distortion
- chorus
- delay
- reverb
- phaser
- flanger

Instead of training on raw waveforms, the pipeline extracts interpretable audio descriptors such as MFCCs, spectral centroid, zero-crossing rate, and RMS energy. These features are then saved as a tabular dataset ready for model training.

## Repository overview

- `organize_dataset.py` – reorganizes downloaded effect datasets into the class-based folder structure expected by the pipeline.
- `extract_features.py` – loads labeled `.wav` files, extracts audio features, and writes a CSV dataset.
- `data/raw_audio/` – source audio grouped by class label.
- `data/features/` – generated feature dataset output.

## Dataset structure

Place your labeled audio data in this layout:

```text
data/
├── raw_audio/
│   ├── clean/
│   ├── distortion/
│   ├── chorus/
│   ├── delay/
│   ├── reverb/
│   ├── phaser/
│   └── flanger/
└── features/
```

Each class folder contains `.wav` files for that tone or effect.

## Setup

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare the dataset.

If you are using the EGFxSet dataset, run:

```bash
python organize_dataset.py --source path/to/unzipped/EGFxSet --dest data/raw_audio
```

If you already have your own labeled recordings, place them in `data/raw_audio/<class_name>/` manually.

## Feature extraction

Run:

```bash
python extract_features.py --data-dir data/raw_audio --output data/features/guitar_tone_features.csv
```

This script extracts per-clip summary statistics for:

- MFCC coefficients
- spectral centroid
- zero-crossing rate
- RMS energy

The result is a CSV with one row per audio clip and one column per extracted feature, along with the filename and label.

## Why this approach

This is a classic machine-learning workflow for audio classification:

- feature engineering instead of end-to-end deep learning
- interpretable signal descriptors
- fast experimentation with tabular models
- easy extension for additional classes or features

## Requirements

The project depends on:

- Python 3
- librosa
- numpy
- pandas
- scikit-learn
- tqdm

## Notes

This repository focuses on the data preparation and feature extraction stages. The project is designed to be a clean foundation for building a classifier on top of the generated feature table.
