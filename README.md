# Guitar Tone & FX Pedal Classifier

A classical ML pipeline (Librosa + scikit-learn — no PyTorch/TensorFlow) that classifies a guitar clip's tone or effect pedal from hand-crafted audio features: MFCCs, spectral centroid, zero-crossing rate, and RMS energy.

## Pipeline

1. **Dataset preparation** — [`organize_dataset.py`](organize_dataset.py) sorts a downloaded [EGFxSet](https://egfxset.github.io/) extraction into `data/raw_audio/<class>/*.wav`.
2. **Feature extraction** — [`extract_features.py`](extract_features.py) computes MFCCs, spectral centroid, zero-crossing rate, and RMS energy (mean + std) per clip into a single feature CSV.
3. **Model training** — [`train.py`](train.py) trains and compares a Random Forest and an SVM on the same stratified split, evaluates both (confusion matrix, precision/recall/F1), and saves the stronger model.
4. **Inference** — [`predict.py`](predict.py) loads the saved model and classifies a single new clip, via a reusable `TonePredictor` class.
5. **Web app** — [`app.py`](app.py) is a Streamlit interface for uploading a clip, viewing its waveform, and seeing a live classification.

## Setup

Navigate to the project root directory:

```bash
cd "e:\Projects - ML\Guitar Tone Classifier"
pip install -r requirements.txt
```

Then run the pipeline:

```bash
# 1. Get labeled training audio
# Replace <YOUR_EGFXSET_PATH> with the actual path to your extracted EGFxSet folder
python organize_dataset.py --source "<YOUR_EGFXSET_PATH>" --dest "data\raw_audio"

# Example:
# python organize_dataset.py --source "D:\Downloads\EGFxSet-master" --dest "data\raw_audio"

# 2. Extract features
python extract_features.py --data-dir "data\raw_audio" --output "data\features\guitar_tone_features.csv"

# 3. Train + evaluate
python train.py --features-csv "data\features\guitar_tone_features.csv"

# 4. Classify a clip from the command line
# Replace <YOUR_WAV_FILE> with the actual path to your audio file
python predict.py --input "<YOUR_WAV_FILE>"

# Example:
# python predict.py --input "D:\audio_samples\my_guitar_clip.wav"

# 5. Or launch the web app
streamlit run app.py
```

## Project structure

```
.
├── data/
│   ├── raw_audio/<class>/   # training clips, one folder per tone/effect label
│   └── features/            # extract_features.py output
├── models/                  # train.py output: best model, scaler, label encoder
├── reports/                 # train.py output: confusion matrices
├── organize_dataset.py
├── extract_features.py
├── train.py
├── predict.py
├── app.py
└── requirements.txt
```
