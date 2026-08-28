"""
app.py

Guitar Tone & FX Pedal Classifier -- Web App
--------------------------------------------------------------------------
Streamlit interface: upload a guitar clip, see its waveform, and get a
live tone/effect classification from the trained model.

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import streamlit as st

from predict import TonePredictor

MODEL_DIR = Path("models")

st.set_page_config(page_title="Guitar Tone & FX Pedal Classifier", page_icon="🎸", layout="centered")


@st.cache_resource
def load_predictor() -> TonePredictor:
    """Load the trained model once per app session, not once per upload.

    joblib deserialization plus rebuilding the scaler/encoder isn't free,
    and re-running it on every interaction is exactly the kind of thing
    that works fine in a demo and then feels sluggish the moment someone
    actually uses the app.
    """
    return TonePredictor(MODEL_DIR)


def plot_waveform(y, sr: int) -> plt.Figure:
    """Render an amplitude-over-time waveform, the standard way to *look*
    at what a clip sounds like before you hear it -- useful here mainly
    as a sanity check that the right file loaded (silence, clipping, or
    an unexpectedly short clip are all visible at a glance)."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#4B4BFF")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def main() -> None:
    st.title("🎸 Guitar Tone & FX Pedal Classifier")
    st.caption(
        "Upload a short guitar clip and the model will classify which "
        "tone or effect pedal it's most likely running through."
    )

    try:
        predictor = load_predictor()
    except FileNotFoundError:
        st.error(
            "No trained model found in `models/`. Run `train.py` first, "
            "then restart this app."
        )
        st.stop()

    with st.sidebar:
        st.subheader("Model info")
        st.write(f"**Type:** {predictor.model_name.replace('_', ' ').title()}")
        st.write(f"**Classes:** {', '.join(predictor.label_encoder.classes_)}")

    uploaded = st.file_uploader("Upload a .wav clip", type=["wav"])
    if uploaded is None:
        st.info("Waiting for a clip to classify.")
        return

    # predict.py's feature extraction reads from a real file path (so it
    # can stay identical to what extract_features.py uses on disk), so the
    # in-memory upload is written to a short-lived temp file rather than
    # teaching that function to also accept a file-like object.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)

    try:
        st.audio(uploaded.getvalue(), format="audio/wav")

        y, sr = librosa.load(tmp_path, sr=None, mono=True)
        st.pyplot(plot_waveform(y, sr))

        with st.spinner("Classifying..."):
            result = predictor.predict(tmp_path)

        st.subheader(f"Prediction: {result['predicted_class']}")
        st.write(f"Confidence: {result['confidence']:.1%}")
        st.bar_chart(result["all_probabilities"])
    except Exception as exc:
        st.error(f"Couldn't process this clip: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
