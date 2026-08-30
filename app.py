"""
app.py

Guitar Tone & FX Pedal Classifier -- Web App
--------------------------------------------------------------------------
Streamlit interface: upload a guitar clip (or pick a bundled example),
watch it move through the signal chain, and get a live tone/effect
reading from the trained model.

Visual language: the page is built like a rack unit, not a generic
dashboard. Panels read like stompbox enclosures, a lit LED marks each
active stage, and the confidence display is a segmented VU-meter ladder
rather than a plain progress bar. Color is not decorative: warm amber
marks gain-stage effects (distortion/overdrive/fuzz), cool teal marks
modulation/time effects (chorus/flanger/phaser/delay/reverb) -- so the
color itself tells you something true about what the model predicted.

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
import numpy as np
import streamlit as st

from predict import TonePredictor

MODEL_DIR = Path("models")
EXAMPLES_DIR = Path("examples")

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #

COLORS = {
    "bg": "#141217",
    "panel": "#1D1A21",
    "panel_border": "#2E2A35",
    "text": "#F2EFEA",
    "text_dim": "#9B96A3",
    "warm": "#FF7A30",   # gain-stage effects: distortion, overdrive, fuzz
    "cool": "#4FD1C5",   # modulation / time effects: chorus, flanger, phaser, delay, reverb
    "neutral": "#E8E4DC",  # clean / no effect
    "danger": "#FF5C5C",
}

# Which family a class belongs to determines its color everywhere in the
# app -- the VU meter, the verdict card, the sidebar signal-chain list.
# Unrecognised class names fall back to "cool" rather than erroring, so a
# retrained model with new class names never breaks the UI.
EFFECT_FAMILY = {
    "clean": "neutral",
    "distortion": "warm", "overdrive": "warm", "fuzz": "warm",
    "chorus": "cool", "flanger": "cool", "phaser": "cool",
    "tremolo": "cool", "vibrato": "cool", "delay": "cool", "reverb": "cool",
}


def family_color(class_name: str) -> str:
    """Look up the theme color for a class, based on which effect family
    it belongs to (gain-stage / modulation-time / clean)."""
    family = EFFECT_FAMILY.get(class_name.lower(), "cool")
    return COLORS[family]


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #141217; --panel: #1D1A21; --panel-border: #2E2A35;
    --text: #F2EFEA; --text-dim: #9B96A3;
    --warm: #FF7A30; --cool: #4FD1C5; --neutral: #E8E4DC; --danger: #FF5C5C;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

.stApp {
    background: radial-gradient(ellipse 80% 45% at 50% -10%, rgba(255,122,48,0.09), transparent), var(--bg);
}

/* Nameplate hero */
.nameplate-mark {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2.15rem;
    color: var(--text); letter-spacing: -0.02em;
}
.nameplate-mark .accent { color: var(--warm); }
.nameplate-tag {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.08em; margin: 2px 0 18px 0;
}

/* Panel section titles, each with an LED */
.panel-title {
    display: flex; align-items: center; gap: 10px;
    font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.12em; color: var(--text-dim);
    margin: 6px 0 10px 0;
}

/* LED indicator */
.led { display: inline-block; border-radius: 50%; background: #3A3640;
       box-shadow: inset 0 1px 1px rgba(0,0,0,0.4); flex-shrink: 0; }
.led-on { background: var(--led-color);
          box-shadow: 0 0 10px var(--led-color), 0 0 3px var(--led-color);
          animation: ledpulse 2.4s ease-in-out infinite; }
@keyframes ledpulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.55; } }

/* VU meter -- the signature element */
.vu-meter { display: flex; flex-direction: column; gap: 11px; padding: 4px 0 2px 0; }
.vu-row { display: flex; align-items: center; gap: 14px; }
.vu-label { width: 92px; flex-shrink: 0; font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-dim); }
.vu-track { display: flex; gap: 2.5px; flex: 1; }
.vu-seg { flex: 1; height: 15px; border-radius: 2px; background: #26232A;
          border: 1px solid rgba(255,255,255,0.02); transition: background 0.25s ease, box-shadow 0.25s ease; }
.vu-seg.lit { background: var(--seg-color); box-shadow: 0 0 5px var(--seg-color); }
.vu-pct { width: 54px; text-align: right; flex-shrink: 0;
          font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: var(--text); }

/* Verdict card */
.verdict { padding: 22px 26px; border-radius: 14px; background: var(--panel);
           border: 1px solid var(--verdict-color); border-left-width: 4px; }
.verdict-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--text-dim);
                  text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 6px; }
.verdict-class { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2rem;
                  color: var(--verdict-color); text-transform: uppercase; letter-spacing: -0.01em; line-height: 1.1; }
.verdict-conf { font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem; color: var(--text-dim); margin-top: 4px; }

/* History rows */
.hist-row { display: flex; align-items: center; gap: 10px; padding: 5px 0;
            font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
.hist-name { color: var(--text); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.hist-class { text-transform: uppercase; font-weight: 500; }
.hist-conf { color: var(--text-dim); font-size: 0.75rem; }

/* Native widget accents */
[data-testid="stFileUploaderDropzone"] { background: var(--panel) !important;
    border: 1.5px dashed var(--panel-border) !important; border-radius: 12px !important; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--warm) !important; }
[data-testid="stSidebar"] { background: #0F0D12 !important; border-right: 1px solid var(--panel-border); }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }
.stButton button { border-radius: 8px !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.8rem !important; }

.rig-footer { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--text-dim);
              text-align: center; padding: 22px 0 4px 0; opacity: 0.6; }
</style>
"""


# --------------------------------------------------------------------------- #
# Small HTML component helpers
# --------------------------------------------------------------------------- #

def led_html(color: str, on: bool = True, size: int = 9) -> str:
    """A small circular LED indicator -- lit and glowing when `on`, dim
    grey otherwise. Used to mark which stage of the signal chain is active."""
    cls = "led led-on" if on else "led"
    style = f"width:{size}px;height:{size}px;" + (f"--led-color:{color};" if on else "")
    return f'<span class="{cls}" style="{style}"></span>'


def panel_title(label: str, color: str = COLORS["warm"], on: bool = True) -> str:
    """Section header used throughout: an LED plus a small-caps label,
    styled like the labeled stage lights on a rack unit."""
    return f'<div class="panel-title">{led_html(color, on)}{label}</div>'


def vu_meter_html(probabilities: dict[str, float], n_segments: int = 24) -> str:
    """Render per-class confidence as a segmented LED-ladder meter -- the
    app's signature visual -- instead of a plain bar chart. Segment count
    lit is proportional to confidence; color is the class's effect family,
    so the meter reads as "how sure, and what kind of sure."
    """
    rows = []
    for class_name, prob in probabilities.items():
        lit = round(prob * n_segments)
        color = family_color(class_name)
        segments = "".join(
            f'<span class="vu-seg{" lit" if i < lit else ""}" style="--seg-color:{color}"></span>'
            for i in range(n_segments)
        )
        rows.append(
            f'<div class="vu-row">'
            f'<span class="vu-label">{class_name}</span>'
            f'<span class="vu-track">{segments}</span>'
            f'<span class="vu-pct">{prob * 100:5.1f}%</span>'
            f'</div>'
        )
    return f'<div class="vu-meter">{"".join(rows)}</div>'


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def plot_waveform(y: np.ndarray, sr: int, color: str) -> plt.Figure:
    """Amplitude-over-time view -- the fastest way to notice silence,
    clipping, or the wrong file before waiting on a prediction."""
    fig, ax = plt.subplots(figsize=(9, 2.1))
    fig.patch.set_facecolor(COLORS["panel"])
    ax.set_facecolor(COLORS["panel"])
    librosa.display.waveshow(y, sr=sr, ax=ax, color=color)
    ax.set_xlabel("Time (s)", color=COLORS["text_dim"], fontsize=9)
    ax.set_ylabel("")
    ax.tick_params(colors=COLORS["text_dim"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(COLORS["panel_border"])
    fig.tight_layout()
    return fig


def plot_spectrogram(y: np.ndarray, sr: int) -> plt.Figure:
    """Mel-spectrogram -- a more direct look at the frequency content the
    MFCC/spectral-centroid features are actually computed from."""
    fig, ax = plt.subplots(figsize=(9, 2.5))
    fig.patch.set_facecolor(COLORS["panel"])
    ax.set_facecolor(COLORS["panel"])
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_xlabel("Time (s)", color=COLORS["text_dim"], fontsize=9)
    ax.set_ylabel("Mel freq", color=COLORS["text_dim"], fontsize=9)
    ax.tick_params(colors=COLORS["text_dim"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(COLORS["panel_border"])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #

@st.cache_resource
def load_predictor() -> TonePredictor:
    """Load the trained model once per app session, not once per upload --
    re-deserializing on every interaction is exactly the kind of thing
    that feels fine in a demo and sluggish the moment someone actually
    uses the app."""
    return TonePredictor(MODEL_DIR)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #

def render_hero() -> None:
    st.markdown(
        '<div class="nameplate-mark">TONE<span class="accent">&nbsp;READER</span></div>'
        '<div class="nameplate-tag">Guitar Tone &amp; FX Pedal Classifier</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Feed it a clip. It reads the signal and tells you what's in the chain.",
    )


def render_sidebar(predictor: TonePredictor) -> None:
    with st.sidebar:
        st.markdown(panel_title("The Rig", COLORS["warm"]), unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.82rem; line-height:1.9;">'
            f'<span style="color:var(--text-dim)">Model</span><br>'
            f'<span style="color:var(--text)">{predictor.model_name.replace("_", " ").upper()}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(panel_title("Signal Chain", COLORS["cool"]), unsafe_allow_html=True)
        for cls in predictor.label_encoder.classes_:
            color = family_color(cls)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;">'
                f'{led_html(color)}'
                f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;'
                f'color:var(--text);text-transform:capitalize;">{cls}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Warm = gain-stage effect. Cool = modulation or time-based effect.")


def render_example_picker() -> None:
    """Buttons for any .wav files bundled in examples/, so a first-time
    visitor can try the app without having their own clip handy. Silently
    does nothing if the folder is empty or missing -- this is optional,
    not a requirement to run the app."""
    if not EXAMPLES_DIR.exists():
        return
    examples = sorted(EXAMPLES_DIR.glob("*.wav"))
    if not examples:
        return

    st.markdown(panel_title("Or Try An Example", COLORS["cool"]), unsafe_allow_html=True)
    cols = st.columns(min(len(examples), 4))
    for col, example_path in zip(cols, examples):
        label = example_path.stem.replace("_", " ").replace("-", " ").title()
        with col:
            if st.button(label, key=f"example_{example_path.stem}", use_container_width=True):
                st.session_state.selected_source = str(example_path)


def render_verdict_and_meter(result: dict) -> None:
    top_class = result["predicted_class"]
    color = family_color(top_class)

    st.markdown(panel_title("Reading", color), unsafe_allow_html=True)
    st.markdown(
        f'<div class="verdict" style="--verdict-color:{color};">'
        f'<div class="verdict-label">Predicted effect</div>'
        f'<div class="verdict-class">{top_class}</div>'
        f'<div class="verdict-conf">{result["confidence"] * 100:.1f}% confidence</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(vu_meter_html(result["all_probabilities"]), unsafe_allow_html=True)


def render_history() -> None:
    history: list[dict] = st.session_state.get("history", [])
    if not history:
        return
    with st.expander(f"Session history ({len(history)})"):
        for entry in history[:10]:
            color = family_color(entry["class"])
            st.markdown(
                f'<div class="hist-row">'
                f'<span class="hist-name">{entry["name"]}</span>'
                f'<span class="hist-class" style="color:{color};">{entry["class"]}</span>'
                f'<span class="hist-conf">{entry["confidence"] * 100:.0f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_model_performance(predictor: TonePredictor) -> None:
    candidates: dict = predictor.metadata.get("all_candidates", {})
    if not candidates:
        return
    with st.expander("Model performance — Random Forest vs SVM"):
        cols = st.columns(len(candidates))
        for col, (name, scores) in zip(cols, candidates.items()):
            is_winner = name == predictor.model_name
            with col:
                st.markdown(
                    panel_title(
                        name.replace("_", " ").upper() + (" — WINNER" if is_winner else ""),
                        COLORS["warm"] if is_winner else "#3A3640",
                        on=is_winner,
                    ),
                    unsafe_allow_html=True,
                )
                st.metric("Accuracy", f"{scores['accuracy'] * 100:.1f}%")
                st.metric("Macro F1", f"{scores['macro_f1'] * 100:.1f}%")


def render_footer() -> None:
    st.markdown(
        '<div class="rig-footer">Random Forest &amp; SVM on hand-built features &mdash; '
        'no deep learning. Built with Librosa + scikit-learn.</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(page_title="Tone Reader — Guitar FX Classifier", page_icon="🎸", layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    render_hero()

    try:
        predictor = load_predictor()
    except FileNotFoundError:
        st.markdown(panel_title("No Model Loaded", COLORS["danger"]), unsafe_allow_html=True)
        st.error("No trained model found in `models/`. Run `train.py` first, then restart this app.")
        st.stop()

    render_sidebar(predictor)

    st.session_state.setdefault("history", [])
    st.session_state.setdefault("selected_source", None)

    st.markdown(panel_title("Input", COLORS["warm"]), unsafe_allow_html=True)
    with st.container(border=True):
        uploaded = st.file_uploader("Drop a .wav clip in the input", type=["wav"], label_visibility="collapsed")
        render_example_picker()

    source_bytes: bytes | None = None
    source_name: str | None = None
    if uploaded is not None:
        source_bytes = uploaded.getvalue()
        source_name = uploaded.name
        st.session_state.selected_source = None  # an explicit upload wins over a prior example pick
    elif st.session_state.selected_source:
        example_path = Path(st.session_state.selected_source)
        if example_path.exists():
            source_bytes = example_path.read_bytes()
            source_name = example_path.name

    if source_bytes is None:
        st.info("Waiting for a clip.")
        render_model_performance(predictor)
        render_footer()
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(source_bytes)
        tmp_path = Path(tmp.name)

    try:
        st.markdown(panel_title("Signal", COLORS["cool"]), unsafe_allow_html=True)
        with st.container(border=True):
            st.audio(source_bytes, format="audio/wav")
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
            st.pyplot(plot_waveform(y, sr, COLORS["cool"]))
            st.pyplot(plot_spectrogram(y, sr))

        with st.status("Reading the signal chain...", expanded=False) as status:
            st.write("Extracting MFCC, spectral centroid, ZCR, and RMS energy...")
            result = predictor.predict(tmp_path)
            status.update(label="Signal read.", state="complete")

        render_verdict_and_meter(result)

        st.session_state.history.insert(
            0, {"name": source_name, "class": result["predicted_class"], "confidence": result["confidence"]}
        )
    except Exception as exc:
        st.error(f"Couldn't process this clip: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    render_history()
    render_model_performance(predictor)
    render_footer()


if __name__ == "__main__":
    main()
