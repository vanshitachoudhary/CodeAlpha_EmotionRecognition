"""Speech Emotion AI - explainable emotion recognition from voice.   Run: streamlit run app.py"""
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import data as D
import features as F
import ui
from predictor import Predictor

st.set_page_config(page_title="Speech Emotion AI", page_icon="🎙️", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)
OUT = Path("outputs")


@st.cache_resource
def get_predictor(prefer):
    return Predictor("models", prefer)


if not Path("models/meta.json").exists():
    st.markdown(ui.hero("Speech Emotion AI", "Model not trained yet.", ["setup needed"]), unsafe_allow_html=True)
    st.error("No trained model found in `models/`. Run `python download_data.py` and then `python train.py` first.")
    st.stop()

meta = json.loads(Path("models/meta.json").read_text(encoding="utf-8"))
with st.sidebar:
    st.markdown("### Settings")
    prefer = st.selectbox("Model", list(meta["models"]), index=list(meta["models"]).index(meta["best"]),
                          help="Best model is chosen on validation speakers.")
    low_conf = st.slider("Low-confidence warning below", 0.30, 0.80, 0.50, 0.05)
    st.warning("Educational demo. Not for hiring, surveillance or decisions about people.")
    st.caption("CodeAlpha ML Internship · Task 2 · RAVDESS + TESS")
pred = get_predictor(prefer)
CL = pred.classes

ci = meta["ci"]
st.markdown(ui.hero("Speech Emotion AI", "Record or upload a voice clip: emotion, confidence, how it changes over time, and which parts of the "
                    "audio drove the decision.", [pred.name, "7 emotions", "speaker-independent evaluation", "occlusion explanations"]),
            unsafe_allow_html=True)
st.markdown(ui.kpi_row([("Test accuracy", f"{ci['accuracy'][0]:.1%}", f"95% CI {ci['accuracy'][1]:.1%} - {ci['accuracy'][2]:.1%}"),
                        ("Macro-F1", f"{ci['macro_f1'][0]:.3f}", f"95% CI {ci['macro_f1'][1]:.3f} - {ci['macro_f1'][2]:.3f}"),
                        ("Test clips", str(meta["counts"]["test"]), "unseen speakers"),
                        ("Trained on", str(meta["counts"]["train"]), "clips (incl. augmented)")]), unsafe_allow_html=True)
if pred.note:
    st.info(pred.note)

t1, t2, t3, t4, t5 = st.tabs(["🎙️ Analyse speech", "📂 Batch", "📊 Model performance", "🧠 How it works", "📋 Model card"])


def read_audio(upload):
    y, _ = F.load_wav(io.BytesIO(upload.getvalue()))
    return y


def explain_figure(o):
    """Waveform + spectrogram; red shading = parts whose removal lowers the prediction most."""
    w, d = o["wave"], o["drops"]
    dur = len(w) / F.SR
    fig, ax = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.5]})
    ax[0].plot(np.arange(len(w)) / F.SR, w, color="#6d28d9", lw=.6)
    mx = max(float(d.max()), 1e-6)
    for t, v in zip(o["times"], d):
        if v > 0:
            ax[0].axvspan(max(t - .25, 0), min(t + .25, dur), color="#ef4444", alpha=min(v / mx, 1) * .55, lw=0)
    ax[0].set_ylabel("amplitude"); ax[0].set_title("Red = silencing this part lowers the prediction most", fontsize=10)
    ax[1].imshow(F.log_mel(w).T, origin="lower", aspect="auto", cmap="magma", extent=[0, dur, 0, F.SR / 2000])
    ax[1].set_ylabel("kHz"); ax[1].set_xlabel("time (s)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


with t1:
    c_up, c_rec = st.columns(2)
    up = c_up.file_uploader("Upload a WAV file", type=["wav"])
    rec = c_rec.audio_input("…or record your voice") if hasattr(st, "audio_input") else None
    src = rec or up
    if src is None:
        st.markdown(ui.panel("Try it", "<p>Upload a <b>.wav</b> clip (acted speech works best, e.g. from RAVDESS/TESS) or record yourself. "
                             "Speak for 2-4 seconds in a clear, expressive voice.</p>"
                             '<div class="note">Tip: the model was trained on acted, studio-quality English speech, so natural, noisy or '
                             "non-English audio will be less reliable.</div>"), unsafe_allow_html=True)
    else:
        try:
            y = read_audio(src)
        except Exception as ex:
            st.error(f"Could not read this file as WAV ({type(ex).__name__}). Use 16-bit PCM WAV."); st.stop()
        st.audio(src)
        if len(y) / F.SR < 1.0:
            st.warning("Very short clip (under 1 s): predictions will be unreliable.")
        P = pred.predict_clip(y)
        order = np.argsort(P)[::-1]
        top, second = int(order[0]), int(order[1])
        times, TP = pred.timeline(y)
        o = pred.occlusion(y)
        yp = F.peak_normalize(F.trim_silence(y))
        pf = dict(zip(F.PROSODY_NAMES, F.prosody_features(yp)))
        left, right = st.columns([1, 1.5])
        with left:
            st.markdown(ui.panel("Detected emotion", ui.emotion_card(CL[top], float(P[top]), (CL[second], float(P[second])),
                                                                     low=P[top] < low_conf) + ui.prob_bars(P, CL)), unsafe_allow_html=True)
        with right:
            st.markdown("**What the model listens to**")
            st.pyplot(explain_figure(o))
            k = st.columns(3)
            k[0].metric("Mean pitch", f"{pf['f0_mean']:.0f} Hz" if pf["f0_mean"] > 0 else "n/a")
            k[1].metric("Voiced speech", f"{pf['voiced_ratio']:.0%}")
            k[2].metric("Loudness (RMS)", f"{pf['rms_mean']:.3f}")
        if len(times) > 1:
            labels = [CL[int(i)] for i in TP.argmax(1)]
            st.markdown(ui.panel("Emotion over time", ui.timeline_strip(times, labels),
                                 "Sliding 2-second windows: how the predicted emotion changes through the clip."), unsafe_allow_html=True)
            st.line_chart(pd.DataFrame(TP, index=np.round(times, 2), columns=CL), color=[D.COLOR[c] for c in CL])
        tl_df = pd.DataFrame(TP, columns=CL)
        tl_df.insert(0, "time_s", np.round(times, 2))
        st.download_button("⬇️ Download timeline (CSV)", tl_df.to_csv(index=False), "emotion_timeline.csv", "text/csv")

with t2:
    files = st.file_uploader("Upload several WAV files", type=["wav"], accept_multiple_files=True)
    if files:
        rows = []
        for f in files:
            try:
                p = pred.predict_clip(read_audio(f))
                rows.append({"file": f.name, "emotion": CL[int(p.argmax())], "confidence": round(float(p.max()), 3)})
            except Exception as ex:
                rows.append({"file": f.name, "emotion": f"error: {type(ex).__name__}", "confidence": None})
        res = pd.DataFrame(rows)
        ok = res[res.confidence.notna()]
        if len(ok):
            st.markdown(ui.kpi_row([("Files", str(len(res)), "analysed"), ("Most common", ok.emotion.mode()[0], "emotion"),
                                    ("Mean confidence", f"{ok.confidence.mean():.0%}", "across files"),
                                    ("Low confidence", str(int((ok.confidence < low_conf).sum())), "files")]), unsafe_allow_html=True)
            st.bar_chart(ok.emotion.value_counts().reindex(CL, fill_value=0), color="#7c3aed")
        st.dataframe(res)
        st.download_button("⬇️ Download results", res.to_csv(index=False), "emotions.csv", "text/csv")

with t3:
    st.markdown("#### Model comparison")
    mt = pd.DataFrame(meta["models"]).T[["val_acc", "val_f1", "test_acc", "test_f1"]].astype(float).round(3)
    mt.columns = ["Val acc", "Val macro-F1", "Test acc", "Test macro-F1"]
    st.dataframe(mt)
    st.caption("Models are selected on validation speakers; test speakers are used once for reporting.")
    if meta.get("leaky"):
        lk = meta["leaky"]
        st.markdown(ui.kpi_row([("Random split (leaky)", f"{lk['random_split_accuracy']:.0%}", "same actors in train and test"),
                                ("Speaker-independent", f"{lk['speaker_independent_accuracy']:.0%}", "unseen actors (honest)"),
                                ("Inflation", f"{(lk['random_split_accuracy'] - lk['speaker_independent_accuracy']) * 100:+.0f} pts", "what leakage hides"),
                                ("Model", lk["model"], "classical baseline")]), unsafe_allow_html=True)
    if meta.get("subgroups"):
        st.markdown("**Subgroups (test split)**")
        st.dataframe(pd.DataFrame(meta["subgroups"]))
    if meta.get("cross"):
        st.markdown("**Cross-corpus generalisation**")
        st.dataframe(pd.DataFrame(meta["cross"]).T.round(3))
    gallery = [("confusion_matrix", "Confusion matrix"), ("per_class_f1", "Per-emotion F1"), ("model_comparison", "Model comparison"),
               ("leaky_vs_honest", "Leaky vs honest evaluation"), ("training_curves", "Training curves"),
               ("emotion_spectrograms", "Spectrogram per emotion"), ("prosody_by_emotion", "Pitch / loudness per emotion")]
    present = [(f, c) for f, c in gallery if (OUT / f"{f}.png").exists()]
    cols = st.columns(2)
    for i, (f, c) in enumerate(present):
        cols[i % 2].image(str(OUT / f"{f}.png"), caption=c)

with t4:
    st.markdown(ui.panel("From voice to emotion", ui.steps([
        ("Record or upload", "WAV audio, resampled to 16 kHz mono. Leading and trailing silence is trimmed and the level normalised."),
        ("Fix the length", "Clips are centre-cropped or zero-padded to 3 seconds so every input has the same shape."),
        ("Extract features (written from scratch)", "64-band log-mel spectrogram -> 40 MFCCs + deltas (sequence for deep models), "
                                                    "plus statistics and 11 prosody numbers: pitch (autocorrelation F0 tracker), loudness, voicing, zero-crossings."),
        ("Classify", "Classical baselines (Logistic Regression, SVM, Random Forest) and, if TensorFlow is installed, a 1D-CNN, a BiLSTM and a CNN-BiLSTM."),
        ("Explain", "Sliding windows show how emotion changes over time; occlusion (silencing each 0.5 s) shows which parts of the audio mattered.")])),
        unsafe_allow_html=True)
    st.markdown('<div class="tip">💡 <b>Why speaker-independent evaluation?</b> If the same actor appears in training and test, models can '
                "memorise voices instead of emotions and look far better than they are. Here, test actors are never seen in training.</div>",
                unsafe_allow_html=True)

with t5:
    p = OUT / "MODEL_CARD.md"
    st.markdown(p.read_text(encoding="utf-8") if p.exists() else "Model card not found. Run `python train.py`.")
    r = OUT / "RESULTS.md"
    if r.exists():
        with st.expander("Full results tables"):
            st.markdown(r.read_text(encoding="utf-8"))

st.markdown(ui.footer(), unsafe_allow_html=True)
