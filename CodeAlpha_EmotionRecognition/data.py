"""Dataset scanning (RAVDESS + TESS), unified 7-class labels and speaker-independent splits."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOJI = {"angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄", "neutral": "😐", "sad": "😢", "surprise": "😲"}
COLOR = {"angry": "#ef4444", "disgust": "#84cc16", "fear": "#8b5cf6", "happy": "#f59e0b",
         "neutral": "#64748b", "sad": "#3b82f6", "surprise": "#ec4899"}
# RAVDESS emotion codes -> unified label ("calm" neutral mein merge: common practice, README mein documented)
RAVDESS_CODES = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad", "05": "angry",
                 "06": "fear", "07": "disgust", "08": "surprise"}
TESS_CODES = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy",
              "neutral": "neutral", "ps": "surprise", "sad": "sad"}
RAVDESS_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$", re.I)
TESS_RE = re.compile(r"^(OAF|YAF)_.+_(angry|disgust|fear|happy|neutral|ps|sad)\.wav$", re.I)


def parse_name(path):
    """Filename -> dict(corpus, speaker, gender, emotion) ya None (agar kisi corpus ka pattern na ho)."""
    n = Path(path).name
    m = RAVDESS_RE.match(n)
    if m:
        modality, channel, emo, _, _, _, actor = m.groups()
        if channel != "01" or emo not in RAVDESS_CODES:      # sirf speech (song nahi)
            return None
        return {"corpus": "ravdess", "speaker": f"ravdess_{actor}", "emotion": RAVDESS_CODES[emo],
                "gender": "male" if int(actor) % 2 else "female"}
    m = TESS_RE.match(n)
    if m:
        return {"corpus": "tess", "speaker": f"tess_{m.group(1).upper()}", "emotion": TESS_CODES[m.group(2).lower()],
                "gender": "female"}
    return None


def scan(root):
    rows = []
    for p in sorted(Path(root).rglob("*.wav")):
        info = parse_name(p)
        if info:
            rows.append({"path": str(p), **info})
    df = pd.DataFrame(rows)
    if len(df):
        df["label"] = df["emotion"].map({e: i for i, e in enumerate(EMOTIONS)})
    return df


def speaker_split(df, seed=42):
    """Speaker-independent split for RAVDESS (alag actors in train / val / test, gender-balanced).
    TESS mein sirf 2 speakers hain, isliye wahan utterance-level stratified split (speaker-dependent, flagged)."""
    df = df.copy()
    df["split"], df["speaker_independent"] = "train", False
    rng = np.random.default_rng(seed)
    rv = df[df.corpus == "ravdess"]
    if len(rv):
        test_sp, val_sp = [], []
        for g in ("male", "female"):
            sp = sorted(rv[rv.gender == g].speaker.unique())
            rng.shuffle(sp)
            k = min(2, max(1, len(sp) // 5)) if len(sp) >= 3 else 0
            test_sp += sp[:k]
            val_sp += sp[k:2 * k]
        df.loc[df.speaker.isin(test_sp), "split"] = "test"
        df.loc[df.speaker.isin(val_sp), "split"] = "val"
        df.loc[df.corpus == "ravdess", "speaker_independent"] = True
    ts = df[df.corpus == "tess"]
    if len(ts):
        tr, rest = _split(ts.index, ts.label, 0.3, seed)
        va, te = _split(rest, ts.loc[rest].label, 0.5, seed)
        df.loc[va, "split"], df.loc[te, "split"] = "val", "test"
    return df


def _split(idx, labels, test_size, seed):
    """Stratified split; bahut chhote classes pe (sirf tiny test data mein) plain random split pe fallback."""
    try:
        return train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
    except ValueError:
        return train_test_split(idx, test_size=test_size, random_state=seed)
