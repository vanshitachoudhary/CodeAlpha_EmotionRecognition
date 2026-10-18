"""SYNTHETIC "speech" generator, sirf pipeline test/smoke-test ke liye. Ye asli speech nahi hai:
isse train kiye model ke results ka koi matlab nahi. Real run ke liye RAVDESS/TESS download karo (download_data.py)."""
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# (f0 mean, f0 vibrato depth, vibrato Hz, loudness, harmonic tilt (higher = brighter), syllable rate Hz)
PARAMS = {"neutral": (150, 0.02, 3, 0.5, 1.2, 3.0), "sad": (115, 0.03, 2, 0.3, 2.2, 2.0), "happy": (240, 0.10, 6, 0.7, 0.9, 4.5),
          "angry": (210, 0.08, 9, 0.95, 0.5, 5.0), "fear": (280, 0.12, 11, 0.5, 1.0, 5.5),
          "disgust": (135, 0.05, 4, 0.55, 1.6, 2.5), "surprise": (320, 0.14, 5, 0.75, 0.8, 3.5)}
CODE = {"neutral": "01", "happy": "03", "sad": "04", "angry": "05", "fear": "06", "disgust": "07", "surprise": "08"}


def synth(emotion, female, rng, sr=16000, dur=2.5):
    f0m, vib, vhz, loud, tilt, syl = PARAMS[emotion]
    f0m *= (1.6 if female else 1.0) * rng.uniform(0.92, 1.08)
    t = np.arange(int(sr * dur)) / sr
    f0 = f0m * (1 + vib * np.sin(2 * np.pi * vhz * t + rng.uniform(0, 6)))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    y = sum(np.sin(h * phase) / h ** tilt for h in range(1, 14))
    env = 0.5 * (1 + np.sin(2 * np.pi * syl * t + rng.uniform(0, 6))) ** 0.7
    y = y * env * loud + rng.normal(0, 0.01, len(t))
    y = np.concatenate([np.zeros(int(0.2 * sr)), y, np.zeros(int(0.2 * sr))])
    return (y / np.abs(y).max() * 0.8 * (0.5 + 0.5 * loud)).astype(np.float32), sr


def make_dataset(root, n_actors=12, per_emotion=2, seed=0, with_tess=True):
    rng = np.random.default_rng(seed)
    root = Path(root)
    for a in range(1, n_actors + 1):
        d = root / "ravdess" / f"Actor_{a:02d}"
        d.mkdir(parents=True, exist_ok=True)
        for emo, code in CODE.items():
            for r in range(per_emotion):
                y, sr = synth(emo, female=(a % 2 == 0), rng=rng)
                wavfile.write(d / f"03-01-{code}-01-0{r + 1}-01-{a:02d}.wav", sr, (y * 32767).astype(np.int16))
    if with_tess:
        d = root / "tess"
        for spk in ("OAF", "YAF"):
            (d / f"{spk}_demo").mkdir(parents=True, exist_ok=True)
            for emo in PARAMS:
                for r in range(per_emotion * 2):
                    y, sr = synth(emo, female=True, rng=rng)
                    tag = "ps" if emo == "surprise" else emo
                    wavfile.write(d / f"{spk}_demo" / f"{spk}_word{r}_{tag}.wav", sr, (y * 32767).astype(np.int16))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "demo_data"
    make_dataset(out)
    print(f"Synthetic demo data written to {out}/  (NOT real speech)")
