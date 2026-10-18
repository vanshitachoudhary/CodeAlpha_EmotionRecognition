"""Inference wrapper: ek hi interface se classical (sklearn) aur deep (Keras) models; timeline + occlusion explanations."""
import json
from pathlib import Path

import joblib
import numpy as np

import features as F
from ml import full_proba


class Predictor:
    def __init__(self, models_dir="models", prefer=None):
        self.dir = Path(models_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.classes = self.meta["classes"]
        self.note = ""
        want = prefer or self.meta["best"]
        try:
            self._load(want)
        except Exception as e:                      # jaise TensorFlow install na ho -> best classical model
            fb = self.meta["best_classical"]
            self._load(fb)
            self.note = f"{want} could not be loaded ({type(e).__name__}); using {fb} instead."

    def _load(self, name):
        info = self.meta["models"][name]
        self.name, self.kind = name, info["kind"]
        if self.kind == "sklearn":
            self.model = joblib.load(self.dir / info["file"])
        else:
            from tensorflow import keras
            self.model = keras.models.load_model(self.dir / info["file"])
            z = np.load(self.dir / "norm.npz")
            self.mu, self.sd = z["mu"], z["sd"]

    # ------------------------------------------------------------------ core
    def proba(self, waves, trim=True):
        """list of waveforms -> (N, n_classes) probabilities."""
        feats = [F.extract(w, trim=trim) for w in waves]
        if self.kind == "sklearn":
            return full_proba(self.model, np.stack([s for _, s in feats]), len(self.classes))
        X = (np.stack([q for q, _ in feats]) - self.mu) / self.sd
        return np.asarray(self.model.predict(X, verbose=0))

    def predict_clip(self, y):
        return self.proba([y])[0]

    # ------------------------------------------------------------------ explanations
    def timeline(self, y, win=2.0, hop=0.5):
        """Sliding-window predictions: emotion kaise badalta hai. Returns (centre_times_s, probs (W, K))."""
        y = F.peak_normalize(F.trim_silence(np.asarray(y, np.float32)))
        n, w, h = len(y), int(win * F.SR), int(hop * F.SR)
        starts = [0] if n <= w else list(range(0, n - w + 1, h))
        if n > w and starts[-1] != n - w:
            starts.append(n - w)
        segs = [y[s:s + w] for s in starts]
        return np.array([(s + len(g) / 2) / F.SR for s, g in zip(starts, segs)]), self.proba(segs, trim=False)

    def occlusion(self, y, win=0.5, hop=0.25):
        """Model-agnostic explanation: clip ke ek-ek hisse ko silence karke dekho predicted emotion ki probability kitni giri.
        Returns dict(times, drops, cls, p0, wave)."""
        y = F.peak_normalize(F.trim_silence(np.asarray(y, np.float32)))
        if len(y) > F.TARGET_LEN:
            s = (len(y) - F.TARGET_LEN) // 2
            y = y[s:s + F.TARGET_LEN]
        w, h = int(win * F.SR), int(hop * F.SR)
        starts = [0] if len(y) <= w else list(range(0, len(y) - w + 1, h))
        variants = [y]
        for s in starts:
            o = y.copy()
            o[s:s + w] = 0.0
            variants.append(o)
        P = self.proba(variants, trim=False)
        c = int(P[0].argmax())
        return {"times": np.array([(s + w / 2) / F.SR for s in starts]), "drops": P[0, c] - P[1:, c],
                "cls": c, "p0": float(P[0, c]), "wave": y}
