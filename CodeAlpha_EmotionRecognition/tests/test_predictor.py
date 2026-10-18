import tempfile

import numpy as np

import features as F
import make_demo_data as M
import train as T
from predictor import Predictor


def _trained(d):
    M.make_dataset(f"{d}/data", n_actors=12, per_emotion=2)
    T.main(["--data", f"{d}/data", "--models", "classical", "--aug", "0", "--out", f"{d}/out",
            "--models-dir", f"{d}/models", "--cache", f"{d}/cache", "--no-cache"])
    return Predictor(f"{d}/models")


def test_predictor_probabilities_timeline_and_occlusion():
    with tempfile.TemporaryDirectory() as d:
        p = _trained(d)
        rng = np.random.default_rng(0)
        y_sad, _ = M.synth("sad", False, rng)
        y_angry, _ = M.synth("angry", False, rng)
        pr = p.predict_clip(y_sad)
        assert pr.shape == (7,) and abs(pr.sum() - 1) < 1e-6 and p.classes[int(pr.argmax())] == "sad"
        assert p.classes[int(p.predict_clip(y_angry).argmax())] == "angry"
        # timeline: sad ke baad angry jodo -> lamba clip, emotion badalna chahiye
        long = np.concatenate([y_sad, y_sad, y_angry, y_angry])
        t, P = p.timeline(long)
        assert len(t) == len(P) >= 4 and P.shape[1] == 7 and np.allclose(P.sum(1), 1)
        assert p.classes[int(P[0].argmax())] == "sad" and p.classes[int(P[-1].argmax())] == "angry"
        o = p.occlusion(y_sad)
        assert len(o["times"]) == len(o["drops"]) > 3 and np.isfinite(o["drops"]).all() and 0 <= o["p0"] <= 1


def test_predictor_falls_back_when_best_model_cannot_load():
    with tempfile.TemporaryDirectory() as d:
        _trained(d)
        import json
        from pathlib import Path
        meta = json.loads((Path(d) / "models" / "meta.json").read_text())
        meta["models"]["Fake CNN"] = {"kind": "keras", "file": "missing.keras", "val_f1": 1, "val_acc": 1, "test_f1": 1, "test_acc": 1}
        meta["best"] = "Fake CNN"
        (Path(d) / "models" / "meta.json").write_text(json.dumps(meta))
        p = Predictor(f"{d}/models")
        assert p.kind == "sklearn" and "using" in p.note                               # graceful fallback
