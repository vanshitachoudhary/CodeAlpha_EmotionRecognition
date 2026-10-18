import tempfile

import make_demo_data as M
import train as T


def test_end_to_end_classical_pipeline_on_synthetic_data():
    """Sirf pipeline check (synthetic audio): files bante hain, speaker-independent split chalta hai, accuracy chance se bahut upar."""
    with tempfile.TemporaryDirectory() as d:
        M.make_dataset(f"{d}/data", n_actors=12, per_emotion=2)
        meta = T.main(["--data", f"{d}/data", "--models", "classical", "--aug", "0", "--out", f"{d}/out",
                       "--models-dir", f"{d}/models", "--cache", f"{d}/cache", "--no-cache", "--cross"])
        assert meta["best"] in meta["models"] and meta["counts"]["test"] > 10
        assert meta["models"][meta["best"]]["test_acc"] > 0.5                         # chance = 1/7 = 0.14
        for f in ("RESULTS.md", "MODEL_CARD.md", "confusion_matrix.png", "emotion_spectrograms.png", "prosody_by_emotion.png"):
            assert (__import__("pathlib").Path(d) / "out" / f).exists(), f
        assert meta["leaky"] is not None and set(meta["cross"]) == {"ravdess -> tess", "tess -> ravdess"}
