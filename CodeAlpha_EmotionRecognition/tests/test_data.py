import tempfile

import numpy as np

import augment as A
import data as D
import make_demo_data as M


def test_parse_ravdess_and_tess_names():
    r = D.parse_name("Actor_03/03-01-06-01-02-01-03.wav")
    assert r == {"corpus": "ravdess", "speaker": "ravdess_03", "emotion": "fear", "gender": "male"}
    assert D.parse_name("03-01-02-01-01-01-04.wav")["emotion"] == "neutral"          # calm -> neutral
    assert D.parse_name("03-01-02-01-01-01-04.wav")["gender"] == "female"
    assert D.parse_name("03-02-05-01-01-01-04.wav") is None                           # song channel skip
    t = D.parse_name("OAF_back_ps.wav")
    assert t["emotion"] == "surprise" and t["speaker"] == "tess_OAF"
    assert D.parse_name("random.wav") is None


def test_scan_and_speaker_independent_split():
    with tempfile.TemporaryDirectory() as d:
        M.make_dataset(d, n_actors=12, per_emotion=1)
        df = D.scan(d)
        assert set(df.emotion) == set(D.EMOTIONS) and set(df.corpus) == {"ravdess", "tess"}
        sp = D.speaker_split(df)
        rv = sp[sp.corpus == "ravdess"]
        groups = {s: set(rv[rv.split == s].speaker) for s in ("train", "val", "test")}
        assert not (groups["train"] & groups["test"]) and not (groups["train"] & groups["val"]) and not (groups["val"] & groups["test"])
        assert len(groups["test"]) >= 2 and len({g for g in rv[rv.split == "test"].gender}) == 2   # dono genders test mein
        assert rv.speaker_independent.all() and not sp[sp.corpus == "tess"].speaker_independent.any()
        assert (sp.split == "train").sum() > (sp.split == "test").sum()


def test_augmentations_keep_audio_valid():
    rng = np.random.default_rng(0)
    y = (0.3 * np.sin(2 * np.pi * 200 * np.arange(32000) / 16000)).astype(np.float32)
    for op in A.OPS:
        out = op(y, rng)
        assert np.isfinite(out).all() and len(out) > 0
    assert abs(len(A.change_speed(y, np.random.default_rng(1))) - len(y)) < 0.2 * len(y)
    assert len(A.add_noise(y, rng)) == len(y) and len(A.augment(y, rng)) > 0
    snr = 10 * np.log10(np.mean(y ** 2) / np.mean((A.add_noise(y, np.random.default_rng(2), (20, 20)) - y) ** 2))
    assert abs(snr - 20) < 1.5
