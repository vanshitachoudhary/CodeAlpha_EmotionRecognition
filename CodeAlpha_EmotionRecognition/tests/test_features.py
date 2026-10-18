import io
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

import features as F


def tone(f, dur=1.0, sr=F.SR, amp=0.5):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_mel_scale_roundtrip_and_reference():
    assert abs(float(F.hz_to_mel(1000.0)) - 1000.0) < 1.0           # HTK: 1000 Hz ~ 1000 mel
    f = np.array([100.0, 440.0, 4000.0])
    assert np.allclose(F.mel_to_hz(F.hz_to_mel(f)), f)


def test_filterbank_shape_nonneg_and_covers_band():
    fb = F.mel_filterbank()
    assert fb.shape == (F.N_MELS, F.N_FFT // 2 + 1) and (fb >= 0).all()
    assert (fb.sum(axis=0)[2:-2] > 0).all()                          # har frequency kisi filter mein aati hai


def test_mfcc_and_sequence_shapes():
    y = F.fix_length(tone(300, 3.0))
    assert F.mfcc(y).shape == (188, F.N_MFCC)
    seq, stats = F.extract(tone(300, 2.0))
    assert seq.shape == (188, 2 * F.N_MFCC) and stats.shape == (F.STAT_DIM,)
    assert np.isfinite(seq).all() and np.isfinite(stats).all()


def test_delta_of_linear_ramp_is_its_slope():
    ramp = np.arange(40, dtype=float)[:, None] * 0.5
    d = F.deltas(ramp)
    assert np.allclose(d[6:-6], 0.5)                                 # edges chhod ke slope = 0.5


def test_spectrum_peaks_at_tone_frequency():
    p = F.stft_power(tone(1000.0)).mean(axis=0)
    assert abs(np.argmax(p) * F.SR / F.N_FFT - 1000.0) < 40


def test_f0_tracker_on_tones_and_noise():
    for f in (120.0, 200.0, 300.0):
        f0 = F.estimate_f0(tone(f, 1.0))
        v = f0[f0 > 0]
        assert len(v) > 0.8 * len(f0) and abs(np.median(v) - f) < 0.03 * f
    noise = np.random.default_rng(0).normal(0, 0.2, F.SR).astype(np.float32)
    assert (F.estimate_f0(noise) > 0).mean() < 0.3


def test_prosody_reflects_pitch_and_loudness():
    lo, hi = F.prosody_features(tone(120.0)), F.prosody_features(tone(280.0, amp=0.9))
    i = F.PROSODY_NAMES.index
    assert hi[i("f0_mean")] > 2 * lo[i("f0_mean")] and hi[i("rms_mean")] > lo[i("rms_mean")]
    assert len(lo) == len(F.PROSODY_NAMES)


def test_trim_and_fix_length():
    y = np.concatenate([np.zeros(8000, np.float32), tone(300, 0.5), np.zeros(8000, np.float32)])
    t = F.trim_silence(y)
    assert len(t) < 0.7 * len(y) and len(t) >= len(tone(300, 0.5))
    assert len(F.fix_length(tone(300, 1.0))) == F.TARGET_LEN and len(F.fix_length(tone(300, 5.0))) == F.TARGET_LEN


def test_load_wav_resamples_and_accepts_file_objects():
    sr = 48000
    y = (tone(440.0, 1.0, sr) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, y)
    buf.seek(0)
    out, rate = F.load_wav(buf)
    assert rate == F.SR and abs(len(out) - F.SR) <= 2 and out.dtype == np.float32
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.wav"
        wavfile.write(p, 16000, y[:16000])
        assert len(F.load_wav(p)[0]) == 16000
