"""Audio front-end written from scratch with numpy/scipy (no librosa):
WAV loading, silence trimming, log-mel spectrogram, MFCC + deltas, F0 tracking and prosody features."""
from functools import lru_cache
from math import gcd

import numpy as np
from scipy.fft import dct, irfft, rfft
from scipy.io import wavfile
from scipy.signal import resample_poly

SR, N_FFT, HOP, N_MELS, N_MFCC = 16000, 512, 256, 64, 40
DURATION = 3.0
TARGET_LEN = int(SR * DURATION)
PROSODY_NAMES = ["f0_mean", "f0_std", "f0_range", "f0_min", "f0_max", "voiced_ratio", "rms_mean", "rms_std",
                 "rms_max", "zcr_mean", "zcr_std"]      # duration jaan-bujh ke nahi: wo corpus ka shortcut (TESS ~2s, RAVDESS ~3.5s) ban jata hai
swv = np.lib.stride_tricks.sliding_window_view


# ------------------------------------------------------------------ I/O + preprocessing
def load_wav(path_or_file, sr=SR):
    """16-bit/32-bit/float WAV (path ya file-like, jaise Streamlit upload) -> mono float32 at `sr`."""
    rate, y = wavfile.read(path_or_file)
    if y.dtype.kind == "i":
        y = y.astype(np.float32) / np.iinfo(y.dtype).max
    elif y.dtype.kind == "u":
        y = (y.astype(np.float32) - 128.0) / 128.0
    else:
        y = y.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if rate != sr:
        g = gcd(int(rate), int(sr))
        y = resample_poly(y, int(sr) // g, int(rate) // g).astype(np.float32)
    return y, sr


def trim_silence(y, top_db=35.0, frame=512, hop=128):
    """Shuru aur aakhir ki khamoshi hatao (peak se `top_db` dB neeche ka sab silence)."""
    if len(y) < frame:
        return y
    rms = np.sqrt((swv(y, frame)[::hop] ** 2).mean(axis=1))
    if rms.max() <= 1e-8:
        return y
    keep = np.where(rms > rms.max() * 10 ** (-top_db / 20))[0]
    return y[max(keep[0] * hop - hop, 0): min(keep[-1] * hop + frame + hop, len(y))]


def peak_normalize(y, peak=0.95):
    m = np.abs(y).max()
    return y if m < 1e-8 else (y * (peak / m)).astype(np.float32)


def fix_length(y, n=TARGET_LEN):
    """Centre-crop ya zero-pad to exactly n samples (model ko fixed input chahiye)."""
    if len(y) >= n:
        s = (len(y) - n) // 2
        return y[s:s + n]
    return np.pad(y, (0, n - len(y)))


# ------------------------------------------------------------------ mel / MFCC
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, float) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m, float) / 2595.0) - 1.0)


@lru_cache(maxsize=4)
def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=None):
    """Triangular mel filters (area-normalised), shape (n_mels, n_fft // 2 + 1)."""
    fmax = fmax or sr / 2
    hz = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        lo, ce, hi = hz[m], hz[m + 1], hz[m + 2]
        fb[m] = np.maximum(0.0, np.minimum((freqs - lo) / (ce - lo), (hi - freqs) / (hi - ce))) * 2.0 / (hi - lo)
    fb.setflags(write=False)
    return fb


def stft_power(y, n_fft=N_FFT, hop=HOP):
    pad = n_fft // 2
    yp = np.pad(y, (pad, pad), mode="reflect" if len(y) > pad else "constant")
    win = np.hanning(n_fft + 1)[:-1]
    return np.abs(rfft(swv(yp, n_fft)[::hop] * win, axis=1)) ** 2


def log_mel(y, top_db=80.0):
    p = stft_power(y) @ mel_filterbank().T
    db = 10.0 * np.log10(np.maximum(p, 1e-10))
    return np.maximum(db, db.max() - top_db)


def mfcc(y, n_mfcc=N_MFCC):
    return dct(log_mel(y), type=2, axis=1, norm="ortho")[:, :n_mfcc]


def deltas(x, width=9):
    """Regression-based delta (time derivative) of a (T, F) feature matrix."""
    n = width // 2
    xp = np.pad(x, ((n, n), (0, 0)), mode="edge")
    num = sum(k * (xp[n + k: n + k + len(x)] - xp[n - k: n - k + len(x)]) for k in range(1, n + 1))
    return num / (2 * sum(k * k for k in range(1, n + 1)))


def sequence_features(y_fixed):
    """(T, 80): 40 MFCC + 40 delta-MFCC. Deep models (CNN / LSTM) yahi dekhte hain."""
    m = mfcc(y_fixed)
    return np.hstack([m, deltas(m)]).astype(np.float32)


# ------------------------------------------------------------------ pitch + prosody
def estimate_f0(y, sr=SR, fmin=70.0, fmax=400.0, frame=1024, hop=HOP, thresh=0.35):
    """Autocorrelation pitch tracker. Returns per-frame F0 in Hz (0 = unvoiced)."""
    if len(y) < frame:
        return np.zeros(0)
    fr = swv(y, frame)[::hop] * np.hanning(frame)
    fr = fr - fr.mean(axis=1, keepdims=True)
    ac = irfft(np.abs(rfft(fr, n=2 * frame, axis=1)) ** 2, axis=1)[:, :frame]
    lo, hi = int(sr / fmax), min(int(sr / fmin), frame - 1)
    seg = ac[:, lo:hi + 1] / (ac[:, :1] + 1e-9)
    lag = seg.argmax(axis=1) + lo
    rms = np.sqrt((fr ** 2).mean(axis=1))
    voiced = (seg.max(axis=1) > thresh) & (rms > 0.03 * (rms.max() + 1e-12))
    return np.where(voiced, sr / lag, 0.0)


def prosody_features(y):
    """11 interpretable prosody numbers: pitch stats, voicing, loudness stats, zero-crossings."""
    f0 = estimate_f0(y)
    v = f0[f0 > 0]
    if len(v) >= 3:
        f = [v.mean(), v.std(), np.percentile(v, 95) - np.percentile(v, 5), v.min(), v.max()]
    else:
        f = [0.0] * 5
    w = swv(y, 512)[::HOP] if len(y) >= 512 else np.pad(y, (0, 512 - len(y)))[None, :]
    rms = np.sqrt((w ** 2).mean(axis=1))
    zcr = (np.abs(np.diff(np.sign(w), axis=1)) > 0).mean(axis=1)
    vr = len(v) / max(len(f0), 1)
    return np.array(f + [vr, rms.mean(), rms.std(), rms.max(), zcr.mean(), zcr.std()], np.float32)


def stat_features(seq):
    return np.concatenate([seq.mean(0), seq.std(0), seq.min(0), seq.max(0)]).astype(np.float32)


STAT_DIM = 4 * 2 * N_MFCC + len(PROSODY_NAMES)


def extract(y, trim=True):
    """waveform -> (sequence (188, 80) for deep models, statistics (332,) for classical models)."""
    y = np.asarray(y, np.float32)
    if trim:
        y = trim_silence(y)
    y = peak_normalize(y)
    seq = sequence_features(fix_length(y))
    return seq, np.concatenate([stat_features(seq), prosody_features(y)])
