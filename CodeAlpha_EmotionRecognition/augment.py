"""Waveform augmentation (numpy/scipy). Training set ko badhane aur noise/speed ke prati robust banane ke liye."""
import numpy as np
from scipy.signal import resample_poly


def add_noise(y, rng, snr_db=(15.0, 35.0)):
    snr = rng.uniform(*snr_db)
    p = np.mean(y ** 2) + 1e-12
    return (y + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), len(y))).astype(np.float32)


def time_shift(y, rng, max_frac=0.15):
    s = int(rng.uniform(-max_frac, max_frac) * len(y))
    out = np.roll(y, s)
    if s > 0:
        out[:s] = 0
    elif s < 0:
        out[s:] = 0
    return out


def change_speed(y, rng, rates=(0.9, 1.1)):
    r = rng.uniform(*rates)
    return resample_poly(y, int(round(100 / r)), 100).astype(np.float32)


def change_gain(y, rng, db=(-6.0, 6.0)):
    return (y * 10 ** (rng.uniform(*db) / 20)).astype(np.float32)


OPS = [add_noise, time_shift, change_speed, change_gain]


def augment(y, rng, n_ops=2):
    for i in rng.choice(len(OPS), size=n_ops, replace=False):
        y = OPS[i](y, rng)
    return y
