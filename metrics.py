"""Ten reconstruction metrics, evaluated exclusively on genuinely hidden channels."""
import numpy as np
from scipy.signal import welch
NAMES = ['RMSE', 'MAE', 'NMSE', 'R2', 'PCC', 'SNR_dB', 'SpectralConvergence', 'LSD_dB', 'PSD_PCC', 'BandPowerRelativeError']
DIRECTIONS = {'RMSE': 'min', 'MAE': 'min', 'NMSE': 'min', 'R2': 'max', 'PCC': 'max', 'SNR_dB': 'max', 'SpectralConvergence': 'min', 'LSD_dB': 'min', 'PSD_PCC': 'max', 'BandPowerRelativeError': 'min'}

def correlation(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1)).clip(1e-12)

def per_sample(pred, target, hidden):
    pred = np.asarray(pred, dtype='float64')
    target = np.asarray(target, dtype='float64')
    if pred.shape != target.shape or hidden.shape != target.shape[:2]:
        raise ValueError('Metric shape mismatch')
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite reconstruction')
    e = pred - target
    sse = (e * e).sum(-1)
    energy = (target * target).sum(-1).clip(1e-12)
    values = {'RMSE': np.sqrt((e * e).mean(-1)), 'MAE': abs(e).mean(-1), 'NMSE': sse / energy, 'R2': 1 - sse / ((target - target.mean(-1, keepdims=True)) ** 2).sum(-1).clip(1e-12), 'PCC': correlation(pred, target), 'SNR_dB': 10 * np.log10(energy / sse.clip(1e-12))}
    ft = np.abs(np.fft.rfft(target, axis=-1))
    fp = np.abs(np.fft.rfft(pred, axis=-1))
    values['SpectralConvergence'] = np.linalg.norm(fp - ft, axis=-1) / np.linalg.norm(ft, axis=-1).clip(1e-12)
    f, pt = welch(target, fs=128, nperseg=128, axis=-1)
    _, pp = welch(pred, fs=128, nperseg=128, axis=-1)
    use = (f >= 1) & (f <= 45)
    floor = np.maximum(pt.max(-1, keepdims=True) * 1e-12, 1e-20)
    db = 10 * np.log10(np.maximum(pp, floor) / np.maximum(pt, floor))
    values['LSD_dB'] = np.sqrt((db[..., use] ** 2).mean(-1))
    values['PSD_PCC'] = correlation(pp[..., use], pt[..., use])
    bands = []
    for lo, hi in [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]:
        band = (f >= lo) & (f < hi)
        true = pt[..., band].sum(-1)
        bands.append(abs(pp[..., band].sum(-1) - true) / true.clip(1e-12))
    values['BandPowerRelativeError'] = np.stack(bands, -1).mean(-1)
    h = np.asarray(hidden, dtype='float64')
    den = h.sum(1)
    if (den <= 0).any():
        raise ValueError('No held-out channel')
    return np.stack([(values[n] * h).sum(1) / den for n in NAMES], axis=1)

def aggregate(values, subjects):
    subject_means = np.stack([values[subjects == s].mean(0) for s in np.unique(subjects)])
    return {name: float(value) for name, value in zip(NAMES, subject_means.mean(0))}

def self_test():
    rng = np.random.default_rng(1)
    target = rng.normal(size=(3, 4, 256))
    hidden = np.tile([True, False, True, False], (3, 1))
    perfect = per_sample(target, target, hidden)
    assert np.allclose(perfect[:, [0, 1, 2, 6, 7, 9]], 0, atol=1e-07)
    assert np.allclose(perfect[:, [3, 4, 8]], 1, atol=1e-07)
    changed = target.copy()
    changed[:, ~hidden[0], :] = rng.normal(size=(3, 2, 256)) * 10000.0
    assert np.allclose(per_sample(changed, target, hidden), perfect)
    bad = per_sample(np.zeros_like(target), target, hidden)
    assert np.all(bad[:, 2] > 0.99) and np.all(bad[:, 4] == 0)
    print('Metrics verified: perfect reconstruction, missing-only scoring, zero predictor.')
if __name__ == '__main__':
    self_test()
