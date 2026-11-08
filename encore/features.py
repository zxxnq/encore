"""Voice features: MFCCs implemented with numpy/scipy only (no librosa needed).

Per-utterance mean/variance normalisation removes loudness and microphone
differences, which matters because a fatigued voice is quieter, not just slower.
"""

import numpy as np
from scipy.fft import dct

_FB_CACHE = {}


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float = 60.0, fmax=None):
    key = (n_mels, n_fft, sr, fmin, fmax)
    if key in _FB_CACHE:
        return _FB_CACHE[key]
    fmax = fmax or sr / 2
    hz = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins))
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        center = max(center, left + 1)
        right = min(max(right, center + 1), n_bins)
        fb[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        fb[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    _FB_CACHE[key] = fb
    return fb


def mfcc(signal, sr: int = 16000, n_mfcc: int = 13, frame_ms: float = 25.0,
         hop_ms: float = 15.0, n_fft: int = 512, n_mels: int = 26,
         preemph: float = 0.97, deltas: bool = True, cmvn: bool = True) -> np.ndarray:
    """Returns a (frames, n_mfcc * 2) float32 array (MFCC + deltas)."""
    x = np.asarray(signal, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("empty signal")
    x = np.append(x[0], x[1:] - preemph * x[:-1])
    flen = int(round(sr * frame_ms / 1000))
    hop = int(round(sr * hop_ms / 1000))
    if x.size < flen:
        x = np.pad(x, (0, flen - x.size))
    n_frames = 1 + (x.size - flen) // hop
    idx = np.arange(flen)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hamming(flen)
    power = np.abs(np.fft.rfft(frames, n_fft)) ** 2 / n_fft
    logmel = np.log(power @ mel_filterbank(n_mels, n_fft, sr).T + 1e-10)
    feats = dct(logmel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    if deltas:
        d = np.gradient(feats, axis=0) if n_frames > 1 else np.zeros_like(feats)
        feats = np.hstack([feats, d])
    if cmvn:
        feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)
    return feats.astype(np.float32)


def voice_features(signal, sr: int = 16000) -> np.ndarray:
    """The single entry point the rest of Encore uses for voice cues."""
    return mfcc(signal, sr=sr)
