"""Audio input/output. Live audio needs the `sounddevice` package (PortAudio).
Everything else (wav files, trimming, quality checks) works without it.
"""

import queue
import threading
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def _sd():
    try:
        import sounddevice as sd
        return sd
    except (ImportError, OSError) as e:
        raise RuntimeError(
            "Live audio needs the 'sounddevice' package and PortAudio.\n"
            "  pip install sounddevice   (macOS: brew install portaudio; "
            "Linux: sudo apt install libportaudio2)\n"
            f"Original error: {e}") from e


def read_wav(path, target_sr: int = 16000) -> np.ndarray:
    sr, data = wavfile.read(str(path))
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    elif data.dtype.kind == "u":
        data = (data.astype(np.float32) - 128.0) / 128.0
    data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    return data


def write_wav(path, signal: np.ndarray, sr: int = 16000):
    x = np.clip(np.asarray(signal, dtype=np.float32), -1.0, 1.0)
    wavfile.write(str(path), sr, (x * 32767).astype(np.int16))


def trim_silence(signal: np.ndarray, sr: int = 16000, rel_db: float = -35.0,
                 pad_ms: int = 80) -> np.ndarray:
    """Cut leading/trailing quiet parts (relative to the loudest 20 ms block)."""
    x = np.asarray(signal, dtype=np.float32).ravel()
    block = int(sr * 0.02)
    if x.size < block * 2:
        return x
    n = x.size // block
    db = 10 * np.log10(np.mean(x[: n * block].reshape(n, block) ** 2, axis=1) + 1e-12)
    loud = np.where(db > db.max() + rel_db)[0]
    if loud.size == 0:
        return x
    pad = int(sr * pad_ms / 1000)
    return x[max(0, loud[0] * block - pad): min(x.size, (loud[-1] + 1) * block + pad)]


def quality_check(signal: np.ndarray, sr: int = 16000) -> dict:
    """Quick checks so a bad take is caught while the user can still redo it."""
    x = np.asarray(signal, dtype=np.float32).ravel()
    block = int(sr * 0.02)
    n = max(1, x.size // block)
    db = np.sort(10 * np.log10(np.mean(x[: n * block].reshape(n, -1) ** 2, axis=1) + 1e-12))
    k = max(1, n // 5)
    snr = float(db[-k:].mean() - db[:k].mean())
    clipped = float(np.mean(np.abs(x) > 0.98))
    peak_db = float(20 * np.log10(np.max(np.abs(x)) + 1e-12))
    warnings = []
    if clipped > 0.001:
        warnings.append("clipping: move the mic back or speak a little softer")
    if peak_db < -30:
        warnings.append("very quiet: move closer to the mic")
    if snr < 15:
        warnings.append("noisy background: try a quieter room")
    if x.size / sr < 0.3:
        warnings.append("very short: did the recording catch the phrase?")
    return {"duration_s": round(x.size / sr, 2), "snr_db": round(snr, 1),
            "peak_db": round(peak_db, 1), "clipped_frac": round(clipped, 4),
            "warnings": warnings}


def record_seconds(seconds: float, sr: int = 16000) -> np.ndarray:
    sd = _sd()
    data = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return data.ravel()


def beep(sr: int = 16000, freq: float = 880.0, ms: int = 150):
    try:
        sd = _sd()
        t = np.arange(int(sr * ms / 1000)) / sr
        sd.play(0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32), sr)
        sd.wait()
    except RuntimeError:
        print("\a", end="", flush=True)


def play_file(path, sr: int = 16000) -> float:
    """Starts playback without blocking. Returns the clip duration in seconds."""
    data = read_wav(path, sr)
    _sd().play(data, sr)
    return data.size / sr


class MicStream:
    """Background microphone capture that yields fixed-size blocks."""

    def __init__(self, sr: int = 16000, block_ms: int = 20):
        self.sr = sr
        self.blocksize = int(sr * block_ms / 1000)
        self.q = queue.Queue(maxsize=500)
        self._stream = None
        self._stop = threading.Event()

    def _callback(self, indata, frames, time_info, status):
        try:
            self.q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def __enter__(self):
        sd = _sd()
        self._stream = sd.InputStream(samplerate=self.sr, channels=1, dtype="float32",
                                      blocksize=self.blocksize, callback=self._callback)
        self._stream.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def blocks(self):
        while not self._stop.is_set():
            try:
                yield self.q.get(timeout=0.5)
            except queue.Empty:
                continue

    def stop(self):
        self._stop.set()
