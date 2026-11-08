"""Cuts a continuous stream into candidate cues.

The same state machine serves both inputs:
  audio    activity = block energy in dB, thresholds relative to a tracked noise floor
  gesture  activity = how far the face is from its relaxed baseline (normalised units)
"""

from collections import deque

import numpy as np


class ActivitySegmenter:
    def __init__(self, on_threshold: float, off_threshold: float, min_on_frames: int = 2,
                 hangover_frames: int = 15, min_frames: int = 10, max_frames: int = 150,
                 preroll_frames: int = 5, adaptive_floor: bool = False,
                 floor_alpha: float = 0.05):
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.min_on_frames = min_on_frames
        self.hangover_frames = hangover_frames
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.adaptive_floor = adaptive_floor
        self.floor_alpha = floor_alpha
        self.floor = None
        self._pre = deque(maxlen=max(preroll_frames, min_on_frames))
        self._seg = []
        self._active = False
        self._on = 0
        self._off = 0

    def _levels(self):
        base = self.floor if (self.adaptive_floor and self.floor is not None) else 0.0
        return base + self.on_threshold, base + self.off_threshold

    def push(self, value: float, payload) -> list:
        """Feed one frame. Returns a list of completed segments (each a list of payloads)."""
        if self.adaptive_floor and self.floor is None:
            self.floor = value
        on_level, off_level = self._levels()
        done = []
        if not self._active:
            self._pre.append(payload)
            if value > on_level:
                self._on += 1
                if self._on >= self.min_on_frames:
                    self._active = True
                    self._seg = list(self._pre)
                    self._pre.clear()
                    self._off = 0
            else:
                self._on = 0
                if self.adaptive_floor:
                    self.floor += self.floor_alpha * (value - self.floor)
        else:
            self._seg.append(payload)
            self._off = self._off + 1 if value < off_level else 0
            if self._off >= self.hangover_frames or len(self._seg) >= self.max_frames:
                keep = len(self._seg) - max(0, self._off - 3)
                seg = self._seg[:keep]
                if len(seg) >= self.min_frames:
                    done.append(seg)
                self._active, self._seg, self._on, self._off = False, [], 0, 0
        return done


class AudioVAD:
    """Energy-based voice activity detection on fixed-size blocks of samples."""

    def __init__(self, sample_rate: int = 16000, block_ms: int = 20):
        self.sr = sample_rate
        frames_per_s = 1000 // block_ms
        self.seg = ActivitySegmenter(
            on_threshold=10.0, off_threshold=6.0,             # dB above the noise floor
            min_on_frames=2, hangover_frames=int(0.3 * frames_per_s),
            min_frames=int(0.2 * frames_per_s), max_frames=int(3.0 * frames_per_s),
            preroll_frames=5, adaptive_floor=True)

    @staticmethod
    def level_db(block: np.ndarray) -> float:
        return 10.0 * np.log10(float(np.mean(np.square(block))) + 1e-12)

    def push(self, block: np.ndarray) -> list:
        """Returns a list of utterances (1-D float arrays) completed by this block."""
        block = np.asarray(block, dtype=np.float32).ravel()
        return [np.concatenate(s) for s in self.seg.push(self.level_db(block), block)]


class GestureSegmenter:
    """Finds deliberate movements away from the relaxed baseline (~30 fps input)."""

    def __init__(self, fps: int = 30, on: float = 2.5, off: float = 1.5):
        self.seg = ActivitySegmenter(
            on_threshold=on, off_threshold=off, min_on_frames=3,
            hangover_frames=int(0.2 * fps), min_frames=int(0.3 * fps),
            max_frames=int(3.0 * fps), preroll_frames=4)

    def push(self, normalized_features: np.ndarray) -> list:
        v = np.asarray(normalized_features, dtype=np.float32)
        return [np.stack(s) for s in self.seg.push(float(np.linalg.norm(v)), v)]
