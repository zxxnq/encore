"""Few-shot cue matching with dynamic time warping (DTW).

Why not a generic speech recogniser? Off-the-shelf ASR is trained on typical
speech and degrades badly on dysarthric speech. Here the user's own examples are
the reference, so a slurred cue still matches as long as it is slurred
consistently. DTW also absorbs slower speech, which is the main way a fatigued
afternoon voice differs from a morning one.

The same matcher handles gestures: any cue is just a (time, features) sequence.
Each cue's acceptance threshold is calibrated from the spread of the user's own
examples, so there is no global number to hand-tune per person.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


def dtw_distance(a: np.ndarray, b: np.ndarray, band_frac: float = 0.35) -> float:
    """Length-normalised DTW distance between two (frames, dims) sequences.

    A band around the (length-scaled) diagonal keeps it fast and stops degenerate
    alignments. Returns inf if no path fits inside the band.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))
    width = max(2, int(band_frac * max(n, m)) + 1)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    scale = m / n
    for i in range(1, n + 1):
        center = i * scale
        j0 = max(1, int(center - width))
        j1 = min(m, int(center + width))
        prev, cur, ci = D[i - 1], D[i], cost[i - 1]
        for j in range(j0, j1 + 1):
            best = prev[j]
            if cur[j - 1] < best:
                best = cur[j - 1]
            if prev[j - 1] < best:
                best = prev[j - 1]
            cur[j] = ci[j - 1] + best
    return float(D[n, m] / (n + m))


@dataclass
class MatchResult:
    label: Optional[str]
    accepted: bool
    distance: float
    confidence: float
    duration_ratio: Optional[float]
    runner_up: Optional[str]
    reason: str


class TemplateMatcher:
    def __init__(self, margin_threshold: float = 0.10, slack: float = 1.6,
                 band_frac: float = 0.35, single_example_threshold: Optional[float] = None,
                 min_ratio: float = 0.5, max_ratio: float = 2.2):
        self.margin_threshold = margin_threshold
        self.slack = slack
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.band_frac = band_frac
        self.single_example_threshold = single_example_threshold
        self.templates = {}
        self.thresholds = {}
        self._dirty = True

    def add(self, label: str, features: np.ndarray):
        self.templates.setdefault(str(label), []).append(np.asarray(features, dtype=np.float32))
        self._dirty = True

    def clear(self):
        self.templates.clear()
        self.thresholds.clear()
        self._dirty = True

    @property
    def labels(self):
        return list(self.templates)

    def calibrate(self):
        """Per-cue threshold = slack x the largest distance between the user's own examples."""
        spreads = []
        for label, examples in self.templates.items():
            if len(examples) >= 2:
                ds = [dtw_distance(examples[i], examples[j], self.band_frac)
                      for i in range(len(examples)) for j in range(i + 1, len(examples))]
                ds = [d for d in ds if np.isfinite(d)]
                if ds:
                    self.thresholds[label] = max(ds) * self.slack
                    spreads.append(max(ds))
        fallback = self.single_example_threshold
        if fallback is None:
            fallback = float(np.median(spreads)) * self.slack if spreads else float("inf")
        for label in self.templates:
            self.thresholds.setdefault(label, fallback)
        self._dirty = False

    def match(self, features: np.ndarray) -> MatchResult:
        if not self.templates:
            return MatchResult(None, False, float("inf"), 0.0, None, None, "no cues enrolled")
        if self._dirty:
            self.calibrate()
        q = np.asarray(features, dtype=np.float32)
        per_label = {}
        for label, examples in self.templates.items():
            dists = [dtw_distance(q, t, self.band_frac) for t in examples]
            k = int(np.argmin(dists))
            per_label[label] = (dists[k], len(examples[k]))
        ranked = sorted(per_label.items(), key=lambda kv: kv[1][0])
        best, (d1, tlen) = ranked[0]
        runner_up, d2 = (ranked[1][0], ranked[1][1][0]) if len(ranked) > 1 else (None, float("inf"))
        margin = 1.0 - d1 / d2 if np.isfinite(d2) and d2 > 0 else 1.0
        thr = self.thresholds[best]
        duration_ratio = len(q) / tlen if tlen else None

        if not np.isfinite(d1):
            return MatchResult(best, False, d1, 0.0, duration_ratio, runner_up, "no alignment")
        closeness = max(0.0, 1.0 - d1 / thr) if np.isfinite(thr) and thr > 0 else 0.5
        confidence = float(np.clip(0.5 * closeness + 0.5 * min(1.0, margin / 0.4), 0.0, 1.0))

        # A cue far longer or shorter than its examples isn't that cue (e.g. a full
        # sentence of lecture). The wide window still allows afternoon slowing.
        if duration_ratio and not (self.min_ratio <= duration_ratio <= self.max_ratio):
            return MatchResult(best, False, d1, confidence * 0.5, duration_ratio, runner_up,
                               f"length x{duration_ratio:.1f} doesn't fit any cue")
        if d1 > thr:
            return MatchResult(best, False, d1, confidence * 0.5, duration_ratio, runner_up,
                               "not close enough to any cue")
        if margin < self.margin_threshold:
            return MatchResult(best, False, d1, confidence * 0.5, duration_ratio, runner_up,
                               f"ambiguous between cues {best} and {runner_up}")
        return MatchResult(best, True, d1, confidence, duration_ratio, runner_up, "ok")
