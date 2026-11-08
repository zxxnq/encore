"""Watches how well voice cues are working and suggests when to lean on gestures.

ALS speech in this case is strongest in the morning and fades through the day
(165 -> 118 words per minute, 98% -> 91% intelligibility). Encore sees the same
thing indirectly: cues get slower (longer than the morning template) and less
confidently matched. When that persists, it suggests hybrid or gesture mode.
It never removes a working input; the advice is shown, not forced.

Every attempt is logged, so the same data gives a free hour-by-hour picture of
the user's day that can be shared with their clinic if they choose. This is a
proxy signal, not a validated clinical measure.
"""

import csv
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Optional

MODES = ("voice", "hybrid", "gesture")

MESSAGES = {
    "voice": "Voice cues are working well. Voice mode.",
    "hybrid": "Voice cues are taking more effort right now. Gesture cues are ready as backup.",
    "gesture": "Voice cues are struggling right now. Gesture cues are the easier path.",
}


@dataclass
class ModeAdvice:
    mode: str
    changed: bool
    message: str
    accept_rate: Optional[float] = None
    confidence: Optional[float] = None
    slowing: Optional[float] = None


class AdaptationMonitor:
    def __init__(self, window: int = 16, min_events: int = 4, patience: int = 3,
                 recover_patience: int = 4, recover_margin: float = 0.08,
                 hybrid_accept: float = 0.75, gesture_accept: float = 0.55,
                 hybrid_conf: float = 0.40, gesture_conf: float = 0.25,
                 hybrid_slowing: float = 1.25, gesture_slowing: float = 1.45):
        self.voice = deque(maxlen=window)
        self.min_events = min_events
        self.patience = patience
        self.recover_patience = recover_patience
        self.recover_margin = recover_margin
        self.limits = dict(hybrid_accept=hybrid_accept, gesture_accept=gesture_accept,
                           hybrid_conf=hybrid_conf, gesture_conf=gesture_conf,
                           hybrid_slowing=hybrid_slowing, gesture_slowing=gesture_slowing)
        self.mode = "voice"
        self._candidate = None
        self._streak = 0

    def record(self, modality: str, accepted: bool, confidence: float,
               duration_ratio: Optional[float] = None) -> ModeAdvice:
        if modality == "voice":
            self.voice.append((bool(accepted), float(confidence), duration_ratio))
        return self._evaluate()

    def _level(self, accept_rate, confidence, slowing, margin: float) -> str:
        """Which mode the numbers call for. margin > 0 makes the limits stricter."""
        L = self.limits
        slow_m = margin * 2   # slowing is a ratio around 1.0-1.5, so give it a wider margin
        if (accept_rate < L["gesture_accept"] + margin or confidence < L["gesture_conf"] + margin
                or (slowing and slowing > L["gesture_slowing"] - slow_m)):
            return "gesture"
        if (accept_rate < L["hybrid_accept"] + margin or confidence < L["hybrid_conf"] + margin
                or (slowing and slowing > L["hybrid_slowing"] - slow_m)):
            return "hybrid"
        return "voice"

    def _evaluate(self) -> ModeAdvice:
        if len(self.voice) < self.min_events:
            return ModeAdvice(self.mode, False, "Collecting a few voice cues first.")
        accept_rate = sum(a for a, _, _ in self.voice) / len(self.voice)
        confidence = sum(c for _, c, _ in self.voice) / len(self.voice)
        ratios = [r for a, _, r in self.voice if a and r]
        slowing = median(ratios) if ratios else None

        target = self._level(accept_rate, confidence, slowing, 0.0)
        # Asymmetric hysteresis: stepping back up to an easier mode needs the numbers to
        # clear the limits by a margin, for longer, so the advice doesn't bounce around.
        if MODES.index(target) < MODES.index(self.mode):
            target = min(MODES.index(self.mode),
                         MODES.index(self._level(accept_rate, confidence, slowing,
                                                 self.recover_margin)))
            target = MODES[target]
        needed = (self.recover_patience if MODES.index(target) < MODES.index(self.mode)
                  else self.patience)

        changed = False
        if target == self.mode:
            self._candidate, self._streak = None, 0
        else:
            self._streak = self._streak + 1 if target == self._candidate else 1
            self._candidate = target
            if self._streak >= needed:
                self.mode, changed = target, True
                self._candidate, self._streak = None, 0
        return ModeAdvice(self.mode, changed, MESSAGES[self.mode],
                          accept_rate, confidence, slowing)


def hourly_summary(events: list) -> list:
    """Groups logged voice cue attempts by hour of day (background talk excluded)."""
    buckets = {}
    for e in events:
        if e["modality"] != "voice" or (e.get("reason") or "").startswith(("background", "arm")):
            continue
        hour = int(e["ts"][11:13])
        buckets.setdefault(hour, []).append(e)
    rows = []
    for hour in sorted(buckets):
        evs = buckets[hour]
        ratios = [e["duration_ratio"] for e in evs if e["accepted"] and e["duration_ratio"]]
        rows.append({
            "hour": f"{hour:02d}:00",
            "attempts": len(evs),
            "accept_rate": round(sum(e["accepted"] for e in evs) / len(evs), 2),  # first-try success
            "mean_confidence": round(sum(e["confidence"] or 0 for e in evs) / len(evs), 2),
            "median_slowing": round(median(ratios), 2) if ratios else None,
        })
    return rows


def format_report(rows: list) -> str:
    if not rows:
        return "No voice cue attempts logged in this period."
    lines = [f"{'hour':<7}{'tries':>6}{'accepted':>10}{'confidence':>12}{'slowing':>9}"]
    for r in rows:
        slow = f"x{r['median_slowing']:.2f}" if r["median_slowing"] else "-"
        lines.append(f"{r['hour']:<7}{r['attempts']:>6}{r['accept_rate']:>10.0%}"
                     f"{r['mean_confidence']:>12.2f}{slow:>9}")
    lines.append("slowing = cue length vs. the recorded template (x1.00 = same pace)")
    return "\n".join(lines)


def write_csv(rows: list, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["hour", "attempts", "accept_rate",
                                          "mean_confidence", "median_slowing"])
        w.writeheader()
        w.writerows(rows)
