"""The core loop: cue in -> match -> speak the phrase -> log -> adapt."""

import hashlib
import threading
import time
from typing import Callable, Optional

import numpy as np

from . import ARM_LABEL
from .config import Config
from .db import PhraseBank
from .fatigue import AdaptationMonitor
from .matcher import MatchResult, TemplateMatcher
from .notify import Notifier


class Engine:
    def __init__(self, cfg: Config, bank: PhraseBank, notifier: Optional[Notifier] = None,
                 clone=None, display: Callable[[str], None] = print,
                 player: Optional[Callable[[str], float]] = None,
                 clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.bank = bank
        self.notifier = notifier or Notifier(display=display)
        self.clone = clone
        self.display = display
        self.player = player            # injectable for tests; defaults to audio_io.play_file
        self.clock = clock              # injectable so the simulator can model timing
        self.matchers = {
            m: TemplateMatcher(cfg.margin_threshold, cfg.threshold_slack) for m in ("voice", "gesture")
        }
        self.monitor = AdaptationMonitor()
        self.lock = threading.Lock()
        self.speaking_until = 0.0
        self.armed_until = 0.0
        self.retry_window_s = 6.0
        self._last_reject = None   # (event_id, time, modality, confidence)
        self.reload()

    def reload(self):
        """(Re)load active cue templates from the phrase bank."""
        for modality, matcher in self.matchers.items():
            matcher.clear()
            for phrase_id, feats in self.bank.active_templates(modality):
                matcher.add(ARM_LABEL if phrase_id is None else str(phrase_id), feats)

    def mic_muted(self) -> bool:
        """True while Encore is talking, so it doesn't hear and trigger itself."""
        return self.clock() < self.speaking_until

    def handle_cue(self, modality: str, features: np.ndarray, ts: Optional[str] = None) -> MatchResult:
        """ts overrides the logged timestamp (used by the simulator)."""
        with self.lock:
            res = self.matchers[modality].match(features)
            now = self.clock()

            if res.accepted and res.label == ARM_LABEL:
                self.armed_until = now + self.cfg.arm_window_s
                self.display(f"[armed for {self.cfg.arm_window_s:.0f}s]")
                self.bank.log_event(modality, None, True, res.confidence, res.distance,
                                    res.duration_ratio, "arm", ts=ts)
                return res

            armed = now <= self.armed_until
            if res.accepted and self.cfg.require_arm and not armed:
                res.accepted, res.reason = False, "cue heard but not armed"

            # Which rejections are real attempts? Armed ones, or ones the user immediately
            # retries. Everything else may just be conversation and is logged as background,
            # so a talkative classroom doesn't look like failing speech.
            phrase_id = int(res.label) if res.label and res.label.isdigit() else None
            if res.accepted:
                reason = "ok"
            elif armed:
                reason = f"missed: {res.reason}"
            else:
                reason = f"background: {res.reason}"
            event_id = self.bank.log_event(modality, phrase_id if res.accepted else None,
                                           res.accepted, res.confidence, res.distance,
                                           res.duration_ratio, reason, ts=ts)

            advices = []
            if res.accepted:
                last = self._last_reject
                if last and last[2] == modality and (now - last[1]) <= self.retry_window_s:
                    self.bank.mark_missed(last[0])
                    advices.append(self.monitor.record(modality, False, last[3]))
                self._last_reject = None
                advices.append(self.monitor.record(modality, True, res.confidence,
                                                   res.duration_ratio))
            elif armed:
                advices.append(self.monitor.record(modality, False, res.confidence))
            else:
                self._last_reject = (event_id, now, modality, res.confidence)
            if any(a.changed for a in advices):
                final = advices[-1]
                self.display(f"\n[mode -> {final.mode}] {final.message}\n")

            if res.accepted and phrase_id is not None:
                self.speak(phrase_id)
                self.armed_until = 0.0
            else:
                self.display(f"  (no phrase: {res.reason}, confidence {res.confidence:.2f})")
            return res

    def speak(self, phrase_id: int) -> bool:
        phrase = self.bank.get_phrase(phrase_id)
        if phrase is None:
            self.display(f"[unknown phrase {phrase_id}]")
            return False
        self.display(f"\n  >>  {phrase.text}\n")   # always captioned on screen too
        path = phrase.audio_path if phrase.has_audio else None
        if path is None and self.clone is not None and self.clone.available:
            key = hashlib.sha1(phrase.text.encode()).hexdigest()[:12]
            cached = self.cfg.cache_dir / f"clone_{key}.wav"
            try:
                path = str(cached if cached.exists() else self.clone.synthesize(phrase.text, cached))
            except Exception as e:
                self.display(f"  [voice clone failed: {e}]")
        if path:
            try:
                play = self.player
                if play is None:
                    from .audio_io import play_file as play
                duration = play(path)
                self.speaking_until = self.clock() + duration + self.cfg.mic_mute_tail_s
            except RuntimeError as e:
                self.display(f"  [playback unavailable: {e}]")
        else:
            self.display("  (no recording yet: record it with `encore record-phrase "
                         f"{phrase_id}`)")
        if phrase.action:
            self.notifier.handle(phrase.action, phrase.text)
        return True
