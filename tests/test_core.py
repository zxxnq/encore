"""Run with:  python -m unittest discover -s tests -v"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from encore import ARM_LABEL
from encore.audio_io import quality_check, read_wav, trim_silence, write_wav
from encore.config import Config
from encore.db import PhraseBank
from encore.engine import Engine
from encore.evaluate import compare_conditions, leave_one_out, load_manifest, train_test
from encore.fatigue import AdaptationMonitor, hourly_summary
from encore.features import voice_features
from encore.gestures import GestureBaseline
from encore.matcher import TemplateMatcher, dtw_distance
from encore.segmenter import AudioVAD, GestureSegmenter
from encore.simulate import SR, fatigue_at, make_vocabulary, synth_gesture, synth_word


class TestFeaturesAndDTW(unittest.TestCase):
    def test_mfcc_shape(self):
        f = voice_features(np.random.default_rng(0).normal(0, 0.1, SR))
        self.assertEqual(f.shape[1], 26)
        self.assertGreater(f.shape[0], 50)

    def test_dtw_identity_and_stretch(self):
        rng = np.random.default_rng(0)
        w = make_vocabulary(2)
        a = voice_features(synth_word(w[0], rng))
        slow = voice_features(synth_word(w[0], rng, rate=1.4))
        other = voice_features(synth_word(w[1], rng))
        self.assertAlmostEqual(dtw_distance(a, a), 0.0, places=5)
        self.assertLess(dtw_distance(a, slow), dtw_distance(a, other))


class TestMatcher(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.vocab = make_vocabulary(8)
        self.m = TemplateMatcher()
        for i, w in enumerate(self.vocab[:6]):
            for _ in range(3):
                self.m.add(str(i), voice_features(synth_word(w, self.rng)))

    def test_morning_recognition(self):
        ok = sum(1 for i, w in enumerate(self.vocab[:6]) for _ in range(3)
                 if (r := self.m.match(voice_features(synth_word(w, self.rng)))).accepted
                 and r.label == str(i))
        self.assertGreaterEqual(ok, 17)  # of 18

    def test_unknown_words_rarely_accepted(self):
        fa = sum(self.m.match(voice_features(synth_word(w, self.rng))).accepted
                 for w in self.vocab[6:] for _ in range(5))
        self.assertLessEqual(fa, 2)  # of 10

    def test_slowing_is_measured(self):
        r = self.m.match(voice_features(synth_word(self.vocab[0], self.rng, **fatigue_at(16))))
        self.assertGreater(r.duration_ratio, 1.2)

    def test_empty_matcher(self):
        self.assertFalse(TemplateMatcher().match(np.zeros((10, 26))).accepted)


class TestGestures(unittest.TestCase):
    def test_baseline_keeps_direction(self):
        rng = np.random.default_rng(2)
        base = GestureBaseline.from_frames(rng.normal(0, 0.005, (60, 6)))
        left = base.normalize(np.array([0, 0, -0.2, 0, 0, 0]))
        right = base.normalize(np.array([0, 0, 0.2, 0, 0, 0]))
        self.assertLess(left[2], 0)
        self.assertGreater(right[2], 0)
        self.assertEqual(GestureBaseline.from_json(base.to_json()).mean.shape, (6,))

    def test_gesture_matching_and_reenrollment(self):
        rng = np.random.default_rng(3)
        names = ["tilt_left", "tilt_right", "nod", "brow_raise"]
        m = TemplateMatcher()
        for g in names:
            for _ in range(3):
                m.add(g, synth_gesture(g, rng))

        def acc(strength):
            return sum(1 for g in names for _ in range(5)
                       if (r := m.match(synth_gesture(g, rng, strength=strength))).accepted
                       and r.label == g) / 20

        self.assertGreaterEqual(acc(1.0), 0.95)
        self.assertLessEqual(acc(0.3), 0.2)          # weakened movement fails old cues
        m.clear()
        for g in names:
            for _ in range(3):
                m.add(g, synth_gesture(g, rng, strength=0.3))
        self.assertGreaterEqual(acc(0.3), 0.9)       # re-recording restores them


class TestSegmenters(unittest.TestCase):
    def test_audio_vad_finds_utterances(self):
        rng = np.random.default_rng(4)
        w = make_vocabulary(2)
        quiet = lambda s: rng.normal(0, 10 ** (-50 / 20), int(SR * s)).astype(np.float32)
        sig = np.concatenate([quiet(1), synth_word(w[0], rng), quiet(0.8),
                              synth_word(w[1], rng), quiet(1)])
        vad, segs = AudioVAD(), []
        for i in range(0, len(sig) - 320, 320):
            segs += vad.push(sig[i:i + 320])
        self.assertEqual(len(segs), 2)

    def test_gesture_segmenter(self):
        rng = np.random.default_rng(5)
        stream = [rng.normal(0, 0.25, 6) for _ in range(30)]
        stream += list(synth_gesture("nod", rng)) + [rng.normal(0, 0.25, 6) for _ in range(30)]
        seg, out = GestureSegmenter(), []
        for f in stream:
            out += seg.push(f)
        self.assertEqual(len(out), 1)


class TestMonitor(unittest.TestCase):
    def test_hysteresis_and_switch(self):
        m = AdaptationMonitor()
        for _ in range(6):
            self.assertEqual(m.record("voice", True, 0.6, 1.0).mode, "voice")
        modes = [m.record("voice", True, 0.2, 1.5).mode for _ in range(14)]
        self.assertEqual(modes[:2], ["voice", "voice"])   # a couple of bad cues don't flip it
        self.assertEqual(modes[-1], "gesture")

    def test_no_bounce_back(self):
        m = AdaptationMonitor()
        for _ in range(20):
            m.record("voice", True, 0.3, 1.35)
        self.assertEqual(m.mode, "hybrid")
        for _ in range(3):                                 # a short good patch...
            m.record("voice", True, 0.7, 1.0)
        self.assertEqual(m.mode, "hybrid")                 # ...doesn't flip it straight back
        for _ in range(16):
            m.record("voice", True, 0.7, 1.0)
        self.assertEqual(m.mode, "voice")                  # sustained recovery does

    def test_gesture_events_ignored(self):
        m = AdaptationMonitor()
        for _ in range(10):
            m.record("gesture", False, 0.0)
        self.assertEqual(m.mode, "voice")


class TestBankAndEngine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config.load(self.tmp.name)
        self.bank = PhraseBank(self.cfg.db_path, self.cfg.features_dir)

    def tearDown(self):
        self.bank.close()
        self.tmp.cleanup()

    def test_seed_and_reenroll(self):
        seed = Path(__file__).resolve().parents[1] / "encore" / "seed_phrases.json"
        n = self.bank.seed_from_json(seed)
        self.assertGreater(n, 15)
        self.assertEqual(self.bank.seed_from_json(seed), 0)   # idempotent
        pid = self.bank.list_phrases("home")[0].id
        self.bank.add_template(pid, "voice", np.zeros((5, 26)))
        self.assertEqual(self.bank.deactivate_templates(pid, "voice"), 1)
        self.assertEqual(self.bank.active_templates("voice"), [])

    def _engine_with_cues(self, require_arm=False):
        rng = np.random.default_rng(6)
        vocab = make_vocabulary(4)
        ids = [self.bank.add_phrase(f"phrase {i}", "test") for i in range(3)]
        for pid, w in zip(ids, vocab):
            for _ in range(3):
                self.bank.add_template(pid, "voice", voice_features(synth_word(w, rng)))
        for _ in range(3):
            self.bank.add_template(None, "voice", voice_features(synth_word(vocab[3], rng)))
        self.cfg.require_arm = require_arm
        t = [1000.0]
        spoken = []
        eng = Engine(self.cfg, self.bank, display=spoken.append, player=lambda p: 0.0,
                     clock=lambda: t[0])
        return eng, vocab, ids, rng, t, spoken

    def test_engine_speaks_and_logs(self):
        eng, vocab, ids, rng, t, spoken = self._engine_with_cues()
        r = eng.handle_cue("voice", voice_features(synth_word(vocab[1], rng)))
        self.assertTrue(r.accepted)
        self.assertTrue(any("phrase 1" in s for s in spoken))
        self.assertEqual(len(self.bank.events_since("2000-01-01")), 1)

    def test_arming(self):
        eng, vocab, ids, rng, t, spoken = self._engine_with_cues(require_arm=True)
        r = eng.handle_cue("voice", voice_features(synth_word(vocab[0], rng)))
        self.assertFalse(r.accepted)                                # not armed yet
        eng.handle_cue("voice", voice_features(synth_word(vocab[3], rng)))  # arm cue
        self.assertEqual(eng.matchers["voice"].labels.count(ARM_LABEL), 1)
        t[0] += 2
        self.assertTrue(eng.handle_cue("voice", voice_features(synth_word(vocab[0], rng))).accepted)

    def test_retry_marks_missed_and_background_excluded(self):
        eng, vocab, ids, rng, t, spoken = self._engine_with_cues()
        noise = np.random.default_rng(9).normal(0, 1, (40, 26)).astype(np.float32)
        eng.handle_cue("voice", noise)                 # rejected -> background for now
        t[0] += 3
        eng.handle_cue("voice", voice_features(synth_word(vocab[0], rng)))   # quick retry
        reasons = [e["reason"] for e in self.bank.events_since("2000-01-01")]
        self.assertTrue(reasons[0].startswith("missed"))
        t[0] += 120
        eng.handle_cue("voice", noise)                 # unrelated, no retry follows
        rows = hourly_summary(self.bank.events_since("2000-01-01"))
        self.assertEqual(sum(r["attempts"] for r in rows), 2)   # background not counted


class TestAudioAndEvaluate(unittest.TestCase):
    def test_wav_roundtrip_trim_quality_and_evaluate(self):
        rng = np.random.default_rng(8)
        vocab = make_vocabulary(3)
        with tempfile.TemporaryDirectory() as d:
            rows = []
            for spk in ("s1", "s2"):
                for li, w in enumerate(vocab):
                    for k in range(3):
                        sig = np.concatenate([np.zeros(4000, np.float32), synth_word(w, rng),
                                              np.zeros(4000, np.float32)])
                        p = Path(d) / f"{spk}_{li}_{k}.wav"
                        write_wav(p, sig)
                        rows.append({"path": p.name, "label": f"w{li}", "speaker": spk,
                                     "group": "mild" if spk == "s1" else "severe"})
            x = read_wav(Path(d) / "s1_0_0.wav")
            self.assertLess(len(trim_silence(x)), len(x))
            self.assertIn("snr_db", quality_check(x))
            man = Path(d) / "manifest.csv"
            with open(man, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["path", "label", "speaker", "group"])
                w.writeheader()
                w.writerows(rows)
            summary = leave_one_out(load_manifest(man))
            self.assertEqual(summary["overall"]["n"], 18)
            self.assertGreaterEqual(summary["overall"]["accuracy"], 0.8)
            self.assertEqual(set(summary["by_group"]), {"mild", "severe"})

    def test_compare_fresh_vs_tired(self):
        rng = np.random.default_rng(10)
        vocab = make_vocabulary(4)
        with tempfile.TemporaryDirectory() as d:
            rows = []
            for group, hour in (("normal", 9), ("slurred", 16)):
                for li, w in enumerate(vocab):
                    for k in range(3):
                        p = Path(d) / f"{group}_{li}_{k}.wav"
                        write_wav(p, synth_word(w, rng, **fatigue_at(hour)))
                        rows.append({"path": p, "label": f"w{li}", "speaker": "a", "group": group})
            tt = train_test(rows, "normal", "slurred")
            self.assertEqual(tt["overall"]["n"], 12)
            lines = compare_conditions(rows)
            self.assertEqual(len(lines), 3)
            self.assertEqual([s["n"] for _, s in lines], [12, 12, 12])


if __name__ == "__main__":
    unittest.main()
