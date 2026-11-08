"""Encore command line.  Run `python -m encore --help`."""

import argparse
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import ARM_LABEL
from .config import Config
from .db import PhraseBank

SEED_FILE = Path(__file__).with_name("seed_phrases.json")


def _open(args):
    cfg = Config.load(args.data)
    return cfg, PhraseBank(cfg.db_path, cfg.features_dir)


def _phrase_arg(value: str):
    return None if value.lower() in ("arm", ARM_LABEL) else int(value)


# ---- setup & phrases -----------------------------------------------------------------
def cmd_init(args):
    cfg, bank = _open(args)
    added = bank.seed_from_json(SEED_FILE) if args.seed else 0
    print(f"Phrase bank ready at {cfg.db_path}" + (f" ({added} starter phrases added)" if args.seed else ""))
    print("Next: `encore phrases`, then `encore record-phrase <id>` (best before 11 AM).")


def cmd_phrases(args):
    _, bank = _open(args)
    counts = bank.template_counts()
    for deck in ([args.deck] if args.deck else bank.decks()):
        print(f"\n[{deck}]")
        for p in bank.list_phrases(deck):
            c = counts.get(p.id, {})
            flags = ("rec " if p.has_audio else "--- ") + f"v{c.get('voice', 0)} g{c.get('gesture', 0)}"
            act = f"  ({p.action})" if p.action else ""
            print(f"  {p.id:>3}  {flags:<12} {p.text}{act}")
    arm = counts.get(None, {})
    if arm:
        print(f"\n  arm cue: v{arm.get('voice', 0)} g{arm.get('gesture', 0)}")
    print("\nrec = recorded in own voice;  vN/gN = voice/gesture cue examples")


def cmd_add_phrase(args):
    _, bank = _open(args)
    pid = bank.add_phrase(args.text, args.deck, args.action)
    print(f"Added phrase {pid}: {args.text}")


def cmd_delete_phrase(args):
    _, bank = _open(args)
    bank.delete_phrase(args.id)
    print(f"Deleted phrase {args.id}")


def cmd_record_phrase(args):
    from .audio_io import (beep, quality_check, read_wav, record_seconds, trim_silence,
                           write_wav)
    cfg, bank = _open(args)
    phrase = bank.get_phrase(args.id)
    if not phrase:
        sys.exit(f"No phrase {args.id}")
    if args.from_file:
        audio = read_wav(args.from_file, cfg.sample_rate)
    else:
        if datetime.now().hour >= 12:
            print("Tip: recordings made in the morning usually sound clearest.")
        print(f'\nSay: "{phrase.text}"')
        input(f"Press Enter, wait for the beep, then speak (up to {args.seconds:.0f}s)...")
        beep(cfg.sample_rate)
        audio = record_seconds(args.seconds, cfg.sample_rate)
    audio = trim_silence(audio, cfg.sample_rate)
    q = quality_check(audio, cfg.sample_rate)
    for w in q["warnings"]:
        print(f"  ! {w}")
    out = cfg.audio_dir / f"phrase_{phrase.id}_{datetime.now():%Y%m%d_%H%M%S}.wav"
    write_wav(out, audio, cfg.sample_rate)
    bank.set_audio(phrase.id, out, q)
    print(f"Saved {q['duration_s']}s recording (SNR {q['snr_db']} dB) -> {out.name}")


def _capture_voice_cue(cfg, seconds):
    from .audio_io import beep, record_seconds, trim_silence
    from .features import voice_features
    beep(cfg.sample_rate)
    audio = trim_silence(record_seconds(seconds, cfg.sample_rate), cfg.sample_rate)
    return voice_features(audio, cfg.sample_rate)


def _phrase_label(bank, pid):
    if pid is None:
        return "the ARM cue"
    phrase = bank.get_phrase(pid)
    if phrase is None:
        sys.exit(f"No phrase {pid}. See `encore phrases`.")
    return f'"{phrase.text}"'


def cmd_train_voice(args):
    cfg, bank = _open(args)
    pid = _phrase_arg(args.id)
    label = _phrase_label(bank, pid)
    if args.replace:
        n = bank.deactivate_templates(pid, "voice")
        print(f"Retired {n} old voice examples (kept in history).")
    print(f"\nTraining a voice cue for {label}.")
    print("Use any short word or sound. It does not need to be clear, just consistent.")
    for k in range(args.examples):
        input(f"Example {k + 1}/{args.examples}: press Enter, then say the cue after the beep...")
        bank.add_template(pid, "voice", _capture_voice_cue(cfg, args.seconds))
    print("Done.")


def _gesture_capture(cam, baseline, timeout_s=6.0):
    from .segmenter import GestureSegmenter
    seg, t0 = GestureSegmenter(), time.time()
    for f in cam.frames():
        if f is not None:
            out = seg.push(baseline.normalize(f))
            if out:
                return out[0]
        if time.time() - t0 > timeout_s:
            return None


def cmd_train_gesture(args):
    from .gestures import FaceCueCamera
    cfg, bank = _open(args)
    pid = _phrase_arg(args.id)
    label = _phrase_label(bank, pid)
    cam = FaceCueCamera(args.camera)
    try:
        input("Relax your face and look at the camera. Press Enter to calibrate (2s)...")
        baseline = cam.capture_baseline()
        if args.replace:
            n = bank.deactivate_templates(pid, "gesture")
            print(f"Retired {n} old gesture examples (kept in history).")
        print(f"\nTraining a gesture cue for {label}. Pick a movement that is easy today.")
        k = 0
        while k < args.examples:
            input(f"Example {k + 1}/{args.examples}: press Enter, do the gesture, then relax...")
            seq = _gesture_capture(cam, baseline)
            if seq is None:
                print("  Didn't catch a movement. Try a slightly bigger one.")
                continue
            bank.add_template(pid, "gesture", seq)
            k += 1
        print("Done.")
    finally:
        cam.close()


# ---- running -------------------------------------------------------------------------
def _build_engine(cfg, bank, args=None):
    from .engine import Engine
    from .notify import Notifier
    from .voice_clone import ElevenLabsVoice
    if args is not None and getattr(args, "require_arm", False):
        cfg.require_arm = True
    notifier = Notifier(bank.get_setting("ntfy_topic"), bank.get_setting("user_name", "Encore user"))
    return Engine(cfg, bank, notifier=notifier, clone=ElevenLabsVoice(bank))


def cmd_run(args):
    cfg, bank = _open(args)
    engine = _build_engine(cfg, bank, args)
    stop = threading.Event()
    threads = []

    if args.voice:
        try:
            from .audio_io import _sd
            _sd()
        except RuntimeError as e:
            print(f"Voice cues disabled: {e}\n")
            args.voice = False
    if args.voice:
        if not engine.matchers["voice"].labels:
            print("No voice cues trained yet (encore train-voice <id>).")

        def mic_loop():
            from .audio_io import MicStream
            from .features import voice_features
            from .segmenter import AudioVAD
            vad = AudioVAD(cfg.sample_rate, cfg.block_ms)
            with MicStream(cfg.sample_rate, cfg.block_ms) as mic:
                for block in mic.blocks():
                    if stop.is_set():
                        break
                    if engine.mic_muted():
                        continue
                    for utt in vad.push(block):
                        engine.handle_cue("voice", voice_features(utt, cfg.sample_rate))
        threads.append(threading.Thread(target=mic_loop, daemon=True))

    if args.gesture:
        from .gestures import FaceCueCamera
        from .segmenter import GestureSegmenter
        cam = FaceCueCamera(args.camera)
        print("Calibrating: relax your face and look at the camera...")
        baseline = cam.capture_baseline()

        def cam_loop():
            seg = GestureSegmenter()
            for f in cam.frames():
                if stop.is_set():
                    break
                if f is not None:
                    for seq in seg.push(baseline.normalize(f)):
                        engine.handle_cue("gesture", seq)
        threads.append(threading.Thread(target=cam_loop, daemon=True))

    for t in threads:
        t.start()
    print("\nEncore is listening. Type a phrase number + Enter to speak it directly,"
          " 'p' to list phrases, 'q' to quit.\n")
    try:
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd == "q":
                break
            if cmd == "p":
                for p in bank.list_phrases():
                    print(f"  {p.id:>3} [{p.deck}] {p.text}")
            elif cmd.isdigit():
                engine.speak(int(cmd))
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        print("Stopped.")


def cmd_say(args):
    cfg, bank = _open(args)
    engine = _build_engine(cfg, bank)
    if args.no_action:
        engine.notifier.topic = None
    engine.speak(args.id)
    time.sleep(max(0.0, engine.speaking_until - time.time()))


# ---- reports, notifications, cloning, evaluation -----------------------------------
def cmd_report(args):
    from .fatigue import format_report, hourly_summary, write_csv
    _, bank = _open(args)
    since = (datetime.now() - timedelta(days=args.days)).isoformat(timespec="seconds")
    rows = hourly_summary(bank.events_since(since))
    print(f"Voice cue performance by hour, last {args.days} day(s):\n")
    print(format_report(rows))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nSaved {args.csv}")


def cmd_notify_setup(args):
    _, bank = _open(args)
    bank.set_setting("ntfy_topic", args.topic)
    bank.set_setting("user_name", args.name)
    print(f"Notifications go to ntfy topic '{args.topic}'. Family: install the ntfy app "
          "and subscribe to that exact topic.")
    if args.test:
        from .notify import Notifier
        Notifier(args.topic, args.name).send(f"{args.name} set up Encore", "Test message")


def cmd_clone_consent(args):
    from .voice_clone import ElevenLabsVoice
    _, bank = _open(args)
    print("Consent: I am the owner of this voice and I agree that a synthetic copy may be "
          "made from my Encore recordings, for my own communication only.")
    if input(f"Type your full name ({args.name}) to agree: ").strip() != args.name:
        sys.exit("Name did not match. No consent recorded.")
    ElevenLabsVoice(bank).record_consent(args.name)
    print("Consent recorded.")


def cmd_clone_create(args):
    from .voice_clone import ElevenLabsVoice
    _, bank = _open(args)
    wavs = [p.audio_path for p in bank.list_phrases() if p.has_audio]
    if len(wavs) < 3:
        sys.exit("Record at least a few phrases first (about 1-2 minutes of speech total works best).")
    vid = ElevenLabsVoice(bank).create_clone(args.name, wavs[:25])
    print(f"Voice clone created: {vid}")


def cmd_clone_say(args):
    from .audio_io import play_file
    from .voice_clone import ElevenLabsVoice
    cfg, bank = _open(args)
    out = cfg.cache_dir / f"clone_say_{datetime.now():%H%M%S}.wav"
    ElevenLabsVoice(bank).synthesize(args.text, out)
    time.sleep(play_file(out))


def cmd_evaluate(args):
    from .evaluate import (compare_conditions, format_comparison, format_summary,
                           leave_one_out, load_manifest)
    rows = load_manifest(args.manifest)
    speakers = sorted({r["speaker"] for r in rows})
    print(f"{len(rows)} recordings from {len(speakers)} speaker(s): {', '.join(speakers)}\n")
    if args.compare:
        fresh, tired = args.compare
        print(format_comparison(compare_conditions(rows, fresh, tired)))
    else:
        print("Leave-one-out within each speaker:")
        print(format_summary(leave_one_out(rows)))


def cmd_collect(args):
    """Guided recording session for testing Encore on real voices."""
    import csv as _csv
    from .audio_io import beep, quality_check, record_seconds, trim_silence, write_wav
    cfg = Config.load(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    words = [w.strip() for w in args.words.split(",") if w.strip()]
    speaker = args.speaker.strip().lower().replace(" ", "_")
    conditions = [
        ("normal", "Say it in your normal voice, clearly."),
        ("slurred", "Now say it like you're exhausted: slowly, mouth barely moving, "
                    "dragging the vowels. Keep it the same way every time."),
    ]
    total = len(words) * args.reps * len(conditions)
    print(f"\nRecording {speaker}: {len(words)} words x {args.reps} takes x 2 voices "
          f"= {total} short clips (about {total * 5 // 60 + 1} minutes).")
    print("Sit about an arm's length from the mic in the quietest spot you can find.\n")

    new_file = not manifest.exists()
    with open(manifest, "a", newline="") as f:
        w = _csv.writer(f)
        if new_file:
            w.writerow(["path", "label", "speaker", "group"])
        n = 0
        for group, instruction in conditions:
            print(f"--- {group.upper()} VOICE ---  {instruction}\n")
            for word in words:
                for k in range(args.reps):
                    n += 1
                    while True:
                        input(f'[{n}/{total}] "{word}" ({group}, take {k + 1}): press Enter, '
                              "then speak after the beep...")
                        beep(cfg.sample_rate)
                        audio = trim_silence(record_seconds(args.seconds, cfg.sample_rate),
                                             cfg.sample_rate)
                        q = quality_check(audio, cfg.sample_rate)
                        if q["warnings"] and input(f"   ! {'; '.join(q['warnings'])}. "
                                                   "Redo? [Y/n] ").strip().lower() != "n":
                            continue
                        break
                    name = f"{speaker}_{group}_{word}_{k + 1}.wav"
                    write_wav(out / name, audio, cfg.sample_rate)
                    w.writerow([name, word, speaker, group])
                    f.flush()
    print(f"\nDone. Saved to {out}/. Next speaker: run collect again with their name.")
    print(f"When everyone is done: python -m encore evaluate {manifest} --compare normal slurred")


# ---- synthetic demo ------------------------------------------------------------------
def _demo_day(seed: int, require_arm: bool) -> dict:
    """Simulate one school day of cue use. Returns stats; prints nothing."""
    from .engine import Engine
    from .fatigue import hourly_summary
    from .features import voice_features
    from .simulate import fatigue_at, make_vocabulary, synth_gesture, synth_word

    rng = np.random.default_rng(seed)
    vocab = make_vocabulary(8, seed=seed)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config.load(tmp)
        cfg.require_arm = require_arm
        bank = PhraseBank(cfg.db_path, cfg.features_dir)
        bank.seed_from_json(SEED_FILE)
        phrases = bank.list_phrases("classroom")[:6]
        for p, word in zip(phrases, vocab):
            for _ in range(3):
                bank.add_template(p.id, "voice", voice_features(synth_word(word, rng)))
        if require_arm:
            for _ in range(3):
                bank.add_template(None, "gesture", synth_gesture("long_blink", rng))

        log, sim_t = [], [0.0]
        engine = Engine(cfg, bank, display=log.append, player=lambda path: 0.0,
                        clock=lambda: sim_t[0])
        day = datetime.now().replace(minute=5, second=0, microsecond=0)
        mode_changes, false_triggers, background = [], 0, 0
        cue_ok = {h: [0, 0] for h in range(9, 17)}        # hour -> [succeeded, tried]
        for hour in range(9, 17):
            f = fatigue_at(hour)
            ts = day.replace(hour=hour).isoformat(timespec="seconds")
            sim_t[0] = hour * 3600.0
            for _ in range(8):
                k = int(rng.integers(0, len(phrases)))
                sim_t[0] += 60.0
                if require_arm:
                    engine.handle_cue("gesture", synth_gesture("long_blink", rng), ts=ts)
                    sim_t[0] += 1.0
                ok = False
                for attempt in range(2):                  # retry once if not understood
                    sim_t[0] += 3.0 if attempt else 0.5
                    r = engine.handle_cue("voice", voice_features(synth_word(vocab[k], rng, **f)), ts=ts)
                    if r.accepted:
                        ok = True
                        break
                cue_ok[hour][0] += ok
                cue_ok[hour][1] += 1
            for k in (6, 7, 6, 7, 6, 7):                  # unrelated classroom talk
                sim_t[0] += 60.0
                r = engine.handle_cue("voice", voice_features(synth_word(vocab[k], rng, **f)), ts=ts)
                background += 1
                false_triggers += int(r.accepted)
            for msg in log:
                if msg.strip().startswith("[mode"):
                    mode_changes.append(f"   {hour:02d}:00  {msg.strip()}")
            log.clear()
        rows = hourly_summary(bank.events_since(day.replace(hour=0).isoformat()))
        bank.close()
    return {"rows": rows, "mode_changes": mode_changes, "false_triggers": false_triggers,
            "background": background, "cue_ok": cue_ok}


def cmd_demo(args):
    from .fatigue import format_report
    from .matcher import TemplateMatcher
    from .simulate import synth_gesture

    print("=" * 72)
    print("ENCORE DEMO  (synthetic signals, not patient data)")
    print("=" * 72)
    print("\n1) 9 AM: six classroom phrases each get a short personal voice cue (3 examples).")
    print("2) A school day. Speech slows x1.40 and loses contrast by 4 PM (case data).")
    print("   Each hour: 8 cue attempts (retrying once if not understood, as people do)")
    print("   plus 6 unrelated utterances of classroom talk that should NOT trigger anything.\n")

    home = _demo_day(args.seed, require_arm=False)
    print(format_report(home["rows"]))
    print("   accepted = understood on the first try")
    print("\n   Mode advice during the day:")
    print("\n".join(home["mode_changes"]) or "   (no change)")

    print("\n3) The tradeoff, and the fix. Matching sensitive enough to work at 4 PM also")
    print("   lets some unrelated talk through. Classroom mode adds an ARM cue (a long")
    print("   deliberate blink): voice cues only fire for a few seconds after it, so talk")
    print("   alone can't trigger a phrase.\n")
    cls = _demo_day(args.seed, require_arm=True)

    def pct(stats, hour):
        ok, n = stats["cue_ok"][hour]
        return f"{ok / n:.0%}" if n else "-"

    print(f"   {'':<28}{'home mode':>12}{'classroom mode':>17}")
    print(f"   {'cue works by 9 AM (w/ retry)':<28}{pct(home, 9):>12}{pct(cls, 9):>17}")
    print(f"   {'cue works by 4 PM (w/ retry)':<28}{pct(home, 16):>12}{pct(cls, 16):>17}")
    print(f"   {'talk wrongly triggering':<28}"
          f"{home['false_triggers']:>6} of {home['background']:<3}"
          f"{cls['false_triggers']:>9} of {cls['background']:<3}")

    print("\n4) Gestures use the same engine. Four head/face cues, 3 examples each.")
    rng = np.random.default_rng(args.seed)
    names = ["tilt_left", "tilt_right", "nod", "brow_raise"]

    def score(m, strength, n=10):
        ok = sum(1 for g in names for _ in range(n)
                 if (r := m.match(synth_gesture(g, rng, strength=strength))).accepted
                 and r.label == g)
        return ok / (len(names) * n)

    m = TemplateMatcher()
    for g in names:
        for _ in range(3):
            m.add(g, synth_gesture(g, rng))
    print(f"   Recorded today, full movement:            {score(m, 1.0):.0%} recognized")
    print(f"   Later, movement shrinks to 40% strength:  {score(m, 0.4):.0%} recognized")
    m.clear()
    for g in names:
        for _ in range(3):
            m.add(g, synth_gesture(g, rng, strength=0.4))
    print(f"   After re-recording cues (~30 seconds):    {score(m, 0.4):.0%} recognized")

    print("\nTakeaway: the cue adapts to the person, not the other way round.")
    print("Results vary by seed (try --seed 1..8). Test real recordings: `encore evaluate`.")


# ---- parser ----------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(prog="encore", description="Personal cue-to-voice engine for ALS.")
    ap.add_argument("--data", help="data folder (default ./encore_data or $ENCORE_HOME)")
    sp = ap.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("init", help="create the phrase bank")
    s.add_argument("--seed", action="store_true", help="add starter phrases from the case")
    s.set_defaults(func=cmd_init)

    s = sp.add_parser("phrases", help="list phrases")
    s.add_argument("--deck")
    s.set_defaults(func=cmd_phrases)

    s = sp.add_parser("add-phrase", help="add a phrase")
    s.add_argument("text")
    s.add_argument("--deck", default="general")
    s.add_argument("--action", help="e.g. notify:checkin, notify:callme, notify:help")
    s.set_defaults(func=cmd_add_phrase)

    s = sp.add_parser("delete-phrase")
    s.add_argument("id", type=int)
    s.set_defaults(func=cmd_delete_phrase)

    s = sp.add_parser("record-phrase", help="record a phrase in your own voice")
    s.add_argument("id", type=int)
    s.add_argument("--seconds", type=float, default=6.0)
    s.add_argument("--from-file", help="use an existing wav (e.g. an old recording)")
    s.set_defaults(func=cmd_record_phrase)

    for name, func, extra in (("train-voice", cmd_train_voice, "seconds"),
                              ("train-gesture", cmd_train_gesture, "camera")):
        s = sp.add_parser(name, help=f"teach a {name.split('-')[1]} cue for a phrase (or 'arm')")
        s.add_argument("id", help="phrase id, or 'arm' for the arming cue")
        s.add_argument("--examples", type=int, default=3)
        s.add_argument("--replace", action="store_true", help="retire old examples first")
        if extra == "seconds":
            s.add_argument("--seconds", type=float, default=2.5)
        else:
            s.add_argument("--camera", type=int, default=0)
        s.set_defaults(func=func)

    s = sp.add_parser("run", help="start listening for cues")
    s.add_argument("--no-voice", dest="voice", action="store_false")
    s.add_argument("--gesture", action="store_true", help="also watch the webcam")
    s.add_argument("--camera", type=int, default=0)
    s.add_argument("--require-arm", action="store_true", help="cues fire only after the arm cue")
    s.set_defaults(func=cmd_run)

    s = sp.add_parser("say", help="speak one phrase now")
    s.add_argument("id", type=int)
    s.add_argument("--no-action", action="store_true", help="don't send its notification")
    s.set_defaults(func=cmd_say)

    s = sp.add_parser("report", help="hour-by-hour voice cue summary")
    s.add_argument("--days", type=int, default=7)
    s.add_argument("--csv")
    s.set_defaults(func=cmd_report)

    s = sp.add_parser("notify-setup", help="set the ntfy topic for check-in/help phrases")
    s.add_argument("topic")
    s.add_argument("--name", default="Encore user")
    s.add_argument("--test", action="store_true")
    s.set_defaults(func=cmd_notify_setup)

    s = sp.add_parser("clone-consent", help="record the voice owner's consent")
    s.add_argument("name")
    s.set_defaults(func=cmd_clone_consent)

    s = sp.add_parser("clone-create", help="create an ElevenLabs clone from recorded phrases")
    s.add_argument("--name", default="Encore voice")
    s.set_defaults(func=cmd_clone_create)

    s = sp.add_parser("clone-say", help="say new text in the cloned voice")
    s.add_argument("text")
    s.set_defaults(func=cmd_clone_say)

    s = sp.add_parser("evaluate", help="accuracy on a CSV manifest of real recordings")
    s.add_argument("manifest")
    s.add_argument("--compare", nargs=2, metavar=("FRESH", "TIRED"),
                   help="e.g. --compare normal slurred: fresh cues vs tired voice vs re-recorded")
    s.set_defaults(func=cmd_evaluate)

    s = sp.add_parser("collect", help="guided recording session to test on real voices")
    s.add_argument("--speaker", required=True, help="who is recording (e.g. their first name)")
    s.add_argument("--words", default="recap,stir,water,okay,help",
                   help="comma-separated cue words")
    s.add_argument("--reps", type=int, default=3, help="takes per word per voice")
    s.add_argument("--seconds", type=float, default=2.5)
    s.add_argument("--out", default="team_recordings")
    s.set_defaults(func=cmd_collect)

    s = sp.add_parser("demo", help="run the synthetic end-to-end demo (no mic needed)")
    s.add_argument("--seed", type=int, default=4)
    s.set_defaults(func=cmd_demo)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
