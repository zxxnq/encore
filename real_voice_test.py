"""
real_voice_test.py  -  test Encore on your team's real voices.

This file is separate on purpose: it only USES the existing encore code and never
changes it, so the working project and its synthetic demo stay as a backup.
Recordings go into their own folder (real_voice_data/) and nothing touches encore_data/.

Put this file in the same folder as README.md (next to the inner "encore" folder), then:

    python real_voice_test.py record --speaker aisha     # each teammate, about 3 minutes
    python real_voice_test.py results                     # once everyone is done
    python real_voice_test.py results --why               # also explain the misses

On a Mac use python3 instead of python.
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from encore.audio_io import beep, quality_check, read_wav, record_seconds, trim_silence, write_wav
    from encore.features import voice_features
    from encore.matcher import TemplateMatcher
except ImportError:
    sys.exit("Can't find the encore code. Put real_voice_test.py in the folder that contains "
             "README.md and the inner 'encore' folder, then run it from there.")

DATA = Path("real_voice_data")
MANIFEST = DATA / "manifest.csv"
SAMPLE_RATE = 16000
VOICES = [
    ("normal", "Say it in your normal voice, clearly."),
    ("tired", "Now say it like you're exhausted: slowly, mouth barely moving, dragging the "
              "vowels. Try to sound tired the SAME way every time."),
]


# ---------------------------------------------------------------- recording
def record(args):
    DATA.mkdir(exist_ok=True)
    speaker = args.speaker.strip().lower().replace(" ", "_")
    words = [w.strip() for w in args.words.split(",") if w.strip()]
    total = len(words) * args.takes * len(VOICES)

    already = [r for r in read_manifest() if r["speaker"] == speaker]
    if already:
        if input(f"{speaker} already has {len(already)} recordings. Record again and "
                 "replace them? [y/N] ").strip().lower() != "y":
            return
        keep = [r for r in read_manifest() if r["speaker"] != speaker]
        write_manifest(keep)

    print(f"\nRecording {speaker}: {len(words)} words x {args.takes} takes x 2 voices "
          f"= {total} short clips (about {total * 6 // 60 + 1} minutes).")
    print("Sit about an arm's length from the mic, in the quietest spot you can find.\n")

    rows, n = [], 0
    for voice, instruction in VOICES:
        print(f"\n--- {voice.upper()} VOICE ---\n{instruction}\n")
        for word in words:
            for take in range(1, args.takes + 1):
                n += 1
                while True:
                    input(f'[{n}/{total}] Say "{word}" ({voice} voice, take {take}). '
                          "Press Enter, then speak after the beep...")
                    beep(SAMPLE_RATE)
                    audio = trim_silence(record_seconds(args.seconds, SAMPLE_RATE), SAMPLE_RATE)
                    problems = quality_check(audio, SAMPLE_RATE)["warnings"]
                    if problems and input(f"   ! {'; '.join(problems)}. Redo this one? [Y/n] "
                                          ).strip().lower() != "n":
                        continue
                    break
                name = f"{speaker}_{voice}_{word}_{take}.wav"
                write_wav(DATA / name, audio, SAMPLE_RATE)
                rows.append({"file": name, "word": word, "speaker": speaker, "voice": voice})

    write_manifest(read_manifest() + rows)
    print(f"\nDone, {len(rows)} clips saved for {speaker}. Next teammate: run the same command "
          "with their name.\nWhen everyone is done: python real_voice_test.py results")


def read_manifest():
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(rows):
    DATA.mkdir(exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "word", "speaker", "voice"])
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- results
def features_for(rows):
    out = {}
    for r in rows:
        path = DATA / r["file"]
        if path.exists():
            out[r["file"]] = voice_features(trim_silence(read_wav(path, SAMPLE_RATE)), SAMPLE_RATE)
    return out


def why(result):
    """Plain-language reason a clip played nothing."""
    reason = result.reason or ""
    if reason.startswith("ambiguous"):
        return "mixed up two words"
    if reason.startswith("length"):
        return "much longer/shorter"
    return "too different"


def score(pairs):
    """pairs: list of (true_word, MatchResult). Returns counts."""
    n = len(pairs)
    correct = sum(1 for w, r in pairs if r.accepted and r.label == w)
    rejected = sum(1 for _, r in pairs if not r.accepted)
    wrong = sum(1 for w, r in pairs if r.accepted and r.label != w)
    reasons = defaultdict(int)
    confused = defaultdict(int)
    top_guess_right = 0
    for w, r in pairs:
        if not r.accepted:
            reasons[why(r)] += 1
            top_guess_right += int(r.label == w)
            if why(r) == "mixed up two words":
                confused[" / ".join(sorted([w, r.runner_up if r.label == w else r.label]))] += 1
    return {"n": n, "correct": correct, "rejected": rejected, "wrong": wrong,
            "reasons": reasons, "confused": confused, "top_guess_right": top_guess_right}


def leave_one_out(rows, feats):
    """Each clip is matched against the same person's OTHER clips of the same voice."""
    pairs = []
    for i, r in enumerate(rows):
        m = TemplateMatcher()
        for j, o in enumerate(rows):
            if j != i:
                m.add(o["word"], feats[o["file"]])
        if r["word"] in m.templates:
            pairs.append((r["word"], m.match(feats[r["file"]])))
    return pairs


def taught_then_used(teach_rows, use_rows, feats):
    m = TemplateMatcher()
    for r in teach_rows:
        m.add(r["word"], feats[r["file"]])
    return [(r["word"], m.match(feats[r["file"]])) for r in use_rows if r["word"] in m.templates]


def both_voices(fresh_rows, tired_rows, feats):
    """Keep the morning examples AND add the tired ones (each tired clip left out in turn)."""
    pairs = []
    for i, r in enumerate(tired_rows):
        m = TemplateMatcher()
        for o in fresh_rows:
            m.add(o["word"], feats[o["file"]])
        for j, o in enumerate(tired_rows):
            if j != i:
                m.add(o["word"], feats[o["file"]])
        if r["word"] in m.templates:
            pairs.append((r["word"], m.match(feats[r["file"]])))
    return pairs


def slowdown(fresh_rows, tired_rows, feats):
    """How much longer the tired takes are than the normal ones (median, per word)."""
    import statistics
    ratios = []
    for word in {r["word"] for r in fresh_rows}:
        f = [len(feats[r["file"]]) for r in fresh_rows if r["word"] == word]
        t = [len(feats[r["file"]]) for r in tired_rows if r["word"] == word]
        if f and t:
            ratios.append(statistics.median(t) / statistics.median(f))
    return statistics.median(ratios) if ratios else None


def results(args):
    rows = [r for r in read_manifest() if (DATA / r["file"]).exists()]
    if not rows:
        sys.exit("No recordings yet. Run: python real_voice_test.py record --speaker <name>")
    feats = features_for(rows)
    by_speaker = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_speaker[r["speaker"]][r["voice"]].append(r)

    scenarios = [
        ("Fresh voice, fresh cues", "Does it work at all? (Alex at 9 AM)"),
        ("Tired voice, fresh cues", "Cues taught in the morning, used tired (Alex at 4 PM)"),
        ("Tired voice, re-taught cues", "After re-recording the cues with the tired voice"),
        ("Tired voice, both voices taught", "Morning examples kept AND tired ones added"),
    ]
    totals = {s: {"n": 0, "correct": 0, "rejected": 0, "wrong": 0, "top_guess_right": 0,
                  "reasons": defaultdict(int), "confused": defaultdict(int)} for s, _ in scenarios}
    per_person, slowdowns = [], []
    for speaker, voices in sorted(by_speaker.items()):
        normal, tired = voices.get("normal", []), voices.get("tired", [])
        runs = {
            scenarios[0][0]: leave_one_out(normal, feats),
            scenarios[1][0]: taught_then_used(normal, tired, feats),
            scenarios[2][0]: leave_one_out(tired, feats),
            scenarios[3][0]: both_voices(normal, tired, feats),
        }
        sd = slowdown(normal, tired, feats)
        if sd:
            slowdowns.append((speaker, sd))
        line = [speaker]
        for name, _ in scenarios:
            s = score(runs[name])
            for k in ("n", "correct", "rejected", "wrong", "top_guess_right"):
                totals[name][k] += s[k]
            for k in ("reasons", "confused"):
                for key, v in s[k].items():
                    totals[name][k][key] += v
            line.append(f"{s['correct'] / s['n']:.0%}" if s["n"] else "-")
        per_person.append(line)

    report = []
    report.append(f"ENCORE REAL-VOICE TEST  ({datetime.now():%Y-%m-%d %H:%M})")
    report.append(f"{len(rows)} clips from {len(by_speaker)} people: {', '.join(sorted(by_speaker))}")
    report.append("")
    report.append(f"{'scenario':<34}{'clips':>6}{'right':>8}{'no phrase':>11}{'WRONG phrase':>14}")
    for name, meaning in scenarios:
        t = totals[name]
        if not t["n"]:
            report.append(f"{name:<34}{'(not enough recordings)':>39}")
            continue
        report.append(f"{name:<34}{t['n']:>6}{t['correct'] / t['n']:>8.0%}"
                      f"{t['rejected'] / t['n']:>11.0%}{t['wrong'] / t['n']:>14.0%}")
        report.append(f"   {meaning}")
    report.append("")
    report.append("right = correct phrase played; no phrase = nothing played, user can retry;")
    report.append("WRONG phrase = the worst outcome, a different phrase played")
    report.append("")
    if args.why:
        report.append("WHY clips played nothing:")
        for name, _ in scenarios:
            t = totals[name]
            if t["rejected"]:
                parts = ", ".join(f"{k} {v}" for k, v in sorted(t["reasons"].items(),
                                                               key=lambda kv: -kv[1]))
                report.append(f"   {name:<32}{parts}")
                report.append(f"   {'':<32}top guess was actually right in "
                              f"{t['top_guess_right']} of {t['rejected']} of these")
                if t["confused"]:
                    pairs = ", ".join(f"{k} ({v}x)" for k, v in sorted(t["confused"].items(),
                                                                     key=lambda kv: -kv[1]))
                    report.append(f"   {'':<32}words mixed up: {pairs}")
        for speaker, sd in slowdowns:
            report.append(f"   {speaker}'s tired voice was x{sd:.2f} slower than normal "
                          "(Alex: x1.40 by 4 PM)")
        report.append("")
    report.append(f"Per person (% right)   {'fresh':>6}   {'tired':>6}   {'re-taught':>9}   {'both':>6}")
    for line in per_person:
        report.append(f"   {line[0]:<19}{line[1]:>6}   {line[2]:>6}   {line[3]:>9}   {line[4]:>6}")
    report.append("")
    report.append("Note: the tired voice is imitated by healthy teammates, not real ALS speech.")

    text = "\n".join(report)
    print("\n" + text)
    out = DATA / "results.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\n(Saved to {out}. Paste this into the chat for the write-up.)")


def main():
    ap = argparse.ArgumentParser(description="Test Encore on your team's real voices.")
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("record", help="record one teammate")
    r.add_argument("--speaker", required=True, help="the person's first name")
    r.add_argument("--words", default="recap,stir,water,okay,help")
    r.add_argument("--takes", type=int, default=3)
    r.add_argument("--seconds", type=float, default=2.5)
    r.set_defaults(func=record)
    s = sp.add_parser("results", help="score everyone's recordings")
    s.add_argument("--why", action="store_true", help="explain why clips played nothing")
    s.set_defaults(func=results)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
