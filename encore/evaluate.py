"""Offline evaluation: how well does personal few-shot matching work on real recordings?

Input is a CSV manifest with columns:
  path     audio file (wav)
  label    which cue/item the recording is
  speaker  (optional) recordings are only compared within the same speaker
  group    (optional) e.g. dysarthria severity or ALSFRS-R speech score

Method: leave-one-out within each speaker. Each recording is matched against
that speaker's other recordings; it counts as correct if accepted with the right
label. Grouping by severity shows whether personal templates keep working as
speech degrades. For VOC-ALS, use the task (vowel / syllable) as the label,
the participant ID as speaker and the ALSFRS-R speech item as group.
"""

import csv
from collections import defaultdict
from pathlib import Path

from .audio_io import read_wav, trim_silence
from .features import voice_features
from .matcher import TemplateMatcher


def load_manifest(path) -> list:
    base = Path(path).parent
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = Path(r["path"])
            rows.append({"path": p if p.is_absolute() else base / p, "label": r["label"],
                         "speaker": r.get("speaker") or "all", "group": r.get("group") or "all"})
    return rows


def leave_one_out(rows: list, feature_fn=None, **matcher_kwargs) -> dict:
    feature_fn = feature_fn or (lambda p: voice_features(trim_silence(read_wav(p))))
    feats = {i: feature_fn(r["path"]) for i, r in enumerate(rows)}
    by_speaker = defaultdict(list)
    for i, r in enumerate(rows):
        by_speaker[r["speaker"]].append(i)

    results = []
    for speaker, idxs in by_speaker.items():
        for i in idxs:
            m = TemplateMatcher(**matcher_kwargs)
            for j in idxs:
                if j != i:
                    m.add(rows[j]["label"], feats[j])
            if rows[i]["label"] not in m.templates:
                continue  # only one recording of this label: nothing to compare against
            res = m.match(feats[i])
            results.append({"speaker": speaker, "group": rows[i]["group"],
                            "label": rows[i]["label"], "pred": res.label,
                            "accepted": res.accepted, "confidence": res.confidence,
                            "correct": res.accepted and res.label == rows[i]["label"]})
    return summarize(results)


def summarize(results: list) -> dict:
    def stats(rs):
        n = len(rs)
        if not n:
            return {"n": 0}
        return {"n": n,
                "accuracy": round(sum(r["correct"] for r in rs) / n, 3),
                "rejected": round(sum(not r["accepted"] for r in rs) / n, 3),
                "wrong_accepts": round(sum(r["accepted"] and not r["correct"] for r in rs) / n, 3),
                "mean_confidence": round(sum(r["confidence"] for r in rs) / n, 3)}
    groups = defaultdict(list)
    for r in results:
        groups[r["group"]].append(r)
    return {"overall": stats(results), "by_group": {g: stats(rs) for g, rs in sorted(groups.items())},
            "results": results}


def format_summary(summary: dict) -> str:
    lines = ["group           n  accuracy  rejected  wrong-accept  confidence"]
    rows = [("overall", summary["overall"])] + list(summary["by_group"].items())
    for name, s in rows:
        if not s.get("n"):
            continue
        lines.append(f"{str(name):<12}{s['n']:>5}{s['accuracy']:>10.0%}{s['rejected']:>10.0%}"
                     f"{s['wrong_accepts']:>14.0%}{s['mean_confidence']:>12.2f}")
    return "\n".join(lines)


def _default_features(path):
    return voice_features(trim_silence(read_wav(path)))


def train_test(rows: list, train_group: str, test_group: str, feature_fn=None,
               **matcher_kwargs) -> dict:
    """Cues taught from one condition, tested on another, within each speaker.

    This mirrors Alex's day: cues taught with the 9 AM voice, used with the 4 PM voice.
    """
    feature_fn = feature_fn or _default_features
    by_speaker = defaultdict(lambda: {"train": [], "test": []})
    for r in rows:
        if r["group"] == train_group:
            by_speaker[r["speaker"]]["train"].append(r)
        if r["group"] == test_group:
            by_speaker[r["speaker"]]["test"].append(r)
    results = []
    for speaker, d in by_speaker.items():
        if not d["train"] or not d["test"]:
            continue
        m = TemplateMatcher(**matcher_kwargs)
        for r in d["train"]:
            m.add(r["label"], feature_fn(r["path"]))
        for r in d["test"]:
            if r["label"] not in m.templates:
                continue
            res = m.match(feature_fn(r["path"]))
            results.append({"speaker": speaker, "group": f"{train_group}->{test_group}",
                            "label": r["label"], "pred": res.label, "accepted": res.accepted,
                            "confidence": res.confidence,
                            "correct": res.accepted and res.label == r["label"]})
    return summarize(results)


def compare_conditions(rows: list, fresh: str = "normal", tired: str = "slurred",
                       feature_fn=None) -> list:
    """The three numbers that tell the Encore story, per condition pair.

    1. fresh cues, fresh voice      (does it work at all?)
    2. fresh cues, tired voice      (what happens when the voice fades?)
    3. re-recorded cues, tired voice (does re-recording rescue it?)
    """
    cache = {}
    base_fn = feature_fn or _default_features

    def fn(path):
        if path not in cache:
            cache[path] = base_fn(path)
        return cache[path]

    def subset(group):
        return [r for r in rows if r["group"] == group]

    return [
        ("cues taught fresh, used fresh", leave_one_out(subset(fresh), fn)["overall"]),
        ("cues taught fresh, used tired", train_test(rows, fresh, tired, fn)["overall"]),
        ("cues re-taught tired, used tired", leave_one_out(subset(tired), fn)["overall"]),
    ]


def format_comparison(lines: list) -> str:
    out = [f"{'scenario':<36}{'n':>5}{'accuracy':>10}{'rejected':>10}{'wrong':>8}"]
    for name, s in lines:
        if not s.get("n"):
            out.append(f"{name:<36}{0:>5}   (no matching recordings)")
            continue
        out.append(f"{name:<36}{s['n']:>5}{s['accuracy']:>10.0%}{s['rejected']:>10.0%}"
                   f"{s['wrong_accepts']:>8.0%}")
    out.append("rejected = no phrase played (user can retry); wrong = the wrong phrase played")
    return "\n".join(out)
