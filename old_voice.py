"""
old_voice.py  -  give Encore a person's voice from OLD recordings and videos.

For when someone can no longer record clean speech: use home videos, voicemails,
voice memos or recorded talks from before their voice changed. This file is separate
on purpose: it only USES the existing encore code and never changes it.

Put this file next to real_voice_test.py (beside the inner "encore" folder), then:

    python old_voice.py prepare ~/Desktop/old_clips     # clean up the old recordings
    python old_voice.py clone --name "Alex Chen"         # make the voice (needs consent + API key)
    python old_voice.py speak "Here's today's recap."    # hear any sentence in that voice
    python old_voice.py fill-phrases                     # voice every unrecorded Encore phrase

Before `clone`, set your ElevenLabs key in the terminal:
    export ELEVENLABS_API_KEY=your_key_here
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from encore.audio_io import quality_check, read_wav, trim_silence, write_wav
    from encore.config import Config
    from encore.db import PhraseBank
    from encore.voice_clone import ElevenLabsVoice
except ImportError:
    sys.exit("Can't find the encore code. Put old_voice.py in the folder that contains "
             "README.md and the inner 'encore' folder, then run it from there.")

SR = 16000
WORK = Path("old_voice_data")
CLEAN = WORK / "clean"
AUDIO_TYPES = {".wav", ".m4a", ".mp3", ".aac", ".aiff", ".aif", ".caf", ".flac",
               ".mp4", ".mov", ".m4v"}
TARGET_SECONDS = 60          # ElevenLabs instant cloning wants roughly 1-2 minutes
CHUNK_SECONDS = 30


# ------------------------------------------------------------------ helpers
def to_wav(src: Path, dst: Path) -> bool:
    """Convert any audio or video file to 16 kHz mono wav. Returns True on success."""
    if src.suffix.lower() == ".wav":
        try:
            write_wav(dst, read_wav(src, SR), SR)
            return True
        except Exception:
            pass  # unusual wav encodings: fall through to a converter
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn",
               "-ac", "1", "-ar", str(SR), str(dst)]
    elif shutil.which("afconvert"):                       # built into every Mac
        cmd = ["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", str(src), str(dst)]
    else:
        return False
    return subprocess.run(cmd, capture_output=True).returncode == 0 and dst.exists()


def get_bank():
    cfg = Config.load()
    return cfg, PhraseBank(cfg.db_path, cfg.features_dir)


def play(path: Path):
    if shutil.which("afplay"):                            # Mac
        subprocess.run(["afplay", str(path)])
        return
    try:
        from encore.audio_io import play_file
        import time
        time.sleep(play_file(path))
    except Exception as e:
        print(f"(Couldn't play audio here: {e}. The file is saved at {path}.)")


# ------------------------------------------------------------------ commands
def prepare(args):
    src_dir = Path(args.folder).expanduser()
    files = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in AUDIO_TYPES) \
        if src_dir.is_dir() else []
    if not files:
        sys.exit(f"No audio or video files found in {src_dir}. "
                 f"Supported: {', '.join(sorted(AUDIO_TYPES))}")
    if CLEAN.exists():
        shutil.rmtree(CLEAN)
    CLEAN.mkdir(parents=True)

    print(f"\nChecking {len(files)} file(s)...\n")
    kept_seconds, kept, problems = 0.0, 0, []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            raw = Path(tmp) / (f.stem + ".wav")
            if not to_wav(f, raw):
                problems.append(f.name)
                print(f"  x  {f.name}: couldn't read it (see tip below)")
                continue
            audio = trim_silence(read_wav(raw, SR), SR)
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 0:
                audio = audio * (0.89 / peak)              # even out loudness between clips
            q = quality_check(audio, SR)
            if q["duration_s"] < 1.0:
                print(f"  -  {f.name}: too short after trimming, skipped")
                continue
            if q["snr_db"] < 10:
                print(f"  -  {f.name}: too noisy (SNR {q['snr_db']} dB), skipped")
                continue
            step = CHUNK_SECONDS * SR
            for i in range(0, audio.size, step):
                piece = audio[i:i + step]
                if piece.size < SR:                         # drop tail scraps under 1 s
                    continue
                write_wav(CLEAN / f"{f.stem}_{i // step + 1}.wav", piece, SR)
                kept += 1
                kept_seconds += piece.size / SR
            note = f"  (SNR {q['snr_db']} dB" + (", " + "; ".join(q["warnings"]) if q["warnings"] else "") + ")"
            print(f"  ok {f.name}: {q['duration_s']:.0f}s usable{note}")

    print(f"\nKept {kept_seconds:.0f} seconds of voice in {kept} clip(s) -> {CLEAN}/")
    if kept_seconds < TARGET_SECONDS:
        print(f"That's under the ~{TARGET_SECONDS}s that voice cloning needs. Add more clips "
              "where this person is talking, ideally alone and in a quiet room.")
    else:
        print('Enough to try a clone. Next: python old_voice.py clone --name "Full Name"')
    print("\nImportant: listen to a few files in old_voice_data/clean/. If other people are "
          "talking in them (kids, TV), delete those files, or the clone will blend voices.")
    if problems:
        print("\nTip for files that couldn't be read: install ffmpeg (brew install ffmpeg), "
              "or open the video in QuickTime > File > Export As > Audio Only, and use that file.")


def clone(args):
    clips = sorted(CLEAN.glob("*.wav"))
    if not clips:
        sys.exit("No prepared clips. Run: python old_voice.py prepare <folder> first.")
    _, bank = get_bank()
    voice = ElevenLabsVoice(bank)
    if not voice.api_key:
        sys.exit("Set your ElevenLabs key first:  export ELEVENLABS_API_KEY=your_key_here")
    if not voice.has_consent():
        print("\nConsent: I am the owner of this voice (or legally speaking for them, with "
              "their agreement) and I agree that a synthetic copy may be made from these "
              "recordings, for their own communication only.")
        if input(f"Type the full name ({args.name}) to agree: ").strip() != args.name:
            sys.exit("Name didn't match. Nothing was uploaded.")
        voice.record_consent(args.name)
    total = sum(read_wav(c, SR).size for c in clips) / SR
    print(f"\nUploading {len(clips)} clip(s), {total:.0f}s, to ElevenLabs...")
    try:
        voice_id = voice.create_clone(args.name, [str(c) for c in clips[:25]])
    except Exception as e:
        reason = getattr(getattr(e, "response", None), "text", "")   # ElevenLabs' own explanation
        sys.exit(f"Clone failed: {e}\n{reason}\nCheck the API key, and that your ElevenLabs plan "
                 "includes instant voice cloning.")
    print(f"Voice created (id {voice_id}).")
    print('Try it: python old_voice.py speak "Hi, it\'s me."')


def speak(args):
    _, bank = get_bank()
    voice = ElevenLabsVoice(bank)
    if not voice.available:
        sys.exit("No voice yet. Run prepare and clone first (and set ELEVENLABS_API_KEY).")
    WORK.mkdir(exist_ok=True)
    out = WORK / f"said_{abs(hash(args.text)) % 10**8}.wav"
    voice.synthesize(args.text, out)
    print(f'Saying: "{args.text}"  (saved {out})')
    play(out)


def fill_phrases(args):
    cfg, bank = get_bank()
    voice = ElevenLabsVoice(bank)
    if not voice.available:
        sys.exit("No voice yet. Run prepare and clone first (and set ELEVENLABS_API_KEY).")
    todo = [p for p in bank.list_phrases() if args.overwrite or not p.has_audio]
    if not todo:
        sys.exit("Every phrase already has a recording. Use --overwrite to replace them.")
    print(f"Voicing {len(todo)} phrase(s) in the rebuilt voice...")
    for p in todo:
        out = cfg.audio_dir / f"phrase_{p.id}_oldvoice.wav"
        try:
            voice.synthesize(p.text, out)
        except Exception as e:
            print(f"  x  {p.id}: {e}")
            continue
        bank.set_audio(p.id, out, {"source": "voice rebuilt from old recordings"})
        print(f"  ok {p.id}: {p.text}")
    print("\nDone. `python -m encore run` will now speak these in the rebuilt voice.")


def main():
    ap = argparse.ArgumentParser(description="Rebuild a voice from old recordings and videos.")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("prepare", help="clean a folder of old recordings/videos")
    s.add_argument("folder")
    s.set_defaults(func=prepare)
    s = sp.add_parser("clone", help="create the voice with ElevenLabs (needs consent)")
    s.add_argument("--name", required=True, help="the voice owner's full name")
    s.set_defaults(func=clone)
    s = sp.add_parser("speak", help="say any sentence in the rebuilt voice")
    s.add_argument("text")
    s.set_defaults(func=speak)
    s = sp.add_parser("fill-phrases", help="voice every Encore phrase that has no recording")
    s.add_argument("--overwrite", action="store_true", help="also replace real recordings")
    s.set_defaults(func=fill_phrases)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
