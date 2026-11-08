"""Optional: say new sentences in the user's own voice with ElevenLabs.

Every phrase the user records for Encore is also clean voice data. With consent,
those recordings can create an instant voice clone, so a sentence nobody
pre-recorded can still come out in his voice. Recorded phrases always play the
real recording; the clone is only for things not in the bank.

Requires: pip install requests, and ELEVENLABS_API_KEY in the environment.
Endpoints follow the public ElevenLabs REST API (xi-api-key header); check the
current docs at elevenlabs.io/docs if a call fails, as model names change.
"""

import json
import os
from pathlib import Path

import numpy as np

from .audio_io import write_wav
from .db import now_iso

API = "https://api.elevenlabs.io/v1"


class ElevenLabsVoice:
    def __init__(self, bank, api_key=None, model_id: str = "eleven_multilingual_v2"):
        self.bank = bank
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.model_id = model_id

    @property
    def voice_id(self):
        return self.bank.get_setting("elevenlabs_voice_id")

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.voice_id and self.has_consent())

    def has_consent(self) -> bool:
        return bool(self.bank.get_setting("voice_clone_consent"))

    def record_consent(self, full_name: str):
        """The voice owner confirms, in their own name, that a clone may be made."""
        self.bank.set_setting("voice_clone_consent",
                              json.dumps({"name": full_name, "at": now_iso()}))

    def _requests(self):
        try:
            import requests
            return requests
        except ImportError as e:
            raise RuntimeError("pip install requests to use voice cloning") from e

    def create_clone(self, name: str, wav_paths: list) -> str:
        if not self.has_consent():
            raise PermissionError("Record the voice owner's consent first (encore clone-consent).")
        if not self.api_key:
            raise RuntimeError("Set ELEVENLABS_API_KEY first.")
        requests = self._requests()
        handles = [open(p, "rb") for p in wav_paths]
        try:
            files = [("files", (Path(p).name, h, "audio/wav")) for p, h in zip(wav_paths, handles)]
            r = requests.post(f"{API}/voices/add", headers={"xi-api-key": self.api_key},
                              data={"name": name, "description": "Encore personal voice"},
                              files=files, timeout=120)
        finally:
            for h in handles:
                h.close()
        r.raise_for_status()
        voice_id = r.json()["voice_id"]
        self.bank.set_setting("elevenlabs_voice_id", voice_id)
        return voice_id

    def synthesize(self, text: str, out_wav) -> Path:
        if not self.available:
            raise RuntimeError("Voice clone not set up (needs API key, consent and a clone).")
        requests = self._requests()
        r = requests.post(f"{API}/text-to-speech/{self.voice_id}",
                          params={"output_format": "pcm_16000"},
                          headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
                          json={"text": text, "model_id": self.model_id}, timeout=60)
        r.raise_for_status()
        pcm = np.frombuffer(r.content, dtype="<i2").astype(np.float32) / 32768.0
        write_wav(out_wav, pcm, 16000)
        return Path(out_wav)
