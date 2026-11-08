"""The phrase bank: a small SQLite database of phrases, cue templates and usage events.

Tables
  phrases    one row per phrase the user wants to say (text, deck, recording, optional action)
  templates  cue examples (voice or gesture) linked to a phrase; old ones are deactivated,
             not deleted, when the user re-records a cue, so the history shows how cues changed
  events     every recognition attempt (accepted or not), used for adaptation and reports
  settings   key/value store (notification topic, gesture baseline, consent records, ...)
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS phrases (
    id          INTEGER PRIMARY KEY,
    text        TEXT NOT NULL,
    deck        TEXT NOT NULL DEFAULT 'general',
    audio_path  TEXT,
    recorded_at TEXT,
    quality     TEXT,
    action      TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS templates (
    id           INTEGER PRIMARY KEY,
    phrase_id    INTEGER,              -- NULL means the special arm cue
    modality     TEXT NOT NULL CHECK (modality IN ('voice', 'gesture')),
    feature_path TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    active       INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (phrase_id) REFERENCES phrases(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS events (
    id             INTEGER PRIMARY KEY,
    ts             TEXT NOT NULL,
    modality       TEXT NOT NULL,
    phrase_id      INTEGER,
    accepted       INTEGER NOT NULL,
    confidence     REAL,
    distance       REAL,
    duration_ratio REAL,
    reason         TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Phrase:
    id: int
    text: str
    deck: str
    audio_path: Optional[str]
    recorded_at: Optional[str]
    quality: Optional[str]
    action: Optional[str]

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_path) and Path(self.audio_path).exists()


class PhraseBank:
    def __init__(self, db_path, features_dir):
        self.db_path = Path(db_path)
        self.features_dir = Path(features_dir)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ---- phrases -------------------------------------------------------
    def add_phrase(self, text: str, deck: str = "general", action: Optional[str] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO phrases (text, deck, action, created_at) VALUES (?, ?, ?, ?)",
            (text.strip(), deck.strip().lower(), action, now_iso()),
        )
        self.conn.commit()
        return cur.lastrowid

    def set_audio(self, phrase_id: int, audio_path, quality: Optional[dict] = None,
                  recorded_at: Optional[str] = None):
        self.conn.execute(
            "UPDATE phrases SET audio_path = ?, recorded_at = ?, quality = ? WHERE id = ?",
            (str(audio_path), recorded_at or now_iso(),
             json.dumps(quality) if quality else None, phrase_id),
        )
        self.conn.commit()

    def get_phrase(self, phrase_id: int) -> Optional[Phrase]:
        row = self.conn.execute("SELECT * FROM phrases WHERE id = ?", (phrase_id,)).fetchone()
        return self._phrase(row) if row else None

    def list_phrases(self, deck: Optional[str] = None) -> list:
        if deck:
            rows = self.conn.execute(
                "SELECT * FROM phrases WHERE deck = ? ORDER BY id", (deck.lower(),)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM phrases ORDER BY deck, id").fetchall()
        return [self._phrase(r) for r in rows]

    def decks(self) -> list:
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT deck FROM phrases ORDER BY deck").fetchall()]

    def delete_phrase(self, phrase_id: int):
        self.conn.execute("DELETE FROM phrases WHERE id = ?", (phrase_id,))
        self.conn.commit()

    @staticmethod
    def _phrase(row) -> Phrase:
        return Phrase(row["id"], row["text"], row["deck"], row["audio_path"],
                      row["recorded_at"], row["quality"], row["action"])

    def seed_from_json(self, path) -> int:
        """Load starter phrases. Skips any phrase text that already exists."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        existing = {p.text for p in self.list_phrases()}
        added = 0
        for deck, items in data["decks"].items():
            for item in items:
                text = item if isinstance(item, str) else item["text"]
                action = None if isinstance(item, str) else item.get("action")
                if text not in existing:
                    self.add_phrase(text, deck, action)
                    added += 1
        return added

    # ---- cue templates ---------------------------------------------------
    def add_template(self, phrase_id: Optional[int], modality: str, features: np.ndarray) -> int:
        fpath = self.features_dir / f"{modality}_{phrase_id or 'arm'}_{uuid.uuid4().hex[:10]}.npy"
        np.save(fpath, np.asarray(features, dtype=np.float32))
        cur = self.conn.execute(
            "INSERT INTO templates (phrase_id, modality, feature_path, created_at) "
            "VALUES (?, ?, ?, ?)", (phrase_id, modality, str(fpath), now_iso()))
        self.conn.commit()
        return cur.lastrowid

    def deactivate_templates(self, phrase_id: Optional[int], modality: str) -> int:
        """Retire a cue's examples (kept for history) before re-recording it."""
        if phrase_id is None:
            cur = self.conn.execute(
                "UPDATE templates SET active = 0 WHERE phrase_id IS NULL AND modality = ?",
                (modality,))
        else:
            cur = self.conn.execute(
                "UPDATE templates SET active = 0 WHERE phrase_id = ? AND modality = ?",
                (phrase_id, modality))
        self.conn.commit()
        return cur.rowcount

    def active_templates(self, modality: str) -> list:
        """Returns [(phrase_id or None, features), ...] for active cues of one modality."""
        rows = self.conn.execute(
            "SELECT phrase_id, feature_path FROM templates WHERE active = 1 AND modality = ?",
            (modality,)).fetchall()
        out = []
        for r in rows:
            p = Path(r["feature_path"])
            if p.exists():
                out.append((r["phrase_id"], np.load(p)))
        return out

    def template_counts(self) -> dict:
        """{phrase_id: {'voice': n, 'gesture': n}} for active templates."""
        counts = {}
        for r in self.conn.execute(
                "SELECT phrase_id, modality, COUNT(*) AS n FROM templates "
                "WHERE active = 1 GROUP BY phrase_id, modality"):
            counts.setdefault(r["phrase_id"], {})[r["modality"]] = r["n"]
        return counts

    # ---- events ------------------------------------------------------------
    def log_event(self, modality: str, phrase_id: Optional[int], accepted: bool,
                  confidence: float, distance: float, duration_ratio: Optional[float],
                  reason: str = "", ts: Optional[str] = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO events (ts, modality, phrase_id, accepted, confidence, distance, "
            "duration_ratio, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts or now_iso(), modality, phrase_id, int(bool(accepted)), float(confidence),
             float(distance) if np.isfinite(distance) else None,
             float(duration_ratio) if duration_ratio else None, reason))
        self.conn.commit()
        return cur.lastrowid

    def mark_missed(self, event_id: int):
        """Reclassify a background rejection as a missed cue (the user retried)."""
        self.conn.execute(
            "UPDATE events SET reason = 'missed: ' || substr(reason, 13) "
            "WHERE id = ? AND reason LIKE 'background: %'", (event_id,))
        self.conn.commit()

    def events_since(self, since_iso: str) -> list:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM events WHERE ts >= ? ORDER BY ts", (since_iso,)).fetchall()]

    # ---- settings ------------------------------------------------------------
    def get_setting(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str):
        self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
        self.conn.commit()
