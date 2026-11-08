"""Runtime configuration. Everything lives in one data folder (default ./encore_data)."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path
    sample_rate: int = 16000
    block_ms: int = 20

    # Recognition. Thresholds are calibrated per cue from the user's own examples;
    # these values only shape that calibration.
    margin_threshold: float = 0.10   # best cue must beat the runner-up by this fraction
    threshold_slack: float = 1.6     # accept up to slack x the spread of the user's own examples

    # Arming: when on, a cue only fires within arm_window_s after the arm cue.
    # Useful in a classroom where the user talks all day.
    require_arm: bool = False
    arm_window_s: float = 6.0

    # Ignore the microphone while Encore itself is speaking, plus this tail.
    mic_mute_tail_s: float = 0.4

    @property
    def db_path(self) -> Path:
        return self.data_dir / "encore.db"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @classmethod
    def load(cls, data_dir=None) -> "Config":
        root = Path(data_dir or os.environ.get("ENCORE_HOME", "encore_data")).resolve()
        cfg = cls(data_dir=root)
        for d in (cfg.data_dir, cfg.audio_dir, cfg.features_dir, cfg.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        return cfg
