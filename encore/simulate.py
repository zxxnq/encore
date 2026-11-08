"""Synthetic signals so the pipeline can be demonstrated and tested with no mic or camera.

THESE ARE NOT PATIENT DATA. "Words" are formant-like tone sequences. Fatigue is
modelled with two well-known features of dysarthric speech:
  - slower rate (case: 165 wpm at 9 AM -> 118 wpm at 4 PM, i.e. x1.40 longer)
  - articulatory undershoot: vowel/formant targets drift toward a neutral centre,
    shrinking the contrast between sounds
Hours between 9 AM and 4 PM are interpolated linearly (an assumption; the case
only reports the two endpoints).
"""

import numpy as np

SR = 16000
CENTER = np.array([500.0, 1500.0, 2500.0])


def make_vocabulary(n_words: int, seed: int = 7) -> list:
    rng = np.random.default_rng(seed)
    vocab = []
    for _ in range(n_words):
        n_ph = int(rng.integers(3, 5))
        f1 = rng.uniform(300, 800, n_ph)
        f2 = rng.uniform(900, 2300, n_ph)
        f3 = rng.uniform(2300, 3100, n_ph)
        vocab.append(np.stack([f1, f2, f3], axis=1))
    return vocab


def synth_word(targets: np.ndarray, rng, rate: float = 1.0, undershoot: float = 0.0,
               base_dur: float = 0.55, noise_db: float = -35.0, jitter: float = 0.04) -> np.ndarray:
    t_targets = CENTER + (targets - CENTER) * (1.0 - undershoot)
    t_targets = t_targets * (1 + rng.normal(0, jitter, t_targets.shape))   # natural variation
    n = int(SR * base_dur * rate * rng.uniform(0.93, 1.07))
    pos = np.linspace(0, len(t_targets) - 1, n)
    traj = np.stack([np.interp(pos, np.arange(len(t_targets)), t_targets[:, k]) for k in range(3)], 1)
    phase = 2 * np.pi * np.cumsum(traj, axis=0) / SR
    sig = np.sin(phase[:, 0]) + 0.6 * np.sin(phase[:, 1]) + 0.3 * np.sin(phase[:, 2])
    env = np.minimum(1.0, np.minimum(np.arange(n), n - np.arange(n)) / (0.04 * SR))
    sig = sig * env * 0.3 * (1.0 - 0.4 * undershoot)        # fatigued voice is quieter too
    sig = sig + rng.normal(0, 10 ** (noise_db / 20), n)
    return sig.astype(np.float32)


def fatigue_at(hour: float) -> dict:
    """Case-anchored fatigue model: 9 AM fresh, 4 PM tired."""
    frac = float(np.clip((hour - 9.0) / 7.0, 0.0, 1.0))
    return {"rate": 1.0 + 0.40 * frac, "undershoot": 0.45 * frac}


# --- gestures -----------------------------------------------------------------
GESTURES = {
    "tilt_left":  np.array([0, 0, -1, 0, 0, 0], dtype=np.float32),
    "tilt_right": np.array([0, 0, 1, 0, 0, 0], dtype=np.float32),
    "nod":        np.array([0, 1, 0, 0, 0, 0], dtype=np.float32),
    "brow_raise": np.array([0, 0, 0, 0, 1, 0], dtype=np.float32),
    "long_blink": np.array([0, 0, 0, 0, 0, -1], dtype=np.float32),
}


def synth_gesture(name: str, rng, strength: float = 1.0, fps: int = 30,
                  dur: float = 0.8, noise: float = 0.25) -> np.ndarray:
    """A normalised-feature trajectory: rise, hold, return, with sensor noise.
    strength < 1 models a smaller movement as muscles weaken."""
    n = int(fps * dur * rng.uniform(0.85, 1.15))
    shape = np.sin(np.linspace(0, np.pi, n)) ** 0.7
    amp = 5.0 * strength * rng.uniform(0.85, 1.15)
    traj = shape[:, None] * GESTURES[name][None, :] * amp
    return (traj + rng.normal(0, noise, traj.shape)).astype(np.float32)
