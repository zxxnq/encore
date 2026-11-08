"""Face and head gesture cues from a webcam.

Why the face: in ALS, eye movements are spared and facial/neck control usually
outlasts hand control. This user's right hand can hold a pinch for only a few
seconds and his left hand is starting to weaken, so cues that depend on the
hands would expire soonest. A head tilt, nod, brow raise or long blink can be
re-recorded as a new cue whenever the old one gets hard.

Features per frame (all scale-free, divided by the distance between the eyes):
  yaw, pitch, roll, mouth opening, brow raise, eye opening
They are expressed relative to a relaxed baseline captured at the start of a
session, so sitting position and lighting matter less. Direction is preserved
(tilting left and tilting right are different cues).
"""

import json

import numpy as np

FEATURE_NAMES = ["yaw", "pitch", "roll", "mouth", "brow", "eyes"]

# MediaPipe FaceMesh landmark indices
R_EYE_OUT, L_EYE_OUT = 33, 263
NOSE = 1
LIP_UP, LIP_DOWN = 13, 14
R_BROW, L_BROW = 105, 334
R_LID_UP, R_LID_DOWN = 159, 145
L_LID_UP, L_LID_DOWN = 386, 374

# Smallest movement per feature that counts as deliberate, so tiny baseline
# jitter can't blow up the normalisation.
MIN_SCALE = np.array([0.04, 0.04, 0.035, 0.03, 0.02, 0.012], dtype=np.float32)


def face_features(pts: np.ndarray) -> np.ndarray:
    """pts: (468+, 3) landmark array in normalised image coordinates."""
    p = np.asarray(pts, dtype=np.float32)[:, :2]
    r_eye, l_eye = p[R_EYE_OUT], p[L_EYE_OUT]
    mid = (r_eye + l_eye) / 2
    iod = float(np.linalg.norm(l_eye - r_eye)) + 1e-6
    yaw = (p[NOSE, 0] - mid[0]) / iod
    pitch = (p[NOSE, 1] - mid[1]) / iod
    roll = float(np.arctan2(l_eye[1] - r_eye[1], l_eye[0] - r_eye[0]))
    mouth = np.linalg.norm(p[LIP_UP] - p[LIP_DOWN]) / iod
    brow = (np.linalg.norm(p[R_BROW] - p[R_LID_UP]) + np.linalg.norm(p[L_BROW] - p[L_LID_UP])) / (2 * iod)
    eyes = (np.linalg.norm(p[R_LID_UP] - p[R_LID_DOWN]) + np.linalg.norm(p[L_LID_UP] - p[L_LID_DOWN])) / (2 * iod)
    return np.array([yaw, pitch, roll, mouth, brow, eyes], dtype=np.float32)


class GestureBaseline:
    """Relaxed-face reference. Normalised feature = (raw - mean) / scale."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def from_frames(cls, frames: np.ndarray) -> "GestureBaseline":
        f = np.asarray(frames, dtype=np.float32)
        return cls(f.mean(axis=0), np.maximum(f.std(axis=0) * 4.0, MIN_SCALE))

    def normalize(self, feat: np.ndarray) -> np.ndarray:
        return (np.asarray(feat, dtype=np.float32) - self.mean) / self.scale

    def to_json(self) -> str:
        return json.dumps({"mean": self.mean.tolist(), "scale": self.scale.tolist()})

    @classmethod
    def from_json(cls, s: str) -> "GestureBaseline":
        d = json.loads(s)
        return cls(np.array(d["mean"]), np.array(d["scale"]))


class FaceCueCamera:
    """Webcam + MediaPipe FaceMesh. Yields raw feature vectors (or None if no face)."""

    def __init__(self, camera_index: int = 0):
        try:
            import cv2
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError(
                "Gesture cues need opencv-python and mediapipe:\n"
                "  pip install opencv-python mediapipe\n"
                f"Original error: {e}") from e
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "This mediapipe version has no legacy 'solutions' API. "
                "Install an older release, e.g. pip install 'mediapipe<0.10.22', "
                "or port FaceCueCamera to the MediaPipe Tasks FaceLandmarker.")
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self._running = True

    def frames(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                break
            res = self.mesh.process(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                yield face_features(np.array([[q.x, q.y, q.z] for q in lm], dtype=np.float32))
            else:
                yield None

    def capture_baseline(self, seconds: float = 2.0, fps: int = 30) -> GestureBaseline:
        rows = []
        for f in self.frames():
            if f is not None:
                rows.append(f)
            if len(rows) >= int(seconds * fps):
                break
        if len(rows) < 10:
            raise RuntimeError("Could not see a face long enough to calibrate. Check lighting.")
        return GestureBaseline.from_frames(np.stack(rows))

    def close(self):
        self._running = False
        self.cap.release()
