from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

# MediaPipe Pose landmark names of interest (subset)
MP_KEYPOINTS = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

LEFT = "left"
RIGHT = "right"


def to_numpy_keypoints(landmarks: Dict[str, Tuple[float, float, float, float]]) -> Dict[str, np.ndarray]:
    return {k: np.array(v, dtype=np.float32) for k, v in landmarks.items()}


def vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return b[:3] - a[:3]


def length(v: np.ndarray) -> float:
    return float(np.linalg.norm(v[:2]))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_xy = v1[:2]
    v2_xy = v2[:2]
    denom = (np.linalg.norm(v1_xy) * np.linalg.norm(v2_xy)) + 1e-6
    cosang = float(np.clip(np.dot(v1_xy, v2_xy) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def horizontal_angle(v1: np.ndarray) -> float:
    # angle w.r.t +x axis in degrees
    v = v1[:2]
    ang = float(np.degrees(np.arctan2(v[1], v[0])))
    return ang


def center_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))