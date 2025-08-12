from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


# Lightweight set of connections for drawing (subset similar to MediaPipe)
CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


class OverlayRenderer:
    def __init__(self, show_fps: bool = False):
        self.show_fps = show_fps

    def _to_px(self, pt: np.ndarray, size: Tuple[int, int]) -> Tuple[int, int]:
        w, h = size
        x = int(pt[0] * w)
        y = int(pt[1] * h)
        return x, y

    def draw(self,
             frame_bgr,
             keypoints: Dict[str, np.ndarray],
             image_size: Tuple[int, int],
             feedback_text: Optional[str] = None,
             impact_point: Optional[np.ndarray] = None,
             fps: Optional[float] = None):
        # Skeleton
        for a, b in CONNECTIONS:
            if a in keypoints and b in keypoints and keypoints[a][3] > 0.3 and keypoints[b][3] > 0.3:
                pa = self._to_px(keypoints[a], image_size)
                pb = self._to_px(keypoints[b], image_size)
                cv2.line(frame_bgr, pa, pb, (0, 255, 0), 2)
        for name, pt in keypoints.items():
            if pt[3] > 0.3:
                x, y = self._to_px(pt, image_size)
                cv2.circle(frame_bgr, (x, y), 3, (0, 165, 255), -1)

        if impact_point is not None:
            x, y = self._to_px(impact_point, image_size)
            cv2.circle(frame_bgr, (x, y), 10, (0, 0, 255), 2)

        if feedback_text:
            cv2.rectangle(frame_bgr, (10, 10), (10 + 620, 60), (0, 0, 0), -1)
            cv2.putText(frame_bgr, feedback_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if self.show_fps and fps is not None:
            cv2.putText(frame_bgr, f"{fps:.1f} FPS", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2)

        return frame_bgr