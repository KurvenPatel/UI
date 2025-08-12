from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np

from ultralytics import YOLO
from .types import Landmark, PoseFrame


# COCO keypoint order for YOLO Pose models
COCO_KP_NAMES = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
]

# Subset we care about overlaps with our rule logic
INTEREST_KEYS = set([
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "nose",
])


class PoseEstimator:
    def __init__(self, model_path: Optional[str] = None):
        # Use small, fast model by default
        self.model = YOLO(model_path or "yolo11n-pose.pt")

    def process(self, frame_bgr, timestamp_ms: int) -> PoseFrame | None:
        h, w = frame_bgr.shape[:2]
        # Run inference; return keypoints for the largest person detected
        results = self.model.predict(frame_bgr, verbose=False)
        if not results or results[0].keypoints is None or len(results[0].keypoints) == 0:
            return None
        res = results[0]
        # Choose largest bbox
        if res.boxes is None or len(res.boxes) == 0:
            person_idx = 0
        else:
            areas = []
            for b in res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = b
                areas.append((x2 - x1) * (y2 - y1))
            person_idx = int(np.argmax(areas))

        # keypoints: (num_persons, K, 2) in pixels
        kps_xy = res.keypoints.xy.cpu().numpy()[person_idx]  # (K,2)
        # confidence may be present as (num_persons, K), else default to 1.0
        if hasattr(res.keypoints, "conf") and res.keypoints.conf is not None:
            kps_conf = res.keypoints.conf.cpu().numpy()[person_idx]
        else:
            kps_conf = np.ones((kps_xy.shape[0],), dtype=np.float32)

        named: Dict[str, Landmark] = {}
        for i, (x, y) in enumerate(kps_xy):
            name = COCO_KP_NAMES[i]
            if name not in INTEREST_KEYS:
                continue
            nx = float(np.clip(x / max(1, w), 0.0, 1.0))
            ny = float(np.clip(y / max(1, h), 0.0, 1.0))
            vis = float(kps_conf[i]) if i < len(kps_conf) else 1.0
            named[name] = Landmark(x=nx, y=ny, z=0.0, visibility=vis)

        if not named:
            return None
        return PoseFrame(timestamp_ms=timestamp_ms, landmarks=named, image_size=(w, h))

    def close(self):
        pass