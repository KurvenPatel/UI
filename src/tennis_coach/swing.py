from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque
import numpy as np

from .types import StrokeType, SwingEvent


@dataclass
class SwingDetectorConfig:
    min_wrist_speed: float = 1.8  # normalized units/sec
    min_frames_between_swings: int = 10
    window_pre_frames: int = 10
    window_post_frames: int = 14


class SwingDetector:
    def __init__(self, config: SwingDetectorConfig, right_handed: bool = True):
        self.cfg = config
        self.right_handed = right_handed
        self.state = "idle"
        self.frames_since_last = 9999
        self.wrist_speeds: Deque[float] = deque(maxlen=60)
        self.buffer_keypoints: Deque[Dict[str, np.ndarray]] = deque(maxlen=200)
        self.buffer_timestamps: Deque[int] = deque(maxlen=200)
        self.last_peak_index: Optional[int] = None
        self.local_max_speed: float = 0.0

    def update(self, frame_idx: int, timestamp_ms: int, keypoints: Dict[str, np.ndarray], wrist_speed: float) -> Optional[SwingEvent]:
        self.frames_since_last += 1
        self.wrist_speeds.append(wrist_speed)
        self.buffer_keypoints.append(keypoints)
        self.buffer_timestamps.append(timestamp_ms)

        # Simple state machine: idle -> accelerating -> decelerating -> impact
        if self.state == "idle":
            if wrist_speed > self.cfg.min_wrist_speed and self.frames_since_last > self.cfg.min_frames_between_swings:
                self.state = "accelerating"
                self.local_max_speed = wrist_speed
                self.last_peak_index = frame_idx
        elif self.state == "accelerating":
            if wrist_speed > self.local_max_speed:
                self.local_max_speed = wrist_speed
                self.last_peak_index = frame_idx
            # if speed drops 30% from peak, consider impact occurred at peak
            if self.local_max_speed > 0 and wrist_speed < 0.7 * self.local_max_speed:
                impact = self.last_peak_index if self.last_peak_index is not None else frame_idx
                start = max(0, impact - self.cfg.window_pre_frames)
                end = impact + self.cfg.window_post_frames
                self.state = "idle"
                self.frames_since_last = 0
                return SwingEvent(impact_frame_index=impact, start_frame_index=start, end_frame_index=end)
        return None

    def get_window_keypoints(self, start_idx: int, end_idx: int) -> List[Dict[str, np.ndarray]]:
        # Reconstruct window keypoints from the buffer tail
        buf_len = len(self.buffer_keypoints)
        start = max(0, buf_len - (end_idx - start_idx + 1))
        return list(self.buffer_keypoints)[start:]


def heuristic_stroke_type(window: List[Dict[str, np.ndarray]], right_handed: bool = True) -> StrokeType:
    if not window:
        return StrokeType.UNKNOWN
    # Use wrist relative x to body center around impact (last third of window)
    impact_slice = window[len(window) // 2 : min(len(window), len(window) // 2 + 4)]
    xs = []
    for kp in impact_slice:
        ls = kp.get("left_shoulder"); rs = kp.get("right_shoulder")
        lw = kp.get("left_wrist"); rw = kp.get("right_wrist")
        if not (ls is not None and rs is not None and lw is not None and rw is not None):
            continue
        body_cx = 0.5 * (ls[0] + rs[0])
        dom_wrist_x = rw[0] if right_handed else lw[0]
        xs.append(dom_wrist_x - body_cx)
    if not xs:
        return StrokeType.UNKNOWN
    rel = float(np.median(xs))
    # For a right-hander: positive means wrist more to right of body center (forehand), negative likely backhand
    if right_handed:
        return StrokeType.FOREHAND if rel > 0 else StrokeType.BACKHAND
    else:
        return StrokeType.BACKHAND if rel > 0 else StrokeType.FOREHAND