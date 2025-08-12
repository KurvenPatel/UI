from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional
import numpy as np


class ExponentialMovingAverage:
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.state: Optional[np.ndarray] = None

    def update(self, x: np.ndarray) -> np.ndarray:
        if self.state is None:
            self.state = x.copy()
        else:
            self.state = self.alpha * x + (1.0 - self.alpha) * self.state
        return self.state


class LandmarkSmoother:
    def __init__(self, alpha: float = 0.4):
        self.filters: Dict[str, ExponentialMovingAverage] = {}
        self.alpha = alpha

    def update(self, keypoints: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        smoothed: Dict[str, np.ndarray] = {}
        for name, arr in keypoints.items():
            if name not in self.filters:
                self.filters[name] = ExponentialMovingAverage(self.alpha)
            smoothed[name] = self.filters[name].update(arr)
        return smoothed


class VelocityEstimator:
    def __init__(self, maxlen: int = 5):
        self.history: Deque[np.ndarray] = deque(maxlen=maxlen)
        self.time_history: Deque[float] = deque(maxlen=maxlen)

    def update(self, position: np.ndarray, timestamp_ms: int) -> float:
        # position is np.array([x, y, z, vis]) normalized coords
        self.history.append(position.copy())
        self.time_history.append(timestamp_ms / 1000.0)
        if len(self.history) < 2:
            return 0.0
        # compute instantaneous speed between last two samples in normalized units per second
        p2 = self.history[-1][:2]
        p1 = self.history[-2][:2]
        dt = max(1e-3, self.time_history[-1] - self.time_history[-2])
        speed = float(np.linalg.norm(p2 - p1) / dt)
        return speed