from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import numpy as np


class StrokeType(Enum):
    UNKNOWN = auto()
    FOREHAND = auto()
    BACKHAND = auto()
    SERVE = auto()
    VOLLEY = auto()


@dataclass
class Landmark:
    x: float  # normalized [0, 1]
    y: float  # normalized [0, 1]
    z: float  # normalized depth (MediaPipe), can be ignored
    visibility: float


@dataclass
class PoseFrame:
    timestamp_ms: int
    landmarks: Dict[str, Landmark]
    image_size: Tuple[int, int]


@dataclass
class SwingEvent:
    impact_frame_index: int
    start_frame_index: int
    end_frame_index: int
    stroke_type: StrokeType = StrokeType.UNKNOWN


@dataclass
class DetectedError:
    code: str
    message: str
    confidence: float


@dataclass
class SwingAnalysis:
    swing: SwingEvent
    errors: List[DetectedError]
    chosen_feedback: Optional[DetectedError]


KeypointArray = Dict[str, np.ndarray]  # name -> np.array([x, y, z, vis])