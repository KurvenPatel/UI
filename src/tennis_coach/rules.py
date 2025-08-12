from __future__ import annotations

from typing import Dict, List
import numpy as np

from .types import DetectedError, StrokeType
from .utils import vec, angle_between, center_point


class RuleEngine:
    def __init__(self, right_handed: bool = True):
        self.right_handed = right_handed

    def check_errors(self, window: List[Dict[str, np.ndarray]], stroke: StrokeType) -> List[DetectedError]:
        errors: List[DetectedError] = []
        if len(window) < 6:
            return errors
        # Choose impact frame as middle
        impact_idx = len(window) // 2
        pre_idx = max(0, impact_idx - 6)
        post_idx = min(len(window) - 1, impact_idx + 6)
        impact = window[impact_idx]
        pre = window[pre_idx]
        post = window[post_idx]

        def get(name: str, fr: Dict[str, np.ndarray]):
            return fr.get(name)

        ls = get("left_shoulder", impact); rs = get("right_shoulder", impact)
        lh = get("left_hip", impact); rh = get("right_hip", impact)
        lw = get("left_wrist", impact); rw = get("right_wrist", impact)
        le = get("left_elbow", impact); re = get("right_elbow", impact)
        if not all(v is not None for v in [ls, rs, lh, rh, lw, rw, le, re]):
            return errors

        body_center_impact = center_point(ls, rs)

        # 1) Late contact
        dom_wrist = rw if self.right_handed else lw
        dom_shoulder = rs if self.right_handed else ls
        rel_x = float(dom_wrist[0] - dom_shoulder[0])
        if rel_x < 0.015:  # wrist not sufficiently in front of shoulder in x
            errors.append(DetectedError(
                code="late_contact",
                message="Meet the ball earlier — step forward.",
                confidence=min(1.0, (0.015 - rel_x) * 80.0),
            ))

        # 2) Closed stance at contact (shoulder line too rotated)
        shoulder_vec = vec(ls, rs)
        hip_vec = vec(lh, rh)
        # angle between shoulder and hip lines: large discrepancy indicates twist/closed stance
        sh_hip_angle = angle_between(shoulder_vec, hip_vec)
        if sh_hip_angle > 25.0:
            errors.append(DetectedError(
                code="closed_stance",
                message="Open your stance a bit and rotate your hips more.",
                confidence=min(1.0, (sh_hip_angle - 25.0) / 25.0),
            ))

        # 3) No weight transfer (COM shift small from pre to impact)
        pre_center = center_point(pre.get("left_hip"), pre.get("right_hip"))
        com_shift = float(np.linalg.norm(body_center_impact[:2] - pre_center[:2]))
        if com_shift < 0.01:
            errors.append(DetectedError(
                code="no_weight_transfer",
                message="Shift your weight forward when you swing.",
                confidence=min(1.0, (0.01 - com_shift) * 100.0),
            ))

        # 4) Short backswing: hand not taken back enough before impact
        pre_dom_wrist = pre.get("right_wrist") if self.right_handed else pre.get("left_wrist")
        back_distance = float(abs(pre_dom_wrist[0] - dom_wrist[0]))
        if back_distance < 0.05:
            errors.append(DetectedError(
                code="short_backswing",
                message="Take the racket further back before you swing.",
                confidence=min(1.0, (0.05 - back_distance) * 20.0),
            ))

        # 5) Racket face open/closed proxy using forearm orientation at impact
        dom_elbow = re if self.right_handed else le
        forearm_vec = vec(dom_elbow, dom_wrist)
        vertical_vec = np.array([0.0, -1.0, 0.0])  # up direction in image coords (y decreases upward)
        forearm_angle_from_vertical = angle_between(forearm_vec, vertical_vec)
        # Acceptable band ~ 20°–70° depending on stroke; use broad band
        if not (15.0 <= forearm_angle_from_vertical <= 80.0):
            errors.append(DetectedError(
                code="racket_face",
                message="Adjust racket angle — face slightly forward at contact.",
                confidence=0.6,
            ))

        # 6) Poor follow-through: wrist height change from impact to post small or downward
        post_dom_wrist = post.get("right_wrist") if self.right_handed else post.get("left_wrist")
        delta_y = float(dom_wrist[1] - post_dom_wrist[1])  # positive if wrist moves up (since y decreases up)
        if delta_y < 0.005:
            errors.append(DetectedError(
                code="poor_follow_through",
                message="Finish your swing — high follow-through.",
                confidence=min(1.0, (0.005 - delta_y) * 120.0 if delta_y < 0 else 0.7),
            ))

        return errors


class FeedbackSelector:
    def select(self, errors: List[DetectedError]) -> DetectedError | None:
        if not errors:
            return None
        # Pick highest confidence; tiebreak by a fixed priority order
        priority = {
            "late_contact": 0,
            "no_weight_transfer": 1,
            "poor_follow_through": 2,
            "closed_stance": 3,
            "short_backswing": 4,
            "racket_face": 5,
        }
        errors_sorted = sorted(errors, key=lambda e: (-e.confidence, priority.get(e.code, 99)))
        return errors_sorted[0]