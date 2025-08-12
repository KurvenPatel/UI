from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Dict

import os
import sys
# Ensure /workspace/src is on sys.path
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import cv2
import numpy as np

from tennis_coach.pose import PoseEstimator
from tennis_coach.smoothing import LandmarkSmoother, VelocityEstimator
from tennis_coach.swing import SwingDetector, SwingDetectorConfig, heuristic_stroke_type
from tennis_coach.rules import RuleEngine, FeedbackSelector
from tennis_coach.overlay import OverlayRenderer
from tennis_coach.feedback import AudioCuePlayer


def to_numpy(kps) -> Dict[str, np.ndarray]:
    return {k: np.array([v.x, v.y, v.z, v.visibility], dtype=np.float32) for k, v in kps.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="camera index (e.g., 0) or video file path")
    parser.add_argument("--right_handed", action="store_true", default=True)
    parser.add_argument("--left_handed", action="store_true", help="override to left-handed")
    parser.add_argument("--min_wrist_speed", type=float, default=1.8)
    parser.add_argument("--show_fps", action="store_true", default=False)
    parser.add_argument("--audio", action="store_true", default=False)
    args = parser.parse_args()

    src = 0
    if args.source.isdigit():
        src = int(args.source)
    else:
        src = args.source

    right_handed = True
    if args.left_handed:
        right_handed = False

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Failed to open video source", src)
        return

    pose = PoseEstimator()
    smoother = LandmarkSmoother(alpha=0.35)
    wrist_velocity = VelocityEstimator(maxlen=5)
    cfg = SwingDetectorConfig(min_wrist_speed=args.min_wrist_speed)
    swing_detector = SwingDetector(cfg, right_handed=right_handed)
    rules = RuleEngine(right_handed=right_handed)
    selector = FeedbackSelector()
    overlay = OverlayRenderer(show_fps=args.show_fps)
    audio = AudioCuePlayer(enable_audio=args.audio)

    fps_deque = deque(maxlen=30)

    frame_idx = 0
    feedback_text = None
    impact_point = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.time()
            timestamp_ms = int(time.time() * 1000)

            res = pose.process(frame, timestamp_ms)
            if res is None:
                cv2.imshow("TennisCoach MVP", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            kps_np = to_numpy(res.landmarks)
            kps_sm = smoother.update(kps_np)

            dom_wrist_name = "right_wrist" if right_handed else "left_wrist"
            if dom_wrist_name in kps_sm:
                speed = wrist_velocity.update(kps_sm[dom_wrist_name], timestamp_ms)
            else:
                speed = 0.0

            swing = swing_detector.update(frame_idx, timestamp_ms, kps_sm, speed)
            if swing is not None:
                # Extract window from buffer tail approximation
                window = list(swing_detector.buffer_keypoints)[- (swing.end_frame_index - swing.start_frame_index + 1) :]
                stroke = heuristic_stroke_type(window, right_handed=right_handed)
                errors = rules.check_errors(window, stroke)
                chosen = selector.select(errors)
                feedback_text = chosen.message if chosen else None
                impact_frame = window[len(window)//2]
                impact_point = impact_frame.get(dom_wrist_name)
                audio.play_feedback(feedback_text or "")

            # Draw overlay
            frame = overlay.draw(
                frame,
                kps_sm,
                res.image_size,
                feedback_text=feedback_text,
                impact_point=impact_point,
                fps=None,
            )

            # FPS
            t1 = time.time()
            dt = t1 - t0
            fps = 1.0 / dt if dt > 1e-4 else 0.0
            fps_deque.append(fps)
            avg_fps = sum(fps_deque) / max(1, len(fps_deque))
            if args.show_fps:
                cv2.putText(frame, f"{avg_fps:.1f} FPS", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2)

            cv2.imshow("TennisCoach MVP", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            frame_idx += 1
    finally:
        pose.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()