# TennisCoach MVP (Python Realtime Prototype)

This is a quick MVP prototype for a vision-based tennis beginner coach. It runs on a laptop with a webcam (or a video file) and performs:

- Realtime pose detection (YOLO Pose via Ultralytics)
- Wrist-velocity swing segmentation with impact-frame detection
- Basic stroke type heuristic (forehand/backhand)
- 4–6 rule-based beginner error checks
- On-screen overlay with skeleton and short cue text

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app/run_realtime.py --source 0  # webcam index, or a video file path
```

Options:

- `--source`: camera index (e.g., 0) or path to a video file
- `--right_handed`: set if the player is right-handed (default: true)
- `--min_wrist_speed`: swing detection threshold (default: 1.8 normalized units/sec)
- `--show_fps`: overlay FPS counter (default: false)
- `--audio`: play a short ping sound on feedback

Keys:

- `q`: quit

## Notes

- For best results, use a well-lit environment and ensure the full body is visible.
- This MVP uses 2D landmarks and simple heuristics; expect imperfections. Thresholds are configurable.
- First run will download the YOLO Pose model weights (`yolo11n-pose.pt`).

## Structure

```
app/
  run_realtime.py
src/tennis_coach/
  __init__.py
  types.py
  pose.py
  smoothing.py
  swing.py
  rules.py
  overlay.py
  feedback.py
  utils.py
```

## Next steps

- Replace heuristic stroke classification with a trained 1D-CNN/LSTM on keypoint trajectories.
- Expand rule coverage and tune thresholds with labeled data.
- Port to on-device (TFLite/CoreML) for mobile.