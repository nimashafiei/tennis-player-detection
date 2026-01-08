# 🎾 Tennis Player Detection (Half-Court)

Fast and robust tennis player detection for broadcast-style videos.
Automatically detects the net line, splits the court into near/far halves,
and tracks the main player using a speed-optimized YOLO pipeline.

## Demo
![Demo](demo.gif)

## Features
- Automatic net-line detection
- Half-court player filtering (near / far)
- YOLO-based person detection
- DET_EVERY + HOLD speed optimization
- Optional YOLO tracking
- CSV and video output

## Installation
```bash
pip install ultralytics opencv-python numpy
pip install openvino
```

## Usage
```bash
python detect_player_halfcourt.py --video_in cam1.mp4
```

## Notes
- For OpenVINO models, imgsz must match model input size.
- Disable --show for best performance.

## License
MIT
