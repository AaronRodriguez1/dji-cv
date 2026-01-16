# YOLOv8 Video Annotation

Run YOLOv8 on a video, draw bounding boxes, and save an annotated MP4. Optional
DeepSORT tracking adds stable IDs across frames.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python detect_video.py --source input.mp4 --output annotated.mp4
```

Enable DeepSORT tracking:

```bash
python detect_video.py --source input.mp4 --output annotated.mp4 --tracker
```

Use a different model or set confidence:

```bash
python detect_video.py --source input.mp4 --weights yolov8s.pt --conf 0.35
```

## Notes

- The first run may download YOLOv8 weights.
- If you do not want DeepSORT, you can remove `deep_sort_realtime` from
  `requirements.txt`.
