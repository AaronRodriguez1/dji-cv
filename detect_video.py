import argparse
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:  # optional dependency
    DeepSort = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 on a video, annotate frames, and save an MP4."
    )
    parser.add_argument("--source", required=True, help="Path to input video.")
    parser.add_argument(
        "--output",
        default="annotated.mp4",
        help="Output MP4 path (default: annotated.mp4).",
    )
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="YOLOv8 weights path or model name (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. cpu, 0, 0,1 (default: auto).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--tracker",
        action="store_true",
        help="Enable DeepSORT tracking (requires deep_sort_realtime).",
    )
    return parser.parse_args()


def ensure_writer(cap: cv2.VideoCapture, output_path: Path) -> cv2.VideoWriter:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def draw_box(frame, x1, y1, x2, y2, label, color=(0, 255, 0)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)

    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    model = YOLO(args.weights)
    names = model.names

    tracker = None
    if args.tracker:
        if DeepSort is None:
            raise RuntimeError(
                "DeepSORT requested but deep_sort_realtime is not installed."
            )
        tracker = DeepSort(max_age=30, n_init=2)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    writer = ensure_writer(cap, output)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    with tqdm(total=total_frames, unit="frame") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                frame, conf=args.conf, device=args.device, verbose=False
            )[0]
            boxes = results.boxes

            if tracker is None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = f"{names.get(cls_id, cls_id)} {conf:.2f}"
                    draw_box(frame, x1, y1, x2, y2, label)
            else:
                detections = []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    if conf < args.conf:
                        continue
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

                tracks = tracker.update_tracks(detections, frame=frame)
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    cls_id = track.det_class
                    track_id = track.track_id
                    label = f"{names.get(cls_id, cls_id)} #{track_id}"
                    draw_box(frame, x1, y1, x2, y2, label, color=(0, 200, 255))

            writer.write(frame)
            pbar.update(1)

    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
