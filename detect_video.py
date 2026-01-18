import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# Try importing DeepSort, handle if missing
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    DeepSort = None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 on a video, annotate frames, and save an MP4."
    )
    parser.add_argument("--source", required=True, help="Path to input video.")
    parser.add_argument("--output", default="annotated.mp4", help="Output MP4 path.")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO weights.")
    parser.add_argument("--device", default=None, help="Inference device (0, cpu).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    
    # --- ADDED: Argument to control inference size ---
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size (pixels). Reduce this (e.g., 416, 320) if objects are too large.")
    
    parser.add_argument("--tracker", action="store_true", help="Enable DeepSORT.")
    return parser.parse_args()

def ensure_writer(cap: cv2.VideoCapture, output_path: Path) -> cv2.VideoWriter:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 'mp4v' is widely supported
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

def draw_box(frame, x1, y1, x2, y2, label, color=(0, 255, 0)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        t_size = cv2.getTextSize(label, 0, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)



def color_for_class(cls_id: int) -> tuple[int, int, int]:
    # Simple deterministic palette (BGR for OpenCV)
    palette = [
        (255, 56, 56),   # red
        (255, 157, 151), # light red
        (255, 112, 31),  # orange
        (255, 178, 29),  # yellow-orange
        (207, 210, 49),  # yellow
        (72, 249, 10),   # green
        (146, 204, 23),  # yellow-green
        (61, 219, 134),  # teal
        (26, 147, 52),   # dark green
        (0, 212, 187),   # cyan
        (44, 153, 168),  # blue-green
        (0, 194, 255),   # light blue
        (52, 69, 147),   # blue
        (100, 115, 255), # indigo
        (0, 24, 236),    # deep blue
        (132, 56, 255),  # purple
        (82, 0, 133),    # dark purple
        (203, 56, 255),  # magenta
        (255, 149, 200), # pink
        (255, 55, 199),  # hot pink
    ]
    return palette[cls_id % len(palette)]
def main():
    args = parse_args()
    
    # Initialize YOLO
    model = YOLO(args.weights)
    
    # Initialize Tracker (optional)
    tracker = None
    if args.tracker and DeepSort:
        tracker = DeepSort(max_age=30)
    elif args.tracker and DeepSort is None:
        print("Warning: DeepSort not installed. Running without tracker.")
    
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error opening video: {args.source}")
        return

    writer = ensure_writer(cap, Path(args.output))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {args.source}")
    print(f"Inference Size: {args.imgsz}")

    for _ in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        # imgsz=args.imgsz scales the image for the model
        results = model(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
        
        detections = []
        # Parse results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Use dictionary get in case names aren't fully populated
                class_name = model.names.get(cls_id, str(cls_id))
                label = f"{class_name} {conf:.2f}"
                color = color_for_class(cls_id)
                
                # Draw directly if no tracker
                if not tracker:
                    draw_box(frame, x1, y1, x2, y2, label, color=color)
                
                # Prepare for tracker: [left, top, w, h], conf, class_id
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], conf, cls_id))

        # Update tracker if enabled
        if tracker:
            tracks = tracker.update_tracks(detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb() # x1, y1, x2, y2
                
                # Draw tracked box
                cls_id = track.get_det_class()
                class_name = model.names.get(cls_id, str(cls_id))
                label = f"ID:{track_id} {class_name}"
                color = color_for_class(cls_id)
                draw_box(frame, int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]), label, color=color)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()