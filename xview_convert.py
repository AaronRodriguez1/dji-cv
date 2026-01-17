import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert xView annotations (GeoJSON) to YOLO format."
    )
    parser.add_argument(
        "--data-root",
        default="data/xview/raw",
        help="Folder containing xView images and GeoJSON (default: data/xview/raw).",
    )
    parser.add_argument(
        "--out",
        default="data/xview/yolo",
        help="Output dataset root (default: data/xview/yolo).",
    )
    parser.add_argument(
        "--train-geojson",
        default="xView_train.geojson",
        help="Train GeoJSON filename (default: xView_train.geojson).",
    )
    parser.add_argument(
        "--val-geojson",
        default="xView_val.geojson",
        help="Val GeoJSON filename (default: xView_val.geojson).",
    )
    parser.add_argument(
        "--train-images",
        default="train_images",
        help="Train images folder name (default: train_images).",
    )
    parser.add_argument(
        "--val-images",
        default="val_images",
        help="Val images folder name (default: val_images).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images into the YOLO dataset (recommended).",
    )
    return parser.parse_args()


def load_annotations(geojson_path: Path) -> dict:
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    mapping = {}

    for feature in features:
        props = feature.get("properties", {})
        image_id = props.get("image_id")
        bounds = props.get("bounds_imcoords")
        class_id = props.get("type_id")

        if image_id is None or class_id is None:
            continue
        if not bounds or bounds in ("0,0,0,0", "-1,-1,-1,-1"):
            continue

        parts = [p for p in bounds.split(",") if p.strip()]
        if len(parts) < 8:
            continue

        try:
            coords = list(map(float, parts))
        except ValueError:
            continue

        xs = coords[0::2]
        ys = coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        if x_max <= x_min or y_max <= y_min:
            continue

        image_key = str(image_id)
        mapping.setdefault(image_key, []).append((int(class_id) - 1, x_min, y_min, x_max, y_max))

    return mapping


def convert_split(images_dir: Path, geojson_path: Path, out_root: Path, copy_images: bool) -> None:
    annotations = load_annotations(geojson_path)

    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_paths = list(images_dir.glob("*"))
    for image_path in image_paths:
        if not image_path.is_file():
            continue
        image_id = image_path.stem
        labels = annotations.get(image_id, [])

        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            continue

        label_lines = []
        for class_id, x_min, y_min, x_max, y_max in labels:
            x_center = (x_min + x_max) / 2.0 / width
            y_center = (y_min + y_max) / 2.0 / height
            w_norm = (x_max - x_min) / width
            h_norm = (y_max - y_min) / height

            if w_norm <= 0 or h_norm <= 0:
                continue

            label_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

        out_label_path = out_labels / f"{image_id}.txt"
        out_label_path.write_text("\n".join(label_lines), encoding="utf-8")

        if copy_images:
            shutil.copy2(image_path, out_images / image_path.name)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out)

    train_images = data_root / args.train_images
    val_images = data_root / args.val_images
    train_geojson = data_root / args.train_geojson
    val_geojson = data_root / args.val_geojson

    if not train_images.exists() or not val_images.exists():
        raise FileNotFoundError("Train/val images folders not found under data root.")
    if not train_geojson.exists() or not val_geojson.exists():
        raise FileNotFoundError("Train/val GeoJSON files not found under data root.")

    convert_split(train_images, train_geojson, out_root / "train", args.copy)
    convert_split(val_images, val_geojson, out_root / "val", args.copy)

    print("Conversion complete.")
    print("If you used --copy, images are in data/xview/yolo/{train,val}/images.")


if __name__ == "__main__":
    main()
