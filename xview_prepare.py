import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


XVIEW_CLASS2INDEX = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0, 1, 2, -1, 3, -1, 4, 5, 6, 7, 8, -1, 9, 10, 11,
    12, 13, 14, 15, -1, -1, 16, 17, 18, 19, 20, 21, 22, -1,
    23, 24, 25, -1, 26, 27, -1, 28, -1, 29, 30, 31, 32, 33,
    34, 35, 36, 37, -1, 38, 39, 40, 41, 42, 43, 44, 45, -1,
    -1, -1, -1, 46, 47, 48, 49, -1, 50, 51, -1, 52, -1, -1,
    -1, 53, 54, -1, 55, -1, -1, 56, -1, 57, -1, 58, 59,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare xView for YOLOv8 (convert labels + autosplit)."
    )
    parser.add_argument(
        "--data-root",
        default="data/xview/raw",
        help="Folder containing xView_train.geojson and train_images/ (default: data/xview/raw).",
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
        "--geojson",
        default="xView_train.geojson",
        help="Train GeoJSON filename (default: xView_train.geojson).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for autosplit (default: 0).",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Train split ratio for autosplit (default: 0.9).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images into images/train|val instead of moving (default: move).",
    )
    return parser.parse_args()


def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    x_center = (x1 + x2) / 2.0 / w
    y_center = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return x_center, y_center, bw, bh


def clamp_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0.0, min(x1, w))
    y1 = max(0.0, min(y1, h))
    x2 = max(0.0, min(x2, w))
    y2 = max(0.0, min(y2, h))
    return x1, y1, x2, y2



def convert_labels(data_root: Path, geojson_path: Path, train_images: Path) -> None:
    labels_root = data_root / "labels" / "train"
    shutil.rmtree(labels_root, ignore_errors=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = data.get("features", [])

    image_sizes = {}

    for feature in tqdm(features, desc="Converting labels"):
        props = feature.get("properties", {})
        bounds = props.get("bounds_imcoords")
        image_id = props.get("image_id")
        class_id = props.get("type_id")

        if not bounds or image_id is None or class_id is None:
            continue

        parts = [p for p in bounds.split(",") if p.strip()]
        if len(parts) != 4:
            continue

        try:
            x1, y1, x2, y2 = map(float, parts)
        except ValueError:
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        mapped = XVIEW_CLASS2INDEX[int(class_id)] if int(class_id) < len(XVIEW_CLASS2INDEX) else -1
        if mapped < 0:
            continue

        image_path = train_images / image_id
        if not image_path.exists():
            continue

        if image_id not in image_sizes:
            with Image.open(image_path) as img:
                image_sizes[image_id] = img.size

        width, height = image_sizes[image_id]
        x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        x_center, y_center, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, width, height)
        if bw <= 0 or bh <= 0:
            continue

        label_path = labels_root / f"{Path(image_id).stem}.txt"
        with label_path.open("a", encoding="utf-8") as f:
            f.write(f"{mapped} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")


def move_images(data_root: Path, train_images: Path, val_images: Path, copy_images: bool) -> Path:
    images_root = data_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    train_dest = images_root / "train"
    val_dest = images_root / "val"
    train_dest.mkdir(parents=True, exist_ok=True)
    val_dest.mkdir(parents=True, exist_ok=True)

    def move_or_copy(src: Path, dst: Path) -> None:
        if copy_images:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))

    for image_path in tqdm(list(train_images.glob("*")), desc="Moving train images"):
        if image_path.is_file():
            move_or_copy(image_path, train_dest / image_path.name)

    for image_path in tqdm(list(val_images.glob("*")), desc="Moving val images"):
        if image_path.is_file():
            move_or_copy(image_path, val_dest / image_path.name)

    if not copy_images:
        train_images.rmdir()
        val_images.rmdir()

    return images_root


def autosplit(images_root: Path, train_ratio: float, seed: int) -> None:
    train_dir = images_root / "train"
    image_files = [p for p in train_dir.glob("*") if p.is_file()]
    rng = random.Random(seed)
    rng.shuffle(image_files)

    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    train_txt = images_root / "autosplit_train.txt"
    val_txt = images_root / "autosplit_val.txt"

    train_txt.write_text(
        "\n".join(str(p.resolve()) for p in train_files),
        encoding="utf-8",
    )
    val_txt.write_text(
        "\n".join(str(p.resolve()) for p in val_files),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    train_images = data_root / args.train_images
    val_images = data_root / args.val_images
    geojson = data_root / args.geojson
    images_root = data_root / "images"
    skip_move = False

    if not train_images.exists() or not val_images.exists():
        if (images_root / "train").exists() and (images_root / "val").exists():
            train_images = images_root / "train"
            val_images = images_root / "val"
            skip_move = True
        else:
            raise FileNotFoundError(
                "train_images/ or val_images/ not found under data root, "
                "and data/xview/raw/images/train|val not present."
            )
    if not geojson.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson}")

    convert_labels(data_root, geojson, train_images)
    if not skip_move:
        images_root = move_images(data_root, train_images, val_images, args.copy)
    autosplit(images_root, args.train_split, args.seed)

    print("xView prepare complete.")
    print("Labels: data/xview/raw/labels/train")
    print("Images: data/xview/raw/images/train and data/xview/raw/images/val")
    print("Splits: data/xview/raw/images/autosplit_train.txt and autosplit_val.txt")


if __name__ == "__main__":
    main()
