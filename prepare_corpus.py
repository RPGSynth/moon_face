import json
from pathlib import Path

import numpy as np

from utils import build_contact_sheet, ensure_dir, load_image_gray, save_image


CORPUS_DIR = Path("data/corpus")
SOURCE_INDEX_PATH = CORPUS_DIR / "corpus_index.json"
PREPARED_INDEX_PATH = CORPUS_DIR / "primitives_index.json"
DEBUG_DIR = Path("out/debug")
DEBUG_SHEET_PATH = DEBUG_DIR / "corpus_primitives_sheet.png"
PREVIEW_FILL_VALUE = 96


def load_mask_alpha(path):
    mask = load_image_gray(path).astype(np.float32) / 255.0
    return np.clip(mask, 0.0, 1.0)


def mask_centroid(mask):
    ys, xs = np.where(mask > 0.01)
    if ys.size == 0:
        height, width = mask.shape
        return [float(width) / 2.0, float(height) / 2.0]
    return [float(xs.mean()), float(ys.mean())]


def build_primitive_record(item):
    mask = load_mask_alpha(item["mask_path"])
    width = int(item["width"])
    height = int(item["height"])

    return {
        "id": item["id"],
        "image_path": item["image_path"],
        "mask_path": item["mask_path"],
        "width": width,
        "height": height,
        "area": int(item["area"]),
        "centroid": mask_centroid(mask),
        "aspect_ratio": float(width / max(height, 1)),
        "bbox": item["bbox"],
    }


def build_preview_items(records):
    items = []
    for record in records:
        mask = load_mask_alpha(record["mask_path"])
        primitive = 255.0 - (255.0 - PREVIEW_FILL_VALUE) * mask
        items.append(
            {
                "image": primitive.astype(np.uint8),
                "mask": mask > 0.01,
                "label": record["id"],
            }
        )
    return items


def prepare_primitives():
    source_records = json.loads(SOURCE_INDEX_PATH.read_text())
    prepared = [build_primitive_record(item) for item in source_records]
    prepared.sort(key=lambda item: item["id"])
    return prepared


def write_prepared_outputs(records):
    ensure_dir(PREPARED_INDEX_PATH.parent)
    ensure_dir(DEBUG_DIR)
    PREPARED_INDEX_PATH.write_text(json.dumps(records, indent=2))

    preview_items = build_preview_items(records)
    sheet = build_contact_sheet(preview_items, cell_size=(160, 160), columns=4, padding=12, bg_value=248)
    save_image(DEBUG_SHEET_PATH, sheet)


def main():
    records = prepare_primitives()
    write_prepared_outputs(records)
    print(f"prepared primitives: {len(records)}")
    print(f"wrote index: {PREPARED_INDEX_PATH}")
    print(f"wrote preview: {DEBUG_SHEET_PATH}")


if __name__ == "__main__":
    main()
