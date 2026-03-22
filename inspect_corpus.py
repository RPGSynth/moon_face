import json
from pathlib import Path

import numpy as np

from utils import build_contact_sheet, ensure_dir, load_image_gray, save_image


INDEX_PATH = Path("data/corpus/corpus_index.json")
DEBUG_DIR = Path("out/debug")
SHEET_COLUMNS = 4
CELL_SIZE = (220, 220)
CELL_PADDING = 14
SHEET_BG = 244


def summarize(values):
    values = np.asarray(values, dtype=np.float32)
    return float(values.min()), float(values.max()), float(values.mean())


def main():
    ensure_dir(DEBUG_DIR)
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Corpus index not found: {INDEX_PATH}")

    with INDEX_PATH.open("r", encoding="utf-8") as handle:
        corpus = json.load(handle)

    corpus = sorted(corpus, key=lambda item: item["area"], reverse=True)
    items = []

    for piece in corpus:
        image = load_image_gray(piece["image_path"])
        mask = load_image_gray(piece["mask_path"])
        label = f'{piece["id"]}  {piece["width"]}x{piece["height"]}'
        items.append({"image": image, "mask": mask, "label": label})

    sheet = build_contact_sheet(items, cell_size=CELL_SIZE, columns=SHEET_COLUMNS, padding=CELL_PADDING, bg_value=SHEET_BG)
    save_image(DEBUG_DIR / "corpus_sheet.png", sheet)

    if not corpus:
        print("pieces: 0")
        print("corpus is empty")
        return

    areas = [piece["area"] for piece in corpus]
    widths = [piece["width"] for piece in corpus]
    heights = [piece["height"] for piece in corpus]

    area_min, area_max, area_mean = summarize(areas)
    width_min, width_max, width_mean = summarize(widths)
    height_min, height_max, height_mean = summarize(heights)

    print(f"pieces: {len(corpus)}")
    print(f"area min/max/mean: {area_min:.1f} / {area_max:.1f} / {area_mean:.1f}")
    print(f"width min/max/mean: {width_min:.1f} / {width_max:.1f} / {width_mean:.1f}")
    print(f"height min/max/mean: {height_min:.1f} / {height_max:.1f} / {height_mean:.1f}")
    print("sheet: out/debug/corpus_sheet.png")


if __name__ == "__main__":
    main()
