from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE = [
    (239, 83, 80),
    (66, 165, 245),
    (102, 187, 106),
    (255, 202, 40),
    (171, 71, 188),
    (255, 112, 67),
    (38, 198, 218),
    (141, 110, 99),
]

_resampling = getattr(Image, "Resampling", None)
RESAMPLE_BILINEAR = _resampling.BILINEAR if _resampling is not None else 2


def to_uint8(array):
    data = np.asarray(array)
    if data.dtype == np.bool_:
        return data.astype(np.uint8) * 255
    if np.issubdtype(data.dtype, np.floating):
        return np.clip(data, 0, 255).astype(np.uint8)
    if data.dtype == np.uint8:
        return data

    data = data.astype(np.float32)
    data -= data.min()
    max_value = float(data.max())
    if max_value <= 0:
        return np.zeros_like(data, dtype=np.uint8)
    return np.clip(data * (255.0 / max_value), 0, 255).astype(np.uint8)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_image_gray(path):
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is not None:
        if image.ndim == 2:
            return to_uint8(image)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return to_uint8(gray)

    with Image.open(path) as handle:
        return np.array(handle.convert("L"), dtype=np.uint8)


def save_image(path, array):
    path = Path(path)
    ensure_dir(path.parent)

    data = to_uint8(array)

    Image.fromarray(data).save(path)


def connected_components(mask):
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    components = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]
        components.append(
            {
                "label": label,
                "bbox": [x, y, w, h],
                "area": area,
                "centroid": [float(cx), float(cy)],
                "mask": labels == label,
            }
        )
    return components


def largest_component(mask):
    components = connected_components(mask)
    if not components:
        return np.zeros_like(mask, dtype=np.uint8)

    largest = max(components, key=lambda item: item["area"])
    return largest["mask"].astype(np.uint8) * 255


def filter_components_by_area(mask, min_area):
    kept = np.zeros_like(mask, dtype=np.uint8)
    for component in connected_components(mask):
        if component["area"] >= min_area:
            kept[component["mask"]] = 255
    return kept


def colorize_regions(base_gray, label_masks_or_stats, alpha=0.4, draw_labels=False):
    base = np.asarray(base_gray)
    if base.ndim == 2:
        canvas = np.stack([base, base, base], axis=-1).astype(np.float32)
    else:
        canvas = base.astype(np.float32).copy()

    for index, region in enumerate(label_masks_or_stats):
        if isinstance(region, dict):
            mask = np.asarray(region["mask"]) > 0
            label_text = str(region.get("id", region.get("label", index + 1)))
            centroid = region.get("centroid")
        else:
            mask = np.asarray(region) > 0
            label_text = str(index + 1)
            centroid = None

        if not np.any(mask):
            continue

        color = np.array(PALETTE[index % len(PALETTE)], dtype=np.float32)
        canvas[mask] = canvas[mask] * (1.0 - alpha) + color * alpha

        contour_mask = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, tuple(int(value) for value in color), 2)

        if draw_labels:
            if centroid is None:
                ys, xs = np.where(mask)
                position = (int(xs.mean()), int(ys.mean()))
            else:
                position = (int(centroid[0]), int(centroid[1]))

            cv2.putText(
                canvas,
                label_text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label_text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                tuple(int(value) for value in color),
                1,
                cv2.LINE_AA,
            )

    return np.clip(canvas, 0, 255).astype(np.uint8)


def crop_to_bbox(image, mask, bbox):
    x, y, w, h = [int(value) for value in bbox]
    image_crop = np.asarray(image)[y : y + h, x : x + w].copy()
    mask_crop = (np.asarray(mask)[y : y + h, x : x + w] > 0).astype(np.uint8) * 255

    if image_crop.ndim == 2:
        image_crop[mask_crop == 0] = 0
    else:
        image_crop[mask_crop == 0] = 0

    return image_crop, mask_crop


def build_contact_sheet(items, cell_size, columns, padding, bg_value):
    cell_w, cell_h = cell_size
    rows = max(1, int(np.ceil(len(items) / max(columns, 1))))

    sheet_w = padding + columns * (cell_w + padding)
    sheet_h = padding + rows * (cell_h + padding)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (bg_value, bg_value, bg_value))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, item in enumerate(items):
        row = index // columns
        col = index % columns
        origin_x = padding + col * (cell_w + padding)
        origin_y = padding + row * (cell_h + padding)

        cell = Image.new("RGB", (cell_w, cell_h), (bg_value, bg_value, bg_value))
        cell_draw = ImageDraw.Draw(cell)
        cell_draw.rectangle((0, 0, cell_w - 1, cell_h - 1), outline=(210, 210, 210))

        image = np.asarray(item["image"])
        if image.ndim == 2:
            display = np.stack([image, image, image], axis=-1)
        else:
            display = image.copy()

        mask = item.get("mask")
        if mask is not None and np.any(mask):
            contour_mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (239, 83, 80), 1)

        tile = Image.fromarray(display.astype(np.uint8))
        max_tile_w = cell_w - 12
        max_tile_h = cell_h - 28
        scale = min(max_tile_w / max(tile.width, 1), max_tile_h / max(tile.height, 1))
        scale = min(scale, 1.0)
        resized = tile.resize(
            (max(1, int(tile.width * scale)), max(1, int(tile.height * scale))),
            resample=RESAMPLE_BILINEAR,
        )

        paste_x = (cell_w - resized.width) // 2
        paste_y = max(6, (max_tile_h - resized.height) // 2 + 4)
        cell.paste(resized, (paste_x, paste_y))

        label = item.get("label", "")
        if label:
            cell_draw.text((6, cell_h - 18), label, fill=(20, 20, 20), font=font)

        sheet.paste(cell, (origin_x, origin_y))

    if not items:
        draw.text((padding, padding), "No corpus pieces found", fill=(20, 20, 20), font=font)

    return np.array(sheet, dtype=np.uint8)
