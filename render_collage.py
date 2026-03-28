import json
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from prepare_corpus import prepare_primitives, write_prepared_outputs
from utils import load_image_gray


PREPARED_INDEX_PATH = Path("data/corpus/primitives_index.json")


class Placement(TypedDict):
    piece_id: str
    x: float
    y: float
    scale: float
    rotation_deg: float
    flip_x: bool
    flip_y: bool
    fill_value: int


def load_primitives(index_path=PREPARED_INDEX_PATH):
    index_path = Path(index_path)
    if not index_path.exists():
        write_prepared_outputs(prepare_primitives())

    records = json.loads(index_path.read_text())
    if records and "image_path" not in records[0]:
        write_prepared_outputs(prepare_primitives())
        records = json.loads(index_path.read_text())

    primitives = {}
    for record in records:
        image = load_image_gray(record["image_path"]).astype(np.float32)
        mask = load_image_gray(record["mask_path"]).astype(np.float32) / 255.0
        primitives[record["id"]] = {
            **record,
            "image": np.clip(image, 0.0, 255.0),
            "mask": np.clip(mask, 0.0, 1.0),
        }
    return primitives


def apply_flips(mask, flip_x, flip_y):
    if flip_x and flip_y:
        return cv2.flip(mask, -1)
    if flip_x:
        return cv2.flip(mask, 1)
    if flip_y:
        return cv2.flip(mask, 0)
    return mask


def transform_image(image, placement, canvas_shape, interpolation=cv2.INTER_LINEAR):
    canvas_h, canvas_w = [int(value) for value in canvas_shape]
    source = apply_flips(image, placement["flip_x"], placement["flip_y"])

    height, width = source.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(placement["rotation_deg"]), float(placement["scale"]))
    matrix[0, 2] += float(placement["x"]) - center[0]
    matrix[1, 2] += float(placement["y"]) - center[1]

    transformed = cv2.warpAffine(
        source.astype(np.float32),
        matrix,
        (canvas_w, canvas_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return transformed.astype(np.float32)


def transform_mask(mask, placement, canvas_shape):
    transformed = transform_image(mask, placement, canvas_shape, interpolation=cv2.INTER_LINEAR)
    return np.clip(transformed, 0.0, 1.0)


def tone_match_image(transformed_image, transformed_mask, fill_value):
    coverage = transformed_mask > 0.01
    if not np.any(coverage):
        return np.clip(transformed_image, 0.0, 255.0)

    weights = transformed_mask[coverage].astype(np.float32)
    mean_value = float(np.sum(transformed_image[coverage] * weights) / max(np.sum(weights), 1e-6))
    shift = float(fill_value) - mean_value
    return np.clip(transformed_image.astype(np.float32) + shift, 0.0, 255.0)


def composite_mask(canvas, transformed_mask, transformed_image, fill_value, global_alpha, use_piece_texture=True):
    effective_alpha = np.clip(transformed_mask.astype(np.float32) * float(global_alpha), 0.0, 1.0)
    if use_piece_texture:
        piece_image = tone_match_image(transformed_image, transformed_mask, fill_value)
    else:
        piece_image = np.full_like(canvas, float(fill_value), dtype=np.float32)
    return canvas.astype(np.float32) * (1.0 - effective_alpha) + piece_image * effective_alpha


def render_canvas(primitives, placements, canvas_shape, global_alpha, background_value=255.0, use_piece_texture=True):
    canvas_h, canvas_w = [int(value) for value in canvas_shape]
    canvas = np.full((canvas_h, canvas_w), float(background_value), dtype=np.float32)
    for placement in placements:
        primitive = primitives[placement["piece_id"]]
        transformed_mask = transform_mask(primitive["mask"], placement, (canvas_h, canvas_w))
        transformed_image = transform_image(primitive["image"], placement, (canvas_h, canvas_w))
        canvas = composite_mask(
            canvas,
            transformed_mask,
            transformed_image,
            placement["fill_value"],
            global_alpha,
            use_piece_texture=use_piece_texture,
        )
    return np.clip(canvas, 0.0, 255.0)
