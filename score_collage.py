from pathlib import Path

import cv2
import numpy as np

from utils import load_image_gray


GRAYSCALE_DIR = Path("data/grayscale")
MASK_DIR = Path("data/masks")


def load_target_assets(stem, canvas_size):
    gray_path = GRAYSCALE_DIR / f"{stem}_gray.png"
    weight_path = MASK_DIR / f"{stem}_weights.png"

    target_gray = load_image_gray(gray_path).astype(np.float32)
    target_weights = load_image_gray(weight_path).astype(np.float32) / 255.0

    target_gray = cv2.resize(target_gray, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)
    target_weights = cv2.resize(target_weights, (canvas_size, canvas_size), interpolation=cv2.INTER_LINEAR)
    target_weights = np.clip(target_weights, 0.0, 1.0)
    return target_gray, target_weights


def weighted_l1_score(rendered, target_gray, target_weights):
    diff = np.abs(rendered.astype(np.float32) - target_gray.astype(np.float32))
    return float(np.mean(diff * target_weights.astype(np.float32)))
