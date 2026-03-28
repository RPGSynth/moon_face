from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from utils import ensure_dir, save_image, to_uint8


SOURCE_DIR = Path("data/source/faces")
GRAYSCALE_DIR = Path("data/grayscale")
MASK_DIR = Path("data/masks")
WEIGHTED_DIR = Path("data/weighted")
DEBUG_DIR = Path("out/debug/weights")
MODEL_PATH = Path("data/models/face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

BASELINE_WEIGHT = 0.35
FACE_WEIGHT = 0.50
CONTOUR_WEIGHT = 0.60
NOSE_WEIGHT = 0.72
BROW_WEIGHT = 0.90
FEATURE_WEIGHT = 1.00
FALLBACK_WEIGHT = 0.50

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
NOSE_BRIDGE = [168, 6, 197, 195, 5]
NOSE_TIP = [4, 1, 19, 94, 2, 98, 327]


def ensure_output_dirs():
    for path in [SOURCE_DIR, GRAYSCALE_DIR, MASK_DIR, WEIGHTED_DIR, DEBUG_DIR, MODEL_PATH.parent]:
        ensure_dir(path)


def list_source_images():
    return sorted(
        path for path in SOURCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def ensure_model():
    if MODEL_PATH.exists():
        return MODEL_PATH.resolve()

    ensure_dir(MODEL_PATH.parent)
    print(f"downloading face landmarker model to {MODEL_PATH}")
    urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH.resolve()


def load_landmarker():
    try:
        model_path = str(ensure_model())
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.FaceLandmarker.create_from_options(options)
    except Exception as exc:
        print(f"warning: could not initialize MediaPipe FaceLandmarker: {exc}")
        return None


def landmark_points(face_landmarks, width, height):
    points = []
    for landmark in face_landmarks:
        x = int(round(np.clip(landmark.x, 0.0, 1.0) * (width - 1)))
        y = int(round(np.clip(landmark.y, 0.0, 1.0) * (height - 1)))
        points.append((x, y))
    return np.asarray(points, dtype=np.int32)


def select_points(points, indices):
    return points[np.asarray(indices, dtype=np.int32)]


def normalize_layer(layer):
    layer = np.asarray(layer, dtype=np.float32)
    max_value = float(layer.max())
    if max_value > 0:
        layer = layer / max_value
    return np.clip(layer, 0.0, 1.0)


def soft_mask_from_polygon(shape, points, blur_size, convex=False, expand=0):
    layer = np.zeros(shape[:2], dtype=np.float32)
    if len(points) < 3:
        return layer

    draw_points = points.astype(np.int32)
    if convex:
        draw_points = cv2.convexHull(draw_points)
        cv2.fillConvexPoly(layer, draw_points, 1.0)
    else:
        cv2.fillPoly(layer, [draw_points], 1.0)

    if expand > 0:
        kernel_size = expand if expand % 2 == 1 else expand + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        layer = cv2.dilate(layer, kernel, iterations=1)

    layer = cv2.GaussianBlur(layer, (blur_size, blur_size), 0)
    return normalize_layer(layer)


def soft_mask_from_polyline(shape, points, thickness, blur_size):
    layer = np.zeros(shape[:2], dtype=np.float32)
    if len(points) < 2:
        return layer
    cv2.polylines(layer, [points.astype(np.int32)], False, 1.0, thickness, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (blur_size, blur_size), 0)
    return normalize_layer(layer)


def ellipse_from_points(layer, points, pad_x, pad_y):
    if len(points) == 0:
        return
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    center = (int(x + w / 2), int(y + h / 2))
    axes = (max(2, int(w / 2 + pad_x)), max(2, int(h / 2 + pad_y)))
    cv2.ellipse(layer, center, axes, 0, 0, 360, 1.0, -1, cv2.LINE_AA)


def soft_nose_mask(shape, points, blur_size):
    height, width = shape[:2]
    layer = np.zeros((height, width), dtype=np.float32)

    bridge_points = select_points(points, NOSE_BRIDGE)
    tip_points = select_points(points, NOSE_TIP)
    pad_x = max(2, width // 90)
    pad_y = max(3, height // 70)

    ellipse_from_points(layer, bridge_points, pad_x=pad_x, pad_y=pad_y * 2)
    ellipse_from_points(layer, tip_points, pad_x=pad_x * 2, pad_y=pad_y)

    layer = cv2.GaussianBlur(layer, (blur_size, blur_size), 0)
    return normalize_layer(layer)


def apply_soft_weight(weight_map, layer, value):
    target = BASELINE_WEIGHT + (value - BASELINE_WEIGHT) * np.clip(layer, 0.0, 1.0)
    np.maximum(weight_map, target, out=weight_map)


def blur_kernel_for_shape(shape):
    height, width = shape[:2]
    base = max(9, min(height, width) // 18)
    base = min(base, 31)
    return base if base % 2 == 1 else base + 1


def make_fallback_map(shape):
    height, width = shape[:2]
    return np.full((height, width), FALLBACK_WEIGHT, dtype=np.float32)


def build_weight_map(points, shape):
    height, width = shape[:2]
    weight_map = np.full((height, width), BASELINE_WEIGHT, dtype=np.float32)
    blur_size = blur_kernel_for_shape(shape)
    feature_expand = max(3, min(height, width) // 80)

    face_points = select_points(points, FACE_OVAL)
    apply_soft_weight(
        weight_map,
        soft_mask_from_polygon(shape, face_points, blur_size, expand=feature_expand * 2),
        FACE_WEIGHT,
    )

    jaw_points = face_points[face_points[:, 1] >= int(face_points[:, 1].mean())]
    if len(jaw_points) >= 2:
        jaw_thickness = max(4, min(height, width) // 35)
        apply_soft_weight(
            weight_map,
            soft_mask_from_polyline(shape, jaw_points, jaw_thickness, blur_size),
            CONTOUR_WEIGHT,
        )

    left_brow = select_points(points, LEFT_BROW)
    right_brow = select_points(points, RIGHT_BROW)
    apply_soft_weight(
        weight_map,
        soft_mask_from_polygon(shape, left_brow, blur_size, convex=True, expand=feature_expand),
        BROW_WEIGHT,
    )
    apply_soft_weight(
        weight_map,
        soft_mask_from_polygon(shape, right_brow, blur_size, convex=True, expand=feature_expand),
        BROW_WEIGHT,
    )

    left_eye = select_points(points, LEFT_EYE)
    right_eye = select_points(points, RIGHT_EYE)
    apply_soft_weight(
        weight_map,
        soft_mask_from_polygon(shape, left_eye, blur_size, expand=feature_expand),
        FEATURE_WEIGHT,
    )
    apply_soft_weight(
        weight_map,
        soft_mask_from_polygon(shape, right_eye, blur_size, expand=feature_expand),
        FEATURE_WEIGHT,
    )
    apply_soft_weight(weight_map, soft_nose_mask(shape, points, blur_size), NOSE_WEIGHT)

    mouth = select_points(points, MOUTH)
    apply_soft_weight(
        weight_map,
        soft_mask_from_polygon(shape, mouth, blur_size, expand=feature_expand),
        FEATURE_WEIGHT,
    )

    weight_map = cv2.GaussianBlur(weight_map, (blur_size, blur_size), 0)
    return np.clip(weight_map, 0.0, 1.0)


def weight_preview(weight_map):
    preview = cv2.applyColorMap(to_uint8(weight_map * 255.0), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def draw_region_outline(image_rgb, points, color, closed=True, thickness=2):
    if len(points) < 2:
        return
    cv2.polylines(image_rgb, [points.astype(np.int32)], closed, color, thickness, cv2.LINE_AA)


def make_overlay(image_rgb, points, fallback_used):
    overlay = image_rgb.copy()
    if fallback_used or points is None:
        cv2.putText(
            overlay,
            "fallback weights",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 80, 80),
            2,
            cv2.LINE_AA,
        )
        return overlay

    draw_region_outline(overlay, select_points(points, FACE_OVAL), (255, 255, 255), closed=True, thickness=2)
    draw_region_outline(overlay, select_points(points, LEFT_BROW), (255, 215, 0), closed=False, thickness=2)
    draw_region_outline(overlay, select_points(points, RIGHT_BROW), (255, 215, 0), closed=False, thickness=2)
    draw_region_outline(overlay, select_points(points, LEFT_EYE), (80, 180, 255), closed=True, thickness=2)
    draw_region_outline(overlay, select_points(points, RIGHT_EYE), (80, 180, 255), closed=True, thickness=2)
    draw_region_outline(overlay, select_points(points, NOSE_BRIDGE), (140, 255, 140), closed=False, thickness=2)
    draw_region_outline(overlay, select_points(points, NOSE_TIP), (140, 255, 140), closed=True, thickness=2)
    draw_region_outline(overlay, select_points(points, MOUTH), (255, 120, 160), closed=True, thickness=2)
    return overlay


def to_rgb(gray_image):
    return np.stack([gray_image, gray_image, gray_image], axis=-1)


def labeled_tile(image_rgb, label):
    tile = image_rgb.copy()
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (255, 255, 255), 1)
    cv2.putText(tile, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(tile, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 1, cv2.LINE_AA)
    return tile


def make_debug_panel(original_rgb, gray_image, preview_rgb, weighted_image):
    top = np.concatenate(
        [
            labeled_tile(original_rgb, "original"),
            labeled_tile(to_rgb(gray_image), "grayscale"),
        ],
        axis=1,
    )
    bottom = np.concatenate(
        [
            labeled_tile(preview_rgb, "weights"),
            labeled_tile(to_rgb(weighted_image), "weighted"),
        ],
        axis=1,
    )
    return np.concatenate([top, bottom], axis=0)


def detect_landmarks(landmarker, image_rgb):
    if landmarker is None:
        return None

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    try:
        result = landmarker.detect(mp_image)
    except Exception as exc:
        print(f"warning: landmark detection failed: {exc}")
        return None

    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def process_image(path, landmarker):
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"warning: could not read {path}")
        return False

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    save_image(GRAYSCALE_DIR / f"{path.stem}_gray.png", gray)

    face_landmarks = detect_landmarks(landmarker, image_rgb)
    if face_landmarks is None:
        points = None
        fallback_used = True
        weight_map = make_fallback_map(gray.shape)
    else:
        points = landmark_points(face_landmarks, gray.shape[1], gray.shape[0])
        fallback_used = False
        weight_map = build_weight_map(points, gray.shape)

    weight_image = to_uint8(weight_map * 255.0)
    multiplier = 0.7 + 0.6 * weight_map
    weighted = np.clip(gray.astype(np.float32) * multiplier, 0, 255).astype(np.uint8)

    overlay = make_overlay(image_rgb, points, fallback_used)
    preview = weight_preview(weight_map)
    panel = make_debug_panel(image_rgb, gray, preview, weighted)

    save_image(MASK_DIR / f"{path.stem}_weights.png", weight_image)
    save_image(WEIGHTED_DIR / f"{path.stem}_weighted.png", weighted)
    save_image(DEBUG_DIR / f"{path.stem}_landmarks_overlay.png", overlay)
    save_image(DEBUG_DIR / f"{path.stem}_weight_preview.png", preview)
    save_image(DEBUG_DIR / f"{path.stem}_debug_panel.png", panel)

    mode = "fallback" if fallback_used else "landmarks"
    print(f"processed {path.name} [{mode}]")
    return True


def main():
    ensure_output_dirs()
    images = list_source_images()
    if not images:
        print(f"no images found in {SOURCE_DIR}")
        return

    landmarker = load_landmarker()
    try:
        processed = 0
        for path in images:
            processed += int(process_image(path, landmarker))
        print(f"done: processed {processed} image(s)")
    finally:
        if landmarker is not None:
            landmarker.close()


if __name__ == "__main__":
    main()
