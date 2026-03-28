import json
import heapq
from pathlib import Path

import cv2
import numpy as np

from utils import colorize_regions, connected_components, crop_to_bbox, ensure_dir, load_image_gray, save_image


# Corpus tuning:
# - MIN_PIECE_AREA: higher = discard more tiny pieces, lower = keep smaller fragments
# - MAX_COMPONENT_AREA: lower = split big regions more aggressively
# - MAX_BBOX_DIM: lower = split wide/tall regions more aggressively
# - MIN_SPLIT_FILL_RATIO: higher = reject sparse/weak split pieces more often
# - MAX_SPLIT_PIECES: upper cap on how many pieces one oversized region can become
# - GEODESIC_NOISE_WEIGHT / GEODESIC_NOISE_BLUR: bend split boundaries so they are less straight
MIN_PIECE_AREA = 3000
MAX_COMPONENT_AREA = 17500
MAX_BBOX_DIM = 3000
MIN_SPLIT_FILL_RATIO = 0.3
MAX_SPLIT_PIECES = 20
GEODESIC_NOISE_WEIGHT = 0.55
GEODESIC_NOISE_BLUR = 65

MASK_PATH = Path("data/masks/maria_mask.png")
GRAY_PATH = Path("data/masks/moon_gray.png")
CORPUS_DIR = Path("data/corpus")
DEBUG_DIR = Path("out/debug")
INDEX_PATH = CORPUS_DIR / "corpus_index.json"


def clear_previous_outputs():
    ensure_dir(CORPUS_DIR)
    for pattern in ("piece_*.png", "piece_*_mask.png", "corpus_index.json"):
        for path in CORPUS_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def estimate_split_count(component, max_area, max_bbox_dim):
    _, _, w, h = component["bbox"]
    area_count = int(np.ceil(component["area"] / float(max_area)))
    dim_count = int(np.ceil(max(w, h) / float(max_bbox_dim)))
    return max(1, min(MAX_SPLIT_PIECES, max(area_count, dim_count)))


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return [x0, y0, x1 - x0, y1 - y0]


def odd_kernel(size):
    return size if size % 2 == 1 else size + 1


def build_seed_points(component_crop, cluster_count):
    ys, xs = np.where(component_crop > 0)
    if ys.size == 0:
        return []

    points = np.column_stack([xs, ys]).astype(np.float32)
    cluster_count = min(cluster_count, len(points))
    if cluster_count <= 1:
        return [(int(points[0, 0]), int(points[0, 1]))]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    best_labels = np.zeros((len(points), 1), dtype=np.int32)
    attempts = 5
    flags = cv2.KMEANS_PP_CENTERS
    _, _, centers = cv2.kmeans(points, cluster_count, best_labels, criteria, attempts, flags)

    seeds = []
    used = set()
    for center in centers:
        distances = np.sum((points - center) ** 2, axis=1)
        index = int(np.argmin(distances))
        sx, sy = int(points[index, 0]), int(points[index, 1])
        if (sx, sy) in used:
            continue
        used.add((sx, sy))
        seeds.append((sx, sy))
    return seeds


def build_noise_field(shape, seed_value):
    rng = np.random.default_rng(seed_value)
    noise = rng.random(shape, dtype=np.float32)
    blur_size = odd_kernel(GEODESIC_NOISE_BLUR)
    noise = cv2.GaussianBlur(noise, (blur_size, blur_size), 0)
    normalized = np.empty_like(noise)
    cv2.normalize(noise, normalized, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    return normalized


def geodesic_partition(component_crop, seeds, noise_field):
    h, w = component_crop.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    distances = np.full((h, w), np.inf, dtype=np.float32)
    heap = []

    for label, (sx, sy) in enumerate(seeds):
        if sx < 0 or sy < 0 or sx >= w or sy >= h or component_crop[sy, sx] == 0:
            continue
        labels[sy, sx] = label
        distances[sy, sx] = 0.0
        heapq.heappush(heap, (0.0, label, sx, sy))

    if not heap:
        return None

    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.4142),
        (-1, 1, 1.4142),
        (1, -1, 1.4142),
        (1, 1, 1.4142),
    ]

    while heap:
        current_cost, label, x, y = heapq.heappop(heap)
        if current_cost > distances[y, x]:
            continue

        for dx, dy, step_cost in neighbors:
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            if component_crop[ny, nx] == 0:
                continue

            noise_term = 1.0 + GEODESIC_NOISE_WEIGHT * float(noise_field[ny, nx] - 0.5)
            candidate_cost = current_cost + step_cost * noise_term
            if candidate_cost < distances[ny, nx]:
                distances[ny, nx] = candidate_cost
                labels[ny, nx] = label
                heapq.heappush(heap, (candidate_cost, label, nx, ny))

    return labels


def split_component_shape(component, image_shape, target_count):
    x, y, w, h = [int(value) for value in component["bbox"]]
    component_crop = component["mask"][y : y + h, x : x + w].astype(np.uint8) * 255
    if target_count <= 1:
        return []

    ys, xs = np.where(component_crop > 0)
    if ys.size == 0:
        return []

    points = np.column_stack([xs, ys]).astype(np.float32)
    max_clusters_by_area = max(1, len(points) // max(MIN_PIECE_AREA, 1))
    cluster_count = min(target_count, max_clusters_by_area, len(points))
    if cluster_count <= 1:
        return []

    seeds = build_seed_points(component_crop, cluster_count)
    if len(seeds) <= 1:
        return []

    seed_value = int((x * 73856093 + y * 19349663 + component["area"] * 83492791) % (2**32))
    noise_field = build_noise_field(component_crop.shape, seed_value)
    labels = geodesic_partition(component_crop, seeds, noise_field)
    if labels is None:
        return []

    pieces = []

    for cluster_id in range(cluster_count):
        piece_crop = (labels == cluster_id).astype(np.uint8) * 255
        if not np.any(piece_crop):
            continue

        piece_area = int(np.count_nonzero(piece_crop))
        if piece_area < MIN_PIECE_AREA:
            continue

        piece_mask = np.zeros(image_shape, dtype=np.uint8)
        piece_mask[y : y + h, x : x + w] = piece_crop
        tight_bbox = mask_bbox(piece_mask)
        if tight_bbox is None:
            continue

        bx, by, bw, bh = tight_bbox
        fill_ratio = piece_area / float(max(bw * bh, 1))
        if fill_ratio < MIN_SPLIT_FILL_RATIO:
            continue

        pieces.append(
            {
                "mask": piece_mask,
                "bbox": tight_bbox,
                "area": piece_area,
            }
        )

    return pieces


def make_piece_masks(component, image_shape, max_area, max_bbox_dim):
    target_count = estimate_split_count(component, max_area, max_bbox_dim)
    if target_count <= 1:
        fallback_mask = component["mask"].astype(np.uint8) * 255
        return [{"mask": fallback_mask, "bbox": component["bbox"], "area": component["area"]}]

    pieces = split_component_shape(component, image_shape, target_count)
    if len(pieces) >= 2:
        return pieces

    fallback_mask = component["mask"].astype(np.uint8) * 255
    return [{"mask": fallback_mask, "bbox": component["bbox"], "area": component["area"]}]


def main():
    ensure_dir(DEBUG_DIR)
    clear_previous_outputs()

    maria_mask = load_image_gray(MASK_PATH)
    moon_gray = load_image_gray(GRAY_PATH)

    max_component_area = MAX_COMPONENT_AREA
    max_bbox_dim = MAX_BBOX_DIM

    source_components = connected_components(maria_mask)
    kept_components = [component for component in source_components if component["area"] >= MIN_PIECE_AREA]

    final_pieces = []
    overlay_regions = []
    split_count = 0

    for component in kept_components:
        oversized = component["area"] > max_component_area or max(component["bbox"][2:]) > max_bbox_dim
        component_pieces = make_piece_masks(component, maria_mask.shape, max_component_area, max_bbox_dim)
        if oversized and len(component_pieces) > 1:
            split_count += 1

        for piece in component_pieces:
            final_pieces.append((component, piece))

    metadata = []
    for index, (component, piece) in enumerate(final_pieces, start=1):
        piece_id = f"piece_{index:03d}"
        image_path = CORPUS_DIR / f"{piece_id}.png"
        mask_path = CORPUS_DIR / f"{piece_id}_mask.png"

        image_crop, mask_crop = crop_to_bbox(moon_gray, piece["mask"], piece["bbox"])
        save_image(image_path, image_crop)
        save_image(mask_path, mask_crop)

        x, y, w, h = piece["bbox"]
        metadata.append(
            {
                "id": piece_id,
                "image_path": image_path.as_posix(),
                "mask_path": mask_path.as_posix(),
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": int(piece["area"]),
                "width": int(w),
                "height": int(h),
                "source_component_id": int(component["label"]),
            }
        )

        overlay_regions.append(
            {
                "id": piece_id.split("_")[-1],
                "mask": piece["mask"] > 0,
                "centroid": [x + w / 2.0, y + h / 2.0],
            }
        )

    with INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    overlay = colorize_regions(moon_gray, overlay_regions, alpha=0.45, draw_labels=True)
    save_image(DEBUG_DIR / "corpus_regions_overlay.png", overlay)

    filtered_count = len(source_components) - len(kept_components)
    print(f"source components: {len(source_components)}")
    print(f"filtered tiny components: {filtered_count}")
    print(f"max component area: {max_component_area}")
    print(f"max bbox dim: {max_bbox_dim}")
    print(f"split oversized components: {split_count}")
    print(f"final pieces: {len(metadata)}")
    print(f"wrote index: {INDEX_PATH}")


if __name__ == "__main__":
    main()
