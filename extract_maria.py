from pathlib import Path

import cv2
import numpy as np

from utils import colorize_regions, connected_components, ensure_dir, filter_components_by_area, largest_component, load_image_gray, save_image


# Source preference: use the tonally managed NASA mosaic first.
SOURCE_CANDIDATES = ["data/source/moon_nasa.png", "data/source/moon.tiff", "data/source/moon.tif"]

# Disc mask tuning: only touch these if the moon/background split is wrong.
BLUR_KERNEL = 9
DISC_THRESHOLD = 18

# Maria tuning:
# - MARIA_BLUR_KERNEL: higher = broader/smoother maria, lower = more detail/noise
# - MARIA_PERCENTILE: higher = keep more dark regions, lower = keep fewer/darker ones
# - MORPH_KERNEL: higher = connect/fill more, lower = preserve separation
# - MIN_REGION_AREA: higher = remove more tiny artifacts, lower = keep smaller regions
MARIA_BLUR_KERNEL = 30
MARIA_PERCENTILE = 40
MORPH_KERNEL = 8
MIN_REGION_AREA = 0

MASK_DIR = Path("data/masks")
DEBUG_DIR = Path("out/debug")
EXTRACT_DEBUG_DIR = DEBUG_DIR / "extract"


def odd_kernel(size):
    return size if size % 2 == 1 else size + 1


def fill_holes(mask):
    mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    h, w = padded.shape
    flood = padded.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(padded, holes)
    return filled[1:-1, 1:-1]


def draw_outline(image_rgb, mask, color, thickness=2):
    outlined = image_rgb.copy()
    contour_mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(outlined, contours, -1, color, thickness)
    return outlined


def find_source():
    for candidate in SOURCE_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    joined = ", ".join(SOURCE_CANDIDATES)
    raise FileNotFoundError(f"No moon source image found. Checked: {joined}")


def clear_extract_debug():
    ensure_dir(EXTRACT_DEBUG_DIR)
    for path in EXTRACT_DEBUG_DIR.glob("*.png"):
        path.unlink()


def main():
    ensure_dir(MASK_DIR)
    ensure_dir(DEBUG_DIR)
    clear_extract_debug()

    source_path = find_source()
    gray = load_image_gray(source_path)
    save_image(MASK_DIR / "moon_gray.png", gray)

    blur_size = odd_kernel(BLUR_KERNEL)
    morph_size = odd_kernel(MORPH_KERNEL)
    maria_blur_size = odd_kernel(MARIA_BLUR_KERNEL)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))

    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    disc_seed = (blurred > DISC_THRESHOLD).astype(np.uint8) * 255
    disc_mask = largest_component(disc_seed)
    disc_mask = fill_holes(disc_mask)
    disc_mask = cv2.morphologyEx(disc_mask, cv2.MORPH_CLOSE, morph_kernel)
    disc_mask = fill_holes(disc_mask)
    save_image(MASK_DIR / "moon_disc_mask.png", disc_mask)

    disc_pixels = gray[disc_mask > 0]
    if disc_pixels.size == 0:
        raise RuntimeError("Moon disc mask is empty. Lower DISC_THRESHOLD or check the source image.")

    disc_median = int(np.median(disc_pixels))
    maria_base = gray.copy()
    maria_base[disc_mask == 0] = disc_median
    maria_blurred = cv2.GaussianBlur(maria_base, (maria_blur_size, maria_blur_size), 0)
    maria_search_mask = cv2.erode(disc_mask, morph_kernel, iterations=10)
    maria_blurred[maria_search_mask == 0] = 0

    maria_threshold = float(np.percentile(maria_blurred[maria_search_mask > 0], MARIA_PERCENTILE))
    maria_candidate_raw = ((maria_blurred <= maria_threshold) & (maria_search_mask > 0)).astype(np.uint8) * 255
    maria_candidate = maria_candidate_raw.copy()
    maria_candidate = cv2.morphologyEx(maria_candidate, cv2.MORPH_CLOSE, morph_kernel)

    min_region_area = max(MIN_REGION_AREA, int(np.count_nonzero(disc_mask) * 0.0002))
    maria_mask = filter_components_by_area(maria_candidate, min_region_area)
    save_image(MASK_DIR / "maria_mask.png", maria_mask)

    components = connected_components(maria_mask)

    threshold_preview = colorize_regions(maria_blurred, [{"mask": maria_candidate > 0, "label": "raw"}], alpha=0.42, draw_labels=False)
    threshold_preview = draw_outline(threshold_preview, disc_mask, (66, 165, 245))
    threshold_preview = draw_outline(threshold_preview, maria_search_mask, (255, 255, 255), thickness=1)
    cleaned_overlay = colorize_regions(gray, [{"mask": maria_mask > 0, "label": "clean"}], alpha=0.45, draw_labels=False)
    maria_overlay = draw_outline(cleaned_overlay, disc_mask, (255, 255, 255), thickness=1)
    maria_regions_colored = colorize_regions(gray, components, alpha=0.45, draw_labels=True)
    maria_regions_colored = draw_outline(maria_regions_colored, disc_mask, (255, 255, 255), thickness=1)

    save_image(DEBUG_DIR / "threshold_preview.png", threshold_preview)
    save_image(DEBUG_DIR / "maria_mask_cleaned.png", maria_mask)
    save_image(DEBUG_DIR / "maria_overlay.png", maria_overlay)
    save_image(DEBUG_DIR / "maria_regions_colored.png", maria_regions_colored)

    save_image(EXTRACT_DEBUG_DIR / "01_gray.png", gray)
    save_image(EXTRACT_DEBUG_DIR / "02_blurred.png", blurred)
    save_image(EXTRACT_DEBUG_DIR / "03_disc_seed.png", disc_seed)
    save_image(EXTRACT_DEBUG_DIR / "04_disc_mask.png", disc_mask)
    save_image(EXTRACT_DEBUG_DIR / "05_maria_blurred.png", maria_blurred)
    save_image(EXTRACT_DEBUG_DIR / "06_maria_search_mask.png", maria_search_mask)
    save_image(EXTRACT_DEBUG_DIR / "07_maria_candidate_raw.png", maria_candidate_raw)
    save_image(EXTRACT_DEBUG_DIR / "08_maria_candidate_closed.png", maria_candidate)
    save_image(EXTRACT_DEBUG_DIR / "09_maria_mask_cleaned.png", maria_mask)

    disc_area = int(np.count_nonzero(disc_mask))
    maria_area = int(np.count_nonzero(maria_mask))
    print(f"source: {source_path}")
    print(f"maria threshold: {maria_threshold:.2f}")
    print(f"disc area: {disc_area}")
    print(f"maria area: {maria_area}")
    print(f"kept regions: {len(components)}")
    print(f"extract debug dir: {EXTRACT_DEBUG_DIR}")


if __name__ == "__main__":
    main()
