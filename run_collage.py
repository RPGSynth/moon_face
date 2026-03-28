import json
from pathlib import Path

import cv2
import numpy as np

from prepare_corpus import prepare_primitives, write_prepared_outputs
from render_collage import composite_mask, load_primitives, transform_image, transform_mask
from score_collage import load_target_assets, weighted_l1_score
from utils import ensure_dir, save_image, to_uint8


TARGET_STEM = "Barack"
CANVAS_SIZE = 320
GLOBAL_ALPHA = 0.35
USE_PIECE_TEXTURE = True
MAX_ACCEPTED_PIECES = 500
CANDIDATES_PER_STEP = 350
PROGRESS_EVERY = 10
MAX_STALLED_STEPS = 20
DARK_FILL_RANGE = (16, 144)
RNG_SEED = 7

PROGRESS_DIR = Path("out/progress")
FINAL_DIR = Path("out/final")


def size_fraction_range(accepted_count):
    piece_number = accepted_count + 1
    if piece_number <= 60:
        return 0.22, 0.42
    if piece_number <= 120:
        return 0.12, 0.26
    return 0.06, 0.16


def sample_anchor(anchor_probs, rng):
    index = int(rng.choice(anchor_probs.size, p=anchor_probs))
    y, x = divmod(index, CANVAS_SIZE)
    return x, y


def clamp_fill(value):
    low, high = DARK_FILL_RANGE
    return int(np.clip(int(round(value)), low, high))


def candidate_fill_values(target_gray, transformed_mask):
    coverage = transformed_mask > 0.01
    if not np.any(coverage):
        return []

    mask_values = transformed_mask[coverage].astype(np.float32)
    mean_value = float(np.sum(target_gray[coverage] * mask_values) / max(np.sum(mask_values), 1e-6))
    base = clamp_fill(mean_value)

    values = []
    for candidate in (base - 16, base, base + 16):
        clamped = clamp_fill(candidate)
        if clamped not in values:
            values.append(clamped)
    return values


def sample_candidate_geometry(primitives_list, target_weights, rng, accepted_count):
    primitive = primitives_list[int(rng.integers(0, len(primitives_list)))]
    fraction_min, fraction_max = size_fraction_range(accepted_count)
    target_fraction = float(rng.uniform(fraction_min, fraction_max))
    target_max_dim = target_fraction * CANVAS_SIZE
    scale = float(target_max_dim / max(int(primitive["width"]), int(primitive["height"]), 1))

    anchor_probs = target_weights.reshape(-1).astype(np.float64)
    anchor_probs = anchor_probs / anchor_probs.sum()
    anchor_x, anchor_y = sample_anchor(anchor_probs, rng)

    jitter = max(6, int(target_max_dim * 0.35))
    x = float(np.clip(anchor_x + rng.normal(0.0, jitter), 0, CANVAS_SIZE - 1))
    y = float(np.clip(anchor_y + rng.normal(0.0, jitter), 0, CANVAS_SIZE - 1))

    return {
        "piece_id": primitive["id"],
        "x": x,
        "y": y,
        "scale": scale,
        "rotation_deg": float(rng.uniform(-180.0, 180.0)),
        "flip_x": bool(rng.integers(0, 2)),
        "flip_y": bool(rng.integers(0, 2)),
        "fill_value": DARK_FILL_RANGE[0],
    }


def evaluate_best_candidate(primitives, primitives_list, current_canvas, current_score, target_gray, target_weights, rng, accepted_count):
    best_placement = None
    best_render = None
    best_score = current_score

    for _ in range(CANDIDATES_PER_STEP):
        placement = sample_candidate_geometry(primitives_list, target_weights, rng, accepted_count)
        primitive = primitives[placement["piece_id"]]
        transformed_mask = transform_mask(primitive["mask"], placement, (CANVAS_SIZE, CANVAS_SIZE))
        transformed_image = transform_image(primitive["image"], placement, (CANVAS_SIZE, CANVAS_SIZE))

        if float(transformed_mask.sum()) < 12.0:
            continue

        fill_values = candidate_fill_values(target_gray, transformed_mask)
        for fill_value in fill_values:
            candidate = {**placement, "fill_value": fill_value}
            trial_render = composite_mask(
                current_canvas,
                transformed_mask,
                transformed_image,
                fill_value,
                GLOBAL_ALPHA,
                use_piece_texture=USE_PIECE_TEXTURE,
            )
            trial_score = weighted_l1_score(trial_render, target_gray, target_weights)
            if trial_score < best_score:
                best_score = trial_score
                best_render = trial_render
                best_placement = candidate

    return best_placement, best_render, best_score


def to_rgb(gray_image):
    image = to_uint8(gray_image)
    return np.stack([image, image, image], axis=-1)


def labeled_tile(image_rgb, label):
    tile = image_rgb.copy()
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (255, 255, 255), 1)
    cv2.putText(tile, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(tile, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 1, cv2.LINE_AA)
    return tile


def progress_panel(rendered, target_gray, target_weights):
    weights_preview = cv2.applyColorMap(to_uint8(target_weights * 255.0), cv2.COLORMAP_INFERNO)
    weights_preview = cv2.cvtColor(weights_preview, cv2.COLOR_BGR2RGB)
    diff = np.abs(rendered.astype(np.float32) - target_gray.astype(np.float32))
    diff_preview = cv2.applyColorMap(to_uint8(np.clip(diff, 0, 255)), cv2.COLORMAP_MAGMA)
    diff_preview = cv2.cvtColor(diff_preview, cv2.COLOR_BGR2RGB)

    top = np.concatenate([labeled_tile(to_rgb(rendered), "collage"), labeled_tile(to_rgb(target_gray), "target")], axis=1)
    bottom = np.concatenate([labeled_tile(weights_preview, "weights"), labeled_tile(diff_preview, "difference")], axis=1)
    return np.concatenate([top, bottom], axis=0)


def save_progress_outputs(stem, accepted_count, canvas, score_history):
    progress_root = PROGRESS_DIR / stem
    ensure_dir(progress_root)

    panel = progress_panel(canvas["rendered"], canvas["target_gray"], canvas["target_weights"])
    save_image(progress_root / f"step_{accepted_count:04d}.png", panel)

    history_path = progress_root / "history.json"
    history_path.write_text(json.dumps(score_history, indent=2))


def save_final_outputs(stem, rendered, placements):
    ensure_dir(FINAL_DIR)
    save_image(FINAL_DIR / f"{stem}_collage.png", rendered)
    (FINAL_DIR / f"{stem}_placements.json").write_text(json.dumps(placements, indent=2))


def main():
    ensure_dir(PROGRESS_DIR)
    ensure_dir(FINAL_DIR)

    records = prepare_primitives()
    write_prepared_outputs(records)

    primitives = load_primitives()
    primitives_list = sorted(primitives.values(), key=lambda item: item["id"])
    target_gray, target_weights = load_target_assets(TARGET_STEM, CANVAS_SIZE)

    rng = np.random.default_rng(RNG_SEED)
    current_canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), 255.0, dtype=np.float32)
    current_score = weighted_l1_score(current_canvas, target_gray, target_weights)
    placements = []
    score_history = [{"accepted": 0, "score": current_score}]
    stalled_steps = 0

    while len(placements) < MAX_ACCEPTED_PIECES and stalled_steps < MAX_STALLED_STEPS:
        best_placement, best_render, best_score = evaluate_best_candidate(
            primitives,
            primitives_list,
            current_canvas,
            current_score,
            target_gray,
            target_weights,
            rng,
            len(placements),
        )

        if best_placement is None:
            stalled_steps += 1
            continue

        placements.append(best_placement)
        current_canvas = best_render
        current_score = best_score
        stalled_steps = 0
        score_history.append({"accepted": len(placements), "score": current_score, "piece_id": best_placement["piece_id"]})

        print(f"accepted {len(placements):03d}: {best_placement['piece_id']} score={current_score:.4f}")

        if len(placements) % PROGRESS_EVERY == 0:
            save_progress_outputs(
                TARGET_STEM,
                len(placements),
                {"rendered": current_canvas, "target_gray": target_gray, "target_weights": target_weights},
                score_history,
            )

    save_progress_outputs(
        TARGET_STEM,
        len(placements),
        {"rendered": current_canvas, "target_gray": target_gray, "target_weights": target_weights},
        score_history,
    )
    save_final_outputs(TARGET_STEM, current_canvas, placements)

    print(f"final accepted pieces: {len(placements)}")
    print(f"final score: {current_score:.4f}")
    print(f"progress dir: {PROGRESS_DIR / TARGET_STEM}")
    print(f"final image: {FINAL_DIR / f'{TARGET_STEM}_collage.png'}")


if __name__ == "__main__":
    main()
