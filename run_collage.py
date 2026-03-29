import json
from pathlib import Path

import cv2
import numpy as np

from collage_mp import MultiprocessingSearchBackend, default_num_workers
from prepare_corpus import prepare_primitives, write_prepared_outputs
from render_collage import composite_roi, compute_piece_roi, load_primitives, transform_image_roi, transform_mask_roi
from score_collage import load_target_assets, weighted_l1_score, weighted_l1_score_delta
from utils import ensure_dir, save_image, to_uint8


TARGET_STEM = "Barack"
CANVAS_SIZE = 320
GLOBAL_ALPHA = 0.35
USE_PIECE_TEXTURE = True
ENABLE_MULTIPROCESSING = True
NUM_WORKERS = default_num_workers()
MAX_ACCEPTED_PIECES = 1000
RANDOM_STARTS_PER_STEP = 40
STARTS_PER_WORKER_CHUNK = 8
HILL_CLIMB_STEPS = 6
MUTATIONS_PER_STEP = 4
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


def mutation_parameters(accepted_count):
    piece_number = accepted_count + 1
    if piece_number <= 60:
        return {"position_sigma": 16.0, "scale_sigma": 0.14, "rotation_sigma": 24.0, "fill_step": 18}
    if piece_number <= 120:
        return {"position_sigma": 10.0, "scale_sigma": 0.10, "rotation_sigma": 16.0, "fill_step": 12}
    return {"position_sigma": 6.0, "scale_sigma": 0.06, "rotation_sigma": 9.0, "fill_step": 8}


def sample_anchor(anchor_probs, rng):
    index = int(rng.choice(anchor_probs.size, p=anchor_probs))
    y, x = divmod(index, CANVAS_SIZE)
    return x, y


def clamp_fill(value):
    low, high = DARK_FILL_RANGE
    return int(np.clip(int(round(value)), low, high))


def clamp_scale(scale, accepted_count):
    _ = accepted_count
    return float(np.clip(scale, 0.03, 4.0))


def candidate_fill_values(target_gray_roi, transformed_mask, seed_fill=None, fill_step=16):
    coverage = transformed_mask > 0.01
    if not np.any(coverage):
        return []

    mask_values = transformed_mask[coverage].astype(np.float32)
    mean_value = float(np.sum(target_gray_roi[coverage] * mask_values) / max(np.sum(mask_values), 1e-6))
    values = []

    for candidate in (mean_value - fill_step, mean_value, mean_value + fill_step):
        clamped = clamp_fill(candidate)
        if clamped not in values:
            values.append(clamped)

    if seed_fill is not None:
        for candidate in (seed_fill - fill_step, seed_fill, seed_fill + fill_step):
            clamped = clamp_fill(candidate)
            if clamped not in values:
                values.append(clamped)
    return values


def sample_candidate_geometry(primitives_list, anchor_probs, rng, accepted_count):
    primitive = primitives_list[int(rng.integers(0, len(primitives_list)))]
    fraction_min, fraction_max = size_fraction_range(accepted_count)
    target_fraction = float(rng.uniform(fraction_min, fraction_max))
    target_max_dim = target_fraction * CANVAS_SIZE
    scale = float(target_max_dim / max(int(primitive["width"]), int(primitive["height"]), 1))

    anchor_x, anchor_y = sample_anchor(anchor_probs, rng)
    jitter = max(6, int(target_max_dim * 0.35))
    x = float(np.clip(anchor_x + rng.normal(0.0, jitter), 0, CANVAS_SIZE - 1))
    y = float(np.clip(anchor_y + rng.normal(0.0, jitter), 0, CANVAS_SIZE - 1))

    return {
        "piece_id": primitive["id"],
        "piece_index": int(primitive["piece_index"]),
        "x": x,
        "y": y,
        "scale": scale,
        "rotation_deg": float(rng.uniform(-180.0, 180.0)),
        "flip_x": bool(rng.integers(0, 2)),
        "flip_y": bool(rng.integers(0, 2)),
        "fill_value": DARK_FILL_RANGE[0],
    }


def mutate_candidate(candidate, accepted_count, rng):
    params = mutation_parameters(accepted_count)
    mutated = dict(candidate)

    mutated["x"] = float(np.clip(mutated["x"] + rng.normal(0.0, params["position_sigma"]), 0, CANVAS_SIZE - 1))
    mutated["y"] = float(np.clip(mutated["y"] + rng.normal(0.0, params["position_sigma"]), 0, CANVAS_SIZE - 1))
    mutated["scale"] = clamp_scale(mutated["scale"] * float(np.exp(rng.normal(0.0, params["scale_sigma"]))), accepted_count)
    mutated["rotation_deg"] = float(((mutated["rotation_deg"] + rng.normal(0.0, params["rotation_sigma"]) + 180.0) % 360.0) - 180.0)
    mutated["fill_value"] = clamp_fill(mutated["fill_value"] + rng.integers(-params["fill_step"], params["fill_step"] + 1))

    if rng.random() < 0.15:
        mutated["flip_x"] = not mutated["flip_x"]
    if rng.random() < 0.15:
        mutated["flip_y"] = not mutated["flip_y"]
    return mutated


def evaluate_candidate(primitives, current_canvas, current_score, target_gray, target_weights, placement, accepted_count):
    primitive = primitives[placement["piece_id"]]
    roi_bounds = compute_piece_roi(primitive["mask"].shape, placement, current_canvas.shape)
    if roi_bounds is None:
        return None

    x0, y0, x1, y1 = roi_bounds
    transformed_mask_roi = transform_mask_roi(primitive["mask"], placement, roi_bounds)
    if float(transformed_mask_roi.sum()) < 12.0:
        return None

    transformed_image_roi = transform_image_roi(primitive["image"], placement, roi_bounds)
    current_canvas_roi = current_canvas[y0:y1, x0:x1]
    target_gray_roi = target_gray[y0:y1, x0:x1]
    target_weights_roi = target_weights[y0:y1, x0:x1]

    params = mutation_parameters(accepted_count)
    fill_values = candidate_fill_values(
        target_gray_roi,
        transformed_mask_roi,
        seed_fill=placement.get("fill_value"),
        fill_step=params["fill_step"],
    )

    best_variant = None
    for fill_value in fill_values:
        trial_canvas_roi = composite_roi(
            current_canvas_roi,
            transformed_mask_roi,
            transformed_image_roi,
            fill_value,
            GLOBAL_ALPHA,
            use_piece_texture=USE_PIECE_TEXTURE,
        )
        score_delta = weighted_l1_score_delta(
            current_canvas_roi,
            trial_canvas_roi,
            target_gray_roi,
            target_weights_roi,
            total_pixels=target_weights.size,
        )
        trial_score = current_score + score_delta

        if best_variant is None or trial_score < best_variant["score"]:
            best_variant = {
                "placement": {**placement, "fill_value": fill_value},
                "score": trial_score,
                "score_delta": score_delta,
                "roi_bounds": [int(x0), int(y0), int(x1), int(y1)],
                "trial_canvas_roi": trial_canvas_roi,
            }
    return best_variant


def hill_climb_candidate(primitives, current_canvas, current_score, target_gray, target_weights, start_candidate, rng, accepted_count):
    best = evaluate_candidate(primitives, current_canvas, current_score, target_gray, target_weights, start_candidate, accepted_count)
    if best is None:
        return None

    for _ in range(HILL_CLIMB_STEPS):
        step_best = best
        for _ in range(MUTATIONS_PER_STEP):
            mutated_placement = mutate_candidate(best["placement"], accepted_count, rng)
            evaluated = evaluate_candidate(primitives, current_canvas, current_score, target_gray, target_weights, mutated_placement, accepted_count)
            if evaluated is not None and evaluated["score"] < step_best["score"]:
                step_best = evaluated
        best = step_best
    return best


def generate_random_starts(primitives_list, target_weights, rng, accepted_count):
    anchor_probs = target_weights.reshape(-1).astype(np.float64)
    anchor_probs = anchor_probs / anchor_probs.sum()
    return [sample_candidate_geometry(primitives_list, anchor_probs, rng, accepted_count) for _ in range(RANDOM_STARTS_PER_STEP)]


def chunk_starts(starts, chunk_size):
    chunk_size = max(1, int(chunk_size))
    return [starts[index : index + chunk_size] for index in range(0, len(starts), chunk_size)]


def reduce_best_candidate(candidates, current_score):
    best_candidate = None
    for candidate in candidates:
        if candidate is None or candidate["score"] >= current_score:
            continue
        if best_candidate is None or candidate["score"] < best_candidate["score"]:
            best_candidate = candidate
    return best_candidate


def evaluate_start_chunk_sequential(primitives, current_canvas, current_score, target_gray, target_weights, starts, chunk_seed, accepted_count):
    best_candidate = None
    for index, start_candidate in enumerate(starts):
        local_seed = int(chunk_seed + index)
        local_rng = np.random.default_rng(local_seed)
        climbed = hill_climb_candidate(primitives, current_canvas, current_score, target_gray, target_weights, start_candidate, local_rng, accepted_count)
        if climbed is not None and climbed["score"] < current_score:
            if best_candidate is None or climbed["score"] < best_candidate["score"]:
                best_candidate = climbed
    return best_candidate


def prepare_backend_tasks(starts, rng, accepted_count, current_score):
    tasks = []
    for chunk in chunk_starts(starts, STARTS_PER_WORKER_CHUNK):
        tasks.append(
            {
                "starts": chunk,
                "chunk_seed": int(rng.integers(0, 2**31 - 1)),
                "accepted_count": int(accepted_count),
                "current_score": float(current_score),
            }
        )
    return tasks


def evaluate_best_candidate_sequential(primitives, current_canvas, current_score, target_gray, target_weights, starts, tasks, accepted_count):
    _ = accepted_count
    results = [
        evaluate_start_chunk_sequential(
            primitives,
            current_canvas,
            current_score,
            target_gray,
            target_weights,
            task["starts"],
            task["chunk_seed"],
            task["accepted_count"],
        )
        for task in tasks
    ]
    return reduce_best_candidate(results, current_score)


def assign_piece_indices(primitives):
    primitive_list = sorted(primitives.values(), key=lambda item: item["id"])
    for index, item in enumerate(primitive_list):
        item["piece_index"] = index
    return primitive_list


def build_primitive_tensors(primitives):
    primitive_list = assign_piece_indices(primitives)
    max_height = max(int(item["height"]) for item in primitive_list)
    max_width = max(int(item["width"]) for item in primitive_list)

    primitive_images = np.zeros((len(primitive_list), max_height, max_width), dtype=np.float32)
    primitive_masks = np.zeros((len(primitive_list), max_height, max_width), dtype=np.float32)
    primitive_dims = np.zeros((len(primitive_list), 2), dtype=np.int32)

    for index, item in enumerate(primitive_list):
        height = int(item["height"])
        width = int(item["width"])
        primitive_images[index, :height, :width] = item["image"]
        primitive_masks[index, :height, :width] = item["mask"]
        primitive_dims[index] = [height, width]
    return primitive_images, primitive_masks, primitive_dims


def maybe_create_mp_backend(primitives, current_canvas, target_gray, target_weights):
    if not ENABLE_MULTIPROCESSING or NUM_WORKERS <= 1:
        return None

    primitive_images, primitive_masks, primitive_dims = build_primitive_tensors(primitives)
    return MultiprocessingSearchBackend(
        current_canvas=current_canvas,
        target_gray=target_gray,
        target_weights=target_weights,
        primitive_images=primitive_images,
        primitive_masks=primitive_masks,
        primitive_dims=primitive_dims,
        canvas_size=CANVAS_SIZE,
        global_alpha=GLOBAL_ALPHA,
        use_piece_texture=USE_PIECE_TEXTURE,
        dark_fill_range=DARK_FILL_RANGE,
        hill_climb_steps=HILL_CLIMB_STEPS,
        mutations_per_step=MUTATIONS_PER_STEP,
        num_workers=NUM_WORKERS,
    )


def evaluate_best_candidate(primitives, primitives_list, current_canvas, current_score, target_gray, target_weights, rng, accepted_count, mp_backend=None):
    starts = generate_random_starts(primitives_list, target_weights, rng, accepted_count)
    tasks = prepare_backend_tasks(starts, rng, accepted_count, current_score)

    if mp_backend is None:
        return evaluate_best_candidate_sequential(
            primitives,
            current_canvas,
            current_score,
            target_gray,
            target_weights,
            starts,
            tasks,
            accepted_count,
        )

    mp_backend.update_canvas(current_canvas)
    results = mp_backend.evaluate_chunks(tasks)
    return reduce_best_candidate(results, current_score)


def apply_accepted_candidate(current_canvas, candidate):
    x0, y0, x1, y1 = candidate["roi_bounds"]
    current_canvas[y0:y1, x0:x1] = candidate["trial_canvas_roi"]
    return current_canvas


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
    (progress_root / "history.json").write_text(json.dumps(score_history, indent=2))


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
    primitive_list = assign_piece_indices(primitives)
    target_gray, target_weights = load_target_assets(TARGET_STEM, CANVAS_SIZE)

    rng = np.random.default_rng(RNG_SEED)
    current_canvas = np.full((CANVAS_SIZE, CANVAS_SIZE), 255.0, dtype=np.float32)
    current_score = weighted_l1_score(current_canvas, target_gray, target_weights)
    placements = []
    score_history = [{"accepted": 0, "score": current_score}]
    stalled_steps = 0
    mp_backend = maybe_create_mp_backend(primitives, current_canvas, target_gray, target_weights)

    try:
        while len(placements) < MAX_ACCEPTED_PIECES and stalled_steps < MAX_STALLED_STEPS:
            best_candidate = evaluate_best_candidate(
                primitives,
                primitive_list,
                current_canvas,
                current_score,
                target_gray,
                target_weights,
                rng,
                len(placements),
                mp_backend=mp_backend,
            )

            if best_candidate is None:
                stalled_steps += 1
                continue

            placements.append(best_candidate["placement"])
            current_canvas = apply_accepted_candidate(current_canvas, best_candidate)
            current_score = best_candidate["score"]
            stalled_steps = 0
            score_history.append(
                {
                    "accepted": len(placements),
                    "score": current_score,
                    "piece_id": best_candidate["placement"]["piece_id"],
                }
            )

            print(f"accepted {len(placements):03d}: {best_candidate['placement']['piece_id']} score={current_score:.4f}")

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
    finally:
        if mp_backend is not None:
            mp_backend.close()


if __name__ == "__main__":
    main()
