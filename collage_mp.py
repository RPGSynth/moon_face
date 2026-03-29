from __future__ import annotations

import atexit
import math
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np

from render_collage import composite_roi, compute_piece_roi, transform_image_roi, transform_mask_roi
from score_collage import weighted_l1_score_delta


_WORKER_STATE: dict[str, Any] = {}


def _dtype_from_name(name: str):
    return np.dtype(name)


def _attach_shared_ndarray(spec: dict[str, Any]):
    shm = SharedMemory(name=spec["name"])
    array = np.ndarray(tuple(spec["shape"]), dtype=_dtype_from_name(spec["dtype"]), buffer=shm.buf)
    return shm, array


def _primitive_view(index: int):
    images = _WORKER_STATE["primitive_images"]
    masks = _WORKER_STATE["primitive_masks"]
    dims = _WORKER_STATE["primitive_dims"][index]
    height = int(dims[0])
    width = int(dims[1])
    return images[index, :height, :width], masks[index, :height, :width]


def _clamp_fill(value: float):
    low, high = _WORKER_STATE["dark_fill_range"]
    return int(np.clip(int(round(value)), low, high))


def _mutation_parameters(accepted_count: int):
    piece_number = accepted_count + 1
    if piece_number <= 60:
        return {"position_sigma": 16.0, "scale_sigma": 0.14, "rotation_sigma": 24.0, "fill_step": 18}
    if piece_number <= 120:
        return {"position_sigma": 10.0, "scale_sigma": 0.10, "rotation_sigma": 16.0, "fill_step": 12}
    return {"position_sigma": 6.0, "scale_sigma": 0.06, "rotation_sigma": 9.0, "fill_step": 8}


def _clamp_scale(scale: float):
    return float(np.clip(scale, 0.03, 4.0))


def _candidate_fill_values(target_gray_roi, transformed_mask_roi, seed_fill=None, fill_step=16):
    coverage = transformed_mask_roi > 0.01
    if not np.any(coverage):
        return []

    mask_values = transformed_mask_roi[coverage].astype(np.float32)
    mean_value = float(np.sum(target_gray_roi[coverage] * mask_values) / max(np.sum(mask_values), 1e-6))
    values = []

    for candidate in (mean_value - fill_step, mean_value, mean_value + fill_step):
        clamped = _clamp_fill(candidate)
        if clamped not in values:
            values.append(clamped)

    if seed_fill is not None:
        for candidate in (seed_fill - fill_step, seed_fill, seed_fill + fill_step):
            clamped = _clamp_fill(candidate)
            if clamped not in values:
                values.append(clamped)
    return values


def _mutate_candidate(candidate: dict[str, Any], accepted_count: int, rng: np.random.Generator, canvas_size: int):
    params = _mutation_parameters(accepted_count)
    mutated = dict(candidate)

    mutated["x"] = float(np.clip(mutated["x"] + rng.normal(0.0, params["position_sigma"]), 0, canvas_size - 1))
    mutated["y"] = float(np.clip(mutated["y"] + rng.normal(0.0, params["position_sigma"]), 0, canvas_size - 1))
    mutated["scale"] = _clamp_scale(mutated["scale"] * float(np.exp(rng.normal(0.0, params["scale_sigma"]))))
    mutated["rotation_deg"] = float(((mutated["rotation_deg"] + rng.normal(0.0, params["rotation_sigma"]) + 180.0) % 360.0) - 180.0)
    mutated["fill_value"] = _clamp_fill(mutated["fill_value"] + rng.integers(-params["fill_step"], params["fill_step"] + 1))

    if rng.random() < 0.15:
        mutated["flip_x"] = not mutated["flip_x"]
    if rng.random() < 0.15:
        mutated["flip_y"] = not mutated["flip_y"]
    return mutated


def _evaluate_candidate(placement: dict[str, Any], current_score: float, accepted_count: int):
    canvas = _WORKER_STATE["current_canvas"]
    target_gray = _WORKER_STATE["target_gray"]
    target_weights = _WORKER_STATE["target_weights"]
    canvas_size = int(_WORKER_STATE["canvas_size"])
    piece_index = int(placement["piece_index"])

    primitive_image, primitive_mask = _primitive_view(piece_index)
    roi_bounds = compute_piece_roi(primitive_mask.shape, placement, (canvas_size, canvas_size))
    if roi_bounds is None:
        return None

    x0, y0, x1, y1 = roi_bounds
    transformed_mask_roi = transform_mask_roi(primitive_mask, placement, roi_bounds)
    if float(transformed_mask_roi.sum()) < 12.0:
        return None

    transformed_image_roi = transform_image_roi(primitive_image, placement, roi_bounds)
    current_canvas_roi = canvas[y0:y1, x0:x1]
    target_gray_roi = target_gray[y0:y1, x0:x1]
    target_weights_roi = target_weights[y0:y1, x0:x1]

    params = _mutation_parameters(accepted_count)
    fill_values = _candidate_fill_values(
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
            _WORKER_STATE["global_alpha"],
            use_piece_texture=_WORKER_STATE["use_piece_texture"],
        )
        score_delta = weighted_l1_score_delta(
            current_canvas_roi,
            trial_canvas_roi,
            target_gray_roi,
            target_weights_roi,
            total_pixels=target_weights.size,
        )
        trial_score = float(current_score + score_delta)

        if best_variant is None or trial_score < best_variant["score"]:
            best_variant = {
                "placement": {**placement, "fill_value": int(fill_value)},
                "score": trial_score,
                "score_delta": float(score_delta),
                "roi_bounds": [int(x0), int(y0), int(x1), int(y1)],
                "trial_canvas_roi": np.asarray(trial_canvas_roi, dtype=np.float32),
            }
    return best_variant


def _hill_climb_start(start_candidate: dict[str, Any], current_score: float, accepted_count: int, local_seed: int):
    rng = np.random.default_rng(local_seed)
    best = _evaluate_candidate(start_candidate, current_score, accepted_count)
    if best is None:
        return None

    hill_steps = int(_WORKER_STATE["hill_climb_steps"])
    mutations_per_step = int(_WORKER_STATE["mutations_per_step"])
    canvas_size = int(_WORKER_STATE["canvas_size"])

    for _ in range(hill_steps):
        step_best = best
        for _ in range(mutations_per_step):
            mutated = _mutate_candidate(best["placement"], accepted_count, rng, canvas_size)
            evaluated = _evaluate_candidate(mutated, current_score, accepted_count)
            if evaluated is not None and evaluated["score"] < step_best["score"]:
                step_best = evaluated
        best = step_best
    return best


def _worker_init(state_specs: dict[str, dict[str, Any]], static_config: dict[str, Any]):
    global _WORKER_STATE
    _WORKER_STATE = {
        "shms": [],
        "global_alpha": float(static_config["global_alpha"]),
        "use_piece_texture": bool(static_config["use_piece_texture"]),
        "dark_fill_range": tuple(static_config["dark_fill_range"]),
        "canvas_size": int(static_config["canvas_size"]),
        "hill_climb_steps": int(static_config["hill_climb_steps"]),
        "mutations_per_step": int(static_config["mutations_per_step"]),
    }

    for key, spec in state_specs.items():
        shm, array = _attach_shared_ndarray(spec)
        _WORKER_STATE["shms"].append(shm)
        _WORKER_STATE[key] = array


def _worker_cleanup():
    for shm in _WORKER_STATE.get("shms", []):
        try:
            shm.close()
        except FileNotFoundError:
            pass


def evaluate_start_chunk(task: dict[str, Any]):
    current_score = float(task["current_score"])
    accepted_count = int(task["accepted_count"])
    chunk_seed = int(task["chunk_seed"])
    starts = task["starts"]

    best_candidate = None
    for index, start in enumerate(starts):
        local_seed = chunk_seed + index
        climbed = _hill_climb_start(start, current_score, accepted_count, local_seed)
        if climbed is not None and climbed["score"] < current_score:
            if best_candidate is None or climbed["score"] < best_candidate["score"]:
                best_candidate = climbed
    return best_candidate


class SharedArrayManager:
    def __init__(self):
        self._shms: list[SharedMemory] = []

    def create(self, array: np.ndarray):
        data = np.ascontiguousarray(array)
        shm = SharedMemory(create=True, size=data.nbytes)
        shared = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        shared[...] = data
        self._shms.append(shm)
        return {
            "name": shm.name,
            "shape": list(data.shape),
            "dtype": data.dtype.str,
        }, shared

    def close(self):
        for shm in self._shms:
            try:
                shm.close()
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
        self._shms.clear()


class MultiprocessingSearchBackend:
    def __init__(
        self,
        current_canvas: np.ndarray,
        target_gray: np.ndarray,
        target_weights: np.ndarray,
        primitive_images: np.ndarray,
        primitive_masks: np.ndarray,
        primitive_dims: np.ndarray,
        *,
        canvas_size: int,
        global_alpha: float,
        use_piece_texture: bool,
        dark_fill_range: tuple[int, int],
        hill_climb_steps: int,
        mutations_per_step: int,
        num_workers: int,
    ):
        self._shared = SharedArrayManager()
        self.state_specs: dict[str, dict[str, Any]] = {}
        self.state_arrays: dict[str, np.ndarray] = {}

        for key, array in {
            "current_canvas": current_canvas,
            "target_gray": target_gray,
            "target_weights": target_weights,
            "primitive_images": primitive_images,
            "primitive_masks": primitive_masks,
            "primitive_dims": primitive_dims,
        }.items():
            spec, shared = self._shared.create(array)
            self.state_specs[key] = spec
            self.state_arrays[key] = shared

        static_config = {
            "canvas_size": int(canvas_size),
            "global_alpha": float(global_alpha),
            "use_piece_texture": bool(use_piece_texture),
            "dark_fill_range": tuple(int(value) for value in dark_fill_range),
            "hill_climb_steps": int(hill_climb_steps),
            "mutations_per_step": int(mutations_per_step),
        }

        ctx = get_context("spawn")
        self.executor = ProcessPoolExecutor(
            max_workers=max(1, int(num_workers)),
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(self.state_specs, static_config),
        )
        atexit.register(self.close)

    def update_canvas(self, canvas: np.ndarray):
        self.state_arrays["current_canvas"][...] = np.asarray(canvas, dtype=np.float32)

    def evaluate_chunks(self, tasks: list[dict[str, Any]]):
        futures = [self.executor.submit(evaluate_start_chunk, task) for task in tasks]
        return [future.result() for future in futures]

    def close(self):
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        shared = getattr(self, "_shared", None)
        if shared is not None:
            shared.close()
            self._shared = None


def default_num_workers():
    return max(1, min(8, (os.cpu_count() or 2) - 1))
