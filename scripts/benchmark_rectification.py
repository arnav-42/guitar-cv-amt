#!/usr/bin/env python3
"""Run deterministic synthetic geometry benchmarks for fretboard rectifiers.

Examples:
    python scripts/benchmark_rectification.py --output-dir tmp/rectification-benchmark
    python scripts/benchmark_rectification.py --output-dir tmp/bench --cases perspective hand_occlusion

The score separates regularity of *known ground-truth nodes after a warp* from
reprojection accuracy of any nodes estimated by the rectifier.  The former is
available for the legacy homography baseline; the latter is emitted only when
a method exposes an ordered estimated node lattice and its target coordinates.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.synthetic_fretboard import SyntheticFretboard, make_cases  # noqa: E402


DEFAULT_CASES = [
    "clean_frontal",
    "perspective",
    "rotated_negative_steep",
    "hand_occlusion",
    "mask_notch_spur",
    "partial_longitudinal_crop",
    "blank_no_evidence",
]
OUTPUT_SIZE = (1024, 256)


def _as_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)
    except Exception:
        return None


def _map_with_homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    source_shape = np.asarray(points).shape
    projected = cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(1, -1, 2),
        np.asarray(matrix, dtype=np.float64),
    )[0]
    return projected.reshape(source_shape)


def _regularity_metrics(
    mapped_nodes: np.ndarray,
    visible: np.ndarray,
    output_size: Tuple[int, int],
) -> Dict[str, Any]:
    """Measure how known strings/frets behave after a supplied image mapping."""
    width, height = output_size
    nodes = np.asarray(mapped_nodes, dtype=np.float64)
    if nodes.ndim != 3 or nodes.shape[-1] != 2:
        return {"available": False, "reason": f"expected (frets, strings, 2), got {nodes.shape}"}
    valid = np.asarray(visible, dtype=bool) & np.isfinite(nodes).all(axis=-1)
    if not valid.any():
        return {"available": False, "reason": "no visible finite ground-truth nodes"}

    # A string traverses rows (fret stations), while a fret traverses columns
    # (strings). Score deviations around each path's own mean, pooling all
    # available point residuals. That works for partially cropped/occluded
    # scenes and does not depend on how many fret stations a method predicts.
    string_residuals = []
    for string_index in range(nodes.shape[1]):
        keep = valid[:, string_index]
        if np.count_nonzero(keep) >= 2:
            y = nodes[keep, string_index, 1]
            string_residuals.extend((y - y.mean()).tolist())
    fret_residuals = []
    for fret_index in range(nodes.shape[0]):
        keep = valid[fret_index]
        if np.count_nonzero(keep) >= 2:
            x = nodes[fret_index, keep, 0]
            fret_residuals.extend((x - x.mean()).tolist())

    # Additive lattice fit is the two-dimensional residual from per-string and
    # per-fret coordinates. This is useful even when the method supplies no
    # detected-node assignments, but is kept distinct from GT reprojection RMSE.
    x_residuals, y_residuals = [], []
    for fret_index in range(nodes.shape[0]):
        keep = valid[fret_index]
        if np.count_nonzero(keep) >= 2:
            x_residuals.extend((nodes[fret_index, keep, 0] - nodes[fret_index, keep, 0].mean()).tolist())
    for string_index in range(nodes.shape[1]):
        keep = valid[:, string_index]
        if np.count_nonzero(keep) >= 2:
            y_residuals.extend((nodes[keep, string_index, 1] - nodes[keep, string_index, 1].mean()).tolist())

    def rms(values):
        return float(np.sqrt(np.mean(np.square(values)))) if values else None

    joint = np.concatenate((np.asarray(x_residuals), np.asarray(y_residuals))) if x_residuals or y_residuals else np.asarray([])
    # Normalize image-axis errors independently, so results remain readable
    # if a caller changes the requested output dimensions.
    string_rms = rms(string_residuals)
    fret_rms = rms(fret_residuals)
    joint_rmse = float(np.sqrt(np.mean(np.square(joint)))) if joint.size else None
    # Known consecutive physical frets should have equal increments in the
    # requested uniform atlas, even though they are not equally spaced on a
    # real guitar. Normalize differences by station-index gaps after occlusion.
    station_ids = np.flatnonzero(np.any(valid, axis=1))
    station_means = np.array([nodes[i, valid[i], 0].mean() for i in station_ids])
    increments = np.diff(station_means) / np.maximum(np.diff(station_ids), 1)
    spacing_cv = (float(np.std(increments)/max(abs(np.mean(increments)), 1e-9))
                  if len(increments) > 1 else None)
    return {
        "available": bool(string_residuals or fret_residuals),
        "visible_node_count": int(valid.sum()),
        "total_node_count": int(valid.size),
        "horizontal_string_straightness_rms_px": string_rms,
        "horizontal_string_straightness_rms_over_height": (string_rms / height) if string_rms is not None else None,
        "vertical_fret_straightness_rms_px": fret_rms,
        "vertical_fret_straightness_rms_over_width": (fret_rms / width) if fret_rms is not None else None,
        "gt_lattice_additive_fit_rmse_px": joint_rmse,
        "gt_lattice_additive_fit_rmse_over_diagonal": (joint_rmse / math.hypot(width, height)) if joint_rmse is not None else None,
        "string_paths_scored": int(sum(np.count_nonzero(valid[:, i]) >= 2 for i in range(nodes.shape[1]))),
        "fret_paths_scored": int(sum(np.count_nonzero(valid[i]) >= 2 for i in range(nodes.shape[0]))),
        "relative_fret_spacing_cv": spacing_cv,
        "mapped_visible_fraction": float(valid.sum()/max(np.asarray(visible).sum(), 1)),
    }


def _estimated_lattice_rmse(
    result: Any,
    scene: SyntheticFretboard,
    output_size: Tuple[int, int],
    mapper: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, Any]:
    """Score ordered estimated nodes against their explicitly reported target grid.

    Exact ordered shape agreement with the fixture is required: without a
    trusted fret/string assignment, assigning detections to labels by proximity
    would make the metric hide missing or extra detections.
    """
    estimated = _as_numpy(getattr(result, "source_nodes", None))
    target_x = _as_numpy(getattr(result, "target_x", None))
    target_y = _as_numpy(getattr(result, "target_y", None))
    if any(value is None for value in (estimated, target_x, target_y)):
        return {"available": False, "reason": "method does not expose source_nodes, target_x, and target_y"}
    estimated = np.asarray(estimated, dtype=np.float64)
    target_x = np.asarray(target_x, dtype=np.float64).squeeze()
    target_y = np.asarray(target_y, dtype=np.float64).squeeze()
    if estimated.ndim != 3 or estimated.shape[-1] != 2:
        return {"available": False, "reason": f"source_nodes has unsupported shape {estimated.shape}"}

    truth = scene.source_nodes.astype(np.float64)
    truth_visible = scene.visible_nodes
    if estimated.shape == truth.shape:
        truth_for_compare = truth
        visible_for_compare = truth_visible
    elif estimated.shape == truth[1:].shape:
        # Some rectifiers report fret-wire crossings and omit the nut station.
        truth_for_compare = truth[1:]
        visible_for_compare = truth_visible[1:]
    else:
        return {
            "available": False,
            "reason": f"no unambiguous station assignment: estimated {estimated.shape}, fixture {truth.shape}",
        }

    fret_count, string_count = estimated.shape[:2]
    if target_x.ndim == 0:
        target_x = np.repeat(target_x.reshape(1), fret_count)
    if target_y.ndim == 0:
        target_y = np.repeat(target_y.reshape(1), string_count)
    if target_x.size != fret_count or target_y.size != string_count:
        return {
            "available": False,
            "reason": f"target coordinate sizes {(target_x.size, target_y.size)} do not match nodes {(fret_count, string_count)}",
        }

    try:
        predicted_grid = np.asarray(mapper(estimated), dtype=np.float64)
    except Exception as exc:
        return {"available": False, "reason": f"image_to_grid failed for estimated nodes: {type(exc).__name__}: {exc}"}
    if predicted_grid.shape != estimated.shape or not np.isfinite(predicted_grid).all():
        return {"available": False, "reason": f"image_to_grid returned unsupported shape/values {predicted_grid.shape}"}

    ideal = np.stack(
        np.broadcast_arrays(
            np.broadcast_to(target_x[:, None], (fret_count, string_count)),
            np.broadcast_to(target_y[None, :], (fret_count, string_count)),
        ),
        axis=-1,
    )
    valid = visible_for_compare & np.isfinite(estimated).all(axis=-1)
    if np.count_nonzero(valid) < 4:
        return {"available": False, "reason": f"only {int(valid.sum())} assigned visible nodes; need at least four"}
    error = predicted_grid[valid] - ideal[valid]
    rmse = float(np.sqrt(np.mean(np.sum(np.square(error), axis=-1))))
    width, height = output_size
    # The map may emit output pixels or normalized [0,1] grid coordinates.
    # Normalize by target extents so either representation has interpretable
    # dimensionless error without guessing units from an individual error.
    x_span = float(np.ptp(target_x))
    y_span = float(np.ptp(target_y))
    scale = math.hypot(x_span, y_span)
    if scale <= 1e-12:
        scale = math.hypot(width, height)
    return {
        "available": True,
        "assigned_node_count": int(valid.sum()),
        "total_assigned_node_count": int(valid.size),
        "rmse_target_units": rmse,
        "rmse_normalized_by_target_diagonal": rmse / scale,
    }


def _get_core_callable():
    try:
        module = importlib.import_module("pipeline.fretboard_rectify")
    except Exception as exc:
        return None, f"pipeline.fretboard_rectify unavailable: {type(exc).__name__}: {exc}"
    rectify = getattr(module, "rectify_fretboard", None)
    if not callable(rectify):
        return None, "pipeline.fretboard_rectify has no callable rectify_fretboard"
    return rectify, None


def _call_legacy(scene: SyntheticFretboard, output_size):
    from pipeline.fret_detect_yolo import rectify_perspective

    details = rectify_perspective(
        scene.image,
        scene.mask,
        output_size=output_size,
        return_details=True,
    )
    if details is None:
        raise RuntimeError("legacy PCA corner estimator returned no rectification")
    return {
        "image": details.get("image"),
        "matrix": details.get("transform"),
        "details": {"corners": np.asarray(details.get("corners")).tolist()},
        "result": details,
        "mapper": lambda pts, matrix=details.get("transform"): _map_with_homography(pts, matrix),
    }


def _call_core(rectify, scene: SyntheticFretboard, output_size):
    result = rectify(
        scene.image,
        scene.mask,
        output_size=output_size,
        num_strings=scene.num_strings,
        spacing="uniform",
        occlusion_mask=scene.occlusion_mask,
    )
    if result is None:
        raise RuntimeError("rectify_fretboard returned None")
    image = getattr(result, "image", None)
    if image is None and isinstance(result, dict):
        image = result.get("image")
    mapper = getattr(result, "image_to_grid", None)
    if mapper is None and isinstance(result, dict):
        mapper = result.get("image_to_grid")
    if not callable(mapper):
        raise RuntimeError("result does not provide image_to_grid(points)")
    matrix = _as_numpy(getattr(result, "image_to_grid_matrix", None))
    return {
        "image": _as_numpy(image),
        "matrix": matrix,
        "details": _json_sanitize(getattr(result, "quality", {})),
        "result": result,
        "mapper": mapper,
    }


def _json_sanitize(value: Any):
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    return repr(value)


def _fit_panel(image: Optional[np.ndarray], cell_size=(400, 270), caption="") -> np.ndarray:
    cw, ch = cell_size
    panel = np.full((ch, cw, 3), (35, 36, 38), np.uint8)
    if image is not None:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.ndim == 3 and arr.shape[-1] == 3:
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            scale = min(cw / arr.shape[1], ch / arr.shape[0])
            rw, rh = max(1, int(arr.shape[1] * scale)), max(1, int(arr.shape[0] * scale))
            resized = cv2.resize(arr, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            x0, y0 = (cw - rw) // 2, (ch - rh) // 2
            panel[y0:y0 + rh, x0:x0 + rw] = resized
    if caption:
        cv2.putText(panel, caption[:62], (8, ch - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 237, 237), 1, cv2.LINE_AA)
    return panel


def _annotated_input(scene: SyntheticFretboard) -> np.ndarray:
    image = scene.image.copy()
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours((scene.mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (55, 225, 70), 2)
    if np.any(scene.occlusion_mask):
        red = np.zeros_like(image)
        red[:] = (230, 45, 45)
        keep = scene.occlusion_mask > 0
        image[keep] = cv2.addWeighted(image[keep], 0.38, red[keep], 0.62, 0)
    for point, visible in zip(scene.source_nodes.reshape(-1, 2), scene.visible_nodes.ravel()):
        if visible:
            cv2.circle(image, tuple(np.rint(point).astype(int)), 2, (40, 245, 245), -1, cv2.LINE_AA)
    return image


def run_benchmark(
    output_dir: Path,
    *,
    cases: Iterable[str] = DEFAULT_CASES,
    output_size: Tuple[int, int] = OUTPUT_SIZE,
    fret_count: int = 12,
    num_strings: int = 6,
    seed: int = 11,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = make_cases(cases, fret_count=fret_count, num_strings=num_strings, seed=seed)
    core_rectify, core_unavailable = _get_core_callable()
    methods = {
        "legacy_pca_warp": lambda scene: _call_legacy(scene, output_size),
        "fretboard_rectify": (lambda scene: _call_core(core_rectify, scene, output_size)) if core_rectify else None,
    }
    results: Dict[str, Any] = {
        "schema_version": "1.0",
        "benchmark": "synthetic_tapered_fretboard_rectification",
        "fixture": {
            "fret_count": fret_count,
            "num_strings": num_strings,
            "output_size": list(output_size),
            "seed": seed,
            "cases": list(scenes),
            "ground_truth_node_convention": "station 0 is nut; stations 1..N follow equal temperament through bridge-side end",
        },
        "methods": {},
        "cases": {},
    }
    method_outputs: Dict[str, Dict[str, Optional[np.ndarray]]] = {name: {} for name in methods}
    method_results: Dict[str, Dict[str, Any]] = {name: {} for name in methods}

    for method_name, method in methods.items():
        method_report: Dict[str, Any] = {
            "available": method is not None,
            "unavailable_reason": core_unavailable if method_name == "fretboard_rectify" else None,
            "attempted_cases": 0,
            "successful_cases": 0,
            "failed_cases": 0,
            "coverage": None,
            "runtime_ms": {"mean": None, "median": None, "total": 0.0},
            "failures": [],
        }
        runtimes = []
        for case_name, scene in scenes.items():
            record: Dict[str, Any] = {"status": "unavailable" if method is None else "pending"}
            if method is not None:
                method_report["attempted_cases"] += 1
                start = time.perf_counter()
                try:
                    output = method(scene)
                    elapsed = (time.perf_counter() - start) * 1000.0
                    runtimes.append(elapsed)
                    method_report["successful_cases"] += 1
                    record.update({"status": "ok", "runtime_ms": float(elapsed)})
                    details = output["details"]
                    record["quality"] = _json_sanitize(details)
                    method_outputs[method_name][case_name] = output.get("image")
                    method_results[method_name][case_name] = output
                    matrix = output.get("matrix")
                    if matrix is not None:
                        record["image_to_grid_matrix"] = _json_sanitize(matrix)
                except Exception as exc:
                    elapsed = (time.perf_counter() - start) * 1000.0
                    runtimes.append(elapsed)
                    method_report["failed_cases"] += 1
                    failure = {"case": case_name, "error": f"{type(exc).__name__}: {exc}"}
                    method_report["failures"].append(failure)
                    record.update({"status": "failed", "runtime_ms": float(elapsed), "error": failure["error"]})
                    method_outputs[method_name][case_name] = None
                    method_results[method_name][case_name] = None
                else:
                    # Benchmark diagnostics are separate from invocation
                    # success: a reporting issue must not relabel a usable
                    # rectifier result as a method failure.
                    try:
                        mapped_truth = np.asarray(output["mapper"](scene.source_nodes), dtype=np.float64)
                        if mapped_truth.shape != scene.source_nodes.shape:
                            raise ValueError(f"ground-truth mapper returned {mapped_truth.shape}; expected {scene.source_nodes.shape}")
                        record["ground_truth_regularity"] = _regularity_metrics(
                            mapped_truth, scene.visible_nodes, output_size
                        )
                    except Exception as exc:
                        record["ground_truth_regularity"] = {
                            "available": False,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    try:
                        record["estimated_gt_lattice_reprojection"] = _estimated_lattice_rmse(
                            output["result"], scene, output_size, output["mapper"]
                        )
                    except Exception as exc:
                        record["estimated_gt_lattice_reprojection"] = {
                            "available": False,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
            results["cases"].setdefault(case_name, {
                "description": scene.description,
                "visible_ground_truth_nodes": int(scene.visible_nodes.sum()),
                "total_ground_truth_nodes": int(scene.visible_nodes.size),
                "occlusion_mask_pixels": int((scene.occlusion_mask > 0).sum()),
                "methods": {},
            })
            results["cases"][case_name]["methods"][method_name] = record

        if runtimes:
            method_report["runtime_ms"] = {
                "mean": float(np.mean(runtimes)),
                "median": float(np.median(runtimes)),
                "total": float(np.sum(runtimes)),
            }
        if method is not None:
            attempted = method_report["attempted_cases"]
            method_report["coverage"] = (method_report["successful_cases"] / attempted) if attempted else None
        results["methods"][method_name] = method_report

    # Pairwise comparisons must score identical points. Preserve each method's
    # own coverage above and separately report the intersection of domains.
    for case_name, scene in scenes.items():
        mappings = {}
        common = scene.visible_nodes.copy()
        for name in methods:
            output = method_results[name].get(case_name)
            if output is None:
                continue
            mapped = np.asarray(output["mapper"](scene.source_nodes))
            inside = np.isfinite(mapped).all(axis=-1)
            inside &= (mapped[..., 0] >= 0) & (mapped[..., 0] <= output_size[0]-1)
            inside &= (mapped[..., 1] >= 0) & (mapped[..., 1] <= output_size[1]-1)
            common &= inside
            mappings[name] = mapped
        if len(mappings) != len(methods):
            continue
        for name, mapped in mappings.items():
            results["cases"][case_name]["methods"][name]["paired_common_domain"] = _regularity_metrics(mapped, common, output_size)

    # Export every source image and mask alongside one contact sheet for quick
    # failure inspection. RGB arrays are converted for OpenCV's BGR writer.
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    for case_name, scene in scenes.items():
        cv2.imwrite(str(inputs_dir / f"{case_name}.png"), cv2.cvtColor(scene.image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(inputs_dir / f"{case_name}_mask.png"), scene.mask)
        if np.any(scene.occlusion_mask):
            cv2.imwrite(str(inputs_dir / f"{case_name}_occlusion.png"), scene.occlusion_mask)

    method_names = list(methods)
    cols = 1 + len(method_names)
    cell_w, cell_h = 400, 270
    header_h = 30
    montage = np.full((len(scenes) * (cell_h + header_h), cols * cell_w, 3), (25, 26, 28), np.uint8)
    for row, (case_name, scene) in enumerate(scenes.items()):
        y0 = row * (cell_h + header_h)
        labels = [f"{case_name} | source + GT nodes"] + method_names
        images = [_annotated_input(scene)]
        for method_name in method_names:
            image = method_outputs[method_name].get(case_name)
            record = results["cases"][case_name]["methods"][method_name]
            if image is None:
                image = np.zeros((120, 240, 3), np.uint8)
                cv2.putText(image, record.get("status", "unavailable"), (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
                reason = record.get("error") or results["methods"][method_name].get("unavailable_reason") or "no result"
                cv2.putText(image, str(reason)[:30], (8, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 190, 160), 1, cv2.LINE_AA)
            images.append(image)
        for col, (label, image) in enumerate(zip(labels, images)):
            x0 = col * cell_w
            cv2.putText(montage, label[:60], (x0 + 7, y0 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (236, 236, 236), 1, cv2.LINE_AA)
            montage[y0 + header_h:y0 + header_h + cell_h, x0:x0 + cell_w] = _fit_panel(image, (cell_w, cell_h))
    montage_path = output_dir / "rectification_montage.png"
    cv2.imwrite(str(montage_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    results["artifacts"] = {
        "montage": montage_path.name,
        "inputs_dir": inputs_dir.name,
        "metrics": "metrics.json",
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    return results


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="directory for source fixtures, montage, and metrics.json")
    parser.add_argument("--cases", nargs="+", choices=DEFAULT_CASES, default=DEFAULT_CASES)
    parser.add_argument("--fret-count", type=int, default=12)
    parser.add_argument("--num-strings", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-width", type=int, default=OUTPUT_SIZE[0])
    parser.add_argument("--output-height", type=int, default=OUTPUT_SIZE[1])
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    result = run_benchmark(
        args.output_dir,
        cases=args.cases,
        output_size=(args.output_width, args.output_height),
        fret_count=args.fret_count,
        num_strings=args.num_strings,
        seed=args.seed,
    )
    print(json.dumps({
        "metrics": str((args.output_dir / "metrics.json").resolve()),
        "montage": str((args.output_dir / "rectification_montage.png").resolve()),
        "methods": {
            name: {
                "available": report["available"],
                "coverage": report["coverage"],
                "successful_cases": report["successful_cases"],
                "failed_cases": report["failed_cases"],
                "runtime_ms": report["runtime_ms"],
            }
            for name, report in result["methods"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
