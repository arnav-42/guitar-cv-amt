"""Ground-truth geometry and failure-contract tests for the evidence atlas."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from pipeline.fretboard_rectify import rectify_fretboard
from pipeline.fret_detect_yolo import rectify_perspective
from pipeline.grid_warp import warp_grid
from pipeline.string_evidence import detect_strings
from tests.synthetic_fretboard import make_case


def _fret_rms(points, keep):
    errors = []
    for row, mask in zip(points, keep):
        xs = row[mask, 0]
        if len(xs) >= 2:
            errors.extend(xs-xs.mean())
    return np.sqrt(np.mean(np.square(errors)))


def test_perspective_grid_improves_true_fret_verticality_and_uniform_spacing():
    scene = make_case("perspective", seed=43)
    result = rectify_fretboard(scene.image, scene.mask)
    assert result.quality["status"] == "grid"
    assert result.quality["lattice_accepted"]
    mapped = result.image_to_grid(scene.source_nodes)
    baseline = rectify_perspective(scene.image, scene.mask, return_details=True)
    old = cv2.perspectiveTransform(scene.source_nodes.reshape(1, -1, 2), baseline["transform"]).reshape(scene.source_nodes.shape)
    keep = scene.visible_nodes & np.isfinite(mapped).all(axis=-1)
    assert keep.sum() >= .75*scene.visible_nodes.sum()
    assert _fret_rms(mapped, keep) < .2*_fret_rms(old, keep)
    # Score consecutive known interior frets rather than estimated grid lines.
    mids = np.nanmean(mapped[1:-1], axis=1)[:, 0]
    mids = mids[np.isfinite(mids)]
    assert np.std(np.diff(mids))/abs(np.mean(np.diff(mids))) < .03
    np.testing.assert_allclose(result.target_y[1:-1], np.linspace(.08, .92, 6)*255)


def test_blank_inside_plausible_mask_is_coarse_and_empty_mask_is_none():
    scene = make_case("clean_frontal")
    blank = np.full_like(scene.image, 80)
    result = rectify_fretboard(blank, scene.mask, output_size=(160, 48))
    assert result.quality["status"] == "coarse"
    assert not result.quality["lattice_accepted"]
    assert result.quality["observed_frets"] == 0
    assert result.image.shape == (48, 160, 3)
    assert rectify_fretboard(blank, np.zeros_like(scene.mask)) is None
    assert detect_strings(np.full((256, 1024), 80, np.uint8), np.ones((256, 1024), np.uint8)) == []


def test_occluded_and_partial_geometry_never_claim_absolute_fret_numbers():
    scene = make_case("hand_occlusion", seed=47)
    result = rectify_fretboard(scene.image, scene.mask, occlusion_mask=scene.occlusion_mask)
    assert result.quality["status"] == "grid"
    assert result.quality["occlusion_known"]
    assert np.count_nonzero(result.visible_mask) < np.count_nonzero(result.valid_mask)
    assert not result.quality["absolute_fret_numbers_known"]
    partial = make_case("partial_longitudinal_crop", seed=37)
    result = rectify_fretboard(partial.image, partial.mask)
    assert result.quality["observed_frets"] < 21
    assert not result.quality["absolute_fret_numbers_known"]
    assert result.quality["visible_fraction"] is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_mapping_roundtrip_interior_points_and_outside_nan():
    scene = make_case("rotated_negative_steep", seed=79)
    result = rectify_fretboard(scene.image, scene.mask)
    rng = np.random.default_rng(8)
    points = rng.uniform([0, 0], [1023, 255], (30, 2))
    source = result.grid_to_image(points)
    np.testing.assert_allclose(result.image_to_grid(source), points, atol=1e-5)
    assert np.isnan(result.grid_to_image([[-1, 12]])).all()


def test_occlusion_visibility_covers_all_bilinear_contributors():
    nodes = np.array([[[.1, .1], [2., .1]], [[.1, 2.], [2., 2.]]])
    image = np.ones((3, 3), np.uint8)
    occlusion = np.zeros((3, 3), np.uint8)
    occlusion[1, 1] = 255
    r = warp_grid(image, nodes, [0, 3], [0, 3], (4, 4), occlusion_mask=occlusion)
    assert r["valid_mask"][0, 0]
    assert not r["visible_mask"][0, 0]


@pytest.mark.parametrize("kwargs", [dict(output_size=(1, 200)), dict(num_strings=1), dict(spacing="magic")])
def test_invalid_options_rejected(kwargs):
    scene = make_case("clean_frontal")
    with pytest.raises(ValueError):
        rectify_fretboard(scene.image, scene.mask, **kwargs)


def test_model_free_cli_exports_real_image_and_inverse_geometry(tmp_path):
    scene = make_case("perspective")
    Image.fromarray(scene.image).save(tmp_path/"frame.png")
    Image.fromarray(scene.mask).save(tmp_path/"mask.png")
    out = tmp_path/"atlas"
    run = subprocess.run([sys.executable, "-m", "pipeline.rectify_atlas", "--image", str(tmp_path/"frame.png"),
                          "--mask", str(tmp_path/"mask.png"), "--output-dir", str(out)],
                         cwd=Path(__file__).parents[1], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = json.loads((out/"geometry.json").read_text())
    assert report["quality"]["status"] == "grid"
    assert report["size"] == [1024, 256]
    assert Image.open(out/"rectified.png").size == (1024, 256)
    assert not (out/"visible_mask.png").exists()  # unknown, not "all visible"
    assert (out/"mapping.npz").is_file()
