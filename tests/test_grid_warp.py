"""Synthetic geometry tests for the piecewise-bilinear grid warp."""

import numpy as np
import pytest

from pipeline.grid_warp import (
    build_grid_maps,
    grid_to_image,
    image_to_grid,
    warp_grid,
)


def _skew_grid():
    # A convex, mildly bowed 3 by 3 grid. Target axes have nonuniform spacing.
    nodes = np.array(
        [
            [[2.0, 1.0], [7.0, 0.5], [12.0, 1.0]],
            [[1.0, 6.0], [7.5, 6.2], [13.0, 5.7]],
            [[0.5, 11.0], [7.0, 11.5], [14.0, 11.0]],
        ]
    )
    target_x = np.array([0.0, 4.0, 8.0])
    target_y = np.array([0.0, 3.0, 6.0])
    return nodes, target_x, target_y


def test_skew_grid_round_trip_and_endpoint_interpolation():
    nodes, tx, ty = _skew_grid()
    targets = np.array(
        [
            [0.0, 0.0], [4.0, 3.0], [8.0, 6.0],
            [1.5, 2.0], [6.1, 4.9], [8.0, 1.25],
        ]
    )
    image_points = grid_to_image(targets, nodes, tx, ty)
    assert np.allclose(image_points[[0, 1, 2]], nodes[[0, 1, 2], [0, 1, 2]], atol=1e-12)

    recovered = image_to_grid(image_points, nodes, tx, ty)
    assert np.allclose(recovered, targets, atol=1e-8)

    outside = grid_to_image(np.array([[-0.01, 1.0], [3.0, 6.01]]), nodes, tx, ty)
    assert np.isnan(outside).all()
    outside_image = image_to_grid(np.array([[100.0, 100.0]]), nodes, tx, ty)
    assert np.isnan(outside_image).all()


def test_build_maps_include_output_endpoints_and_grid_nodes():
    nodes, tx, ty = _skew_grid()
    map_x, map_y = build_grid_maps(nodes, tx, ty, (9, 7))
    assert map_x.shape == (7, 9)
    assert map_y.shape == (7, 9)
    assert np.allclose([map_x[0, 0], map_y[0, 0]], nodes[0, 0])
    assert np.allclose([map_x[3, 4], map_y[3, 4]], nodes[1, 1])
    assert np.allclose([map_x[-1, -1], map_y[-1, -1]], nodes[-1, -1])


def test_rejects_folded_or_degenerate_source_cell():
    nodes, tx, ty = _skew_grid()
    folded = nodes.copy()
    folded[1, 1] = [20.0, -2.0]
    with pytest.raises(ValueError, match="convex|orientation|fold"):
        build_grid_maps(folded, tx, ty, (9, 7))

    collapsed = nodes.copy()
    collapsed[0, 1] = collapsed[0, 0]
    with pytest.raises(ValueError, match="convex|Jacobian|fold"):
        build_grid_maps(collapsed, tx, ty, (9, 7))


def test_rejects_locally_valid_grid_that_wraps_and_overlaps_itself():
    theta = np.linspace(0, 2.2*np.pi, 15)
    nodes = np.stack([np.column_stack([r*np.cos(theta), r*np.sin(theta)]) for r in [5., 8.]])
    with pytest.raises(ValueError, match="boundary|global fold"):
        build_grid_maps(nodes, np.linspace(0, 140, 15), [0, 30], (141, 31))


def test_requires_full_output_coverage_and_valid_dimensions():
    nodes, tx, ty = _skew_grid()
    with pytest.raises(ValueError, match="target_x"):
        build_grid_maps(nodes, np.array([1.0, 4.0, 8.0]), ty, (9, 7))
    with pytest.raises(ValueError, match="output_size"):
        build_grid_maps(nodes, tx, ty, (9.0, 7))


def test_valid_and_visible_masks_track_bounds_and_occlusion():
    # The warp's left side samples outside a 5 by 5 image; the central source
    # square is separately marked as occluded.
    nodes = np.array(
        [[[-1.0, -1.0], [2.0, -1.0]], [[-1.0, 3.0], [2.0, 3.0]]]
    )
    tx = np.array([0.0, 4.0])
    ty = np.array([0.0, 4.0])
    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    occ = np.zeros((5, 5), dtype=np.uint8)
    occ[1:3, 1:3] = 255

    result = warp_grid(image, nodes, tx, ty, (5, 5), occlusion_mask=occ)
    assert result["image"].shape == (5, 5)
    assert result["valid_mask"].dtype == np.bool_
    assert result["visible_mask"].dtype == np.bool_
    assert not np.any(result["visible_mask"] & ~result["valid_mask"])
    assert not result["valid_mask"][0, 0]
    assert result["valid_mask"][-1, -1]
    assert not result["visible_mask"][2, 4]  # Maps exactly to occluded source pixel (2, 1).
    assert result["visible_mask"][4, 4]  # Maps to a clear, in-frame source pixel.
    assert result["image"][0, 0] == 0  # Far outside the source, remap uses constant 0.

    no_occ = warp_grid(image, nodes, tx, ty, (5, 5))
    assert np.array_equal(no_occ["visible_mask"], no_occ["valid_mask"])


def test_rgb_warp_preserves_channels():
    nodes = np.array([[[0.0, 0.0], [3.0, 0.0]], [[0.0, 2.0], [3.0, 2.0]]])
    tx = np.array([0.0, 3.0])
    ty = np.array([0.0, 2.0])
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[..., 0] = 17
    image[..., 1] = 83
    image[..., 2] = 201

    result = warp_grid(image, nodes, tx, ty, (4, 3))
    assert result["image"].shape == (3, 4, 3)
    assert np.array_equal(result["image"], image)
    assert result["valid_mask"].all()
