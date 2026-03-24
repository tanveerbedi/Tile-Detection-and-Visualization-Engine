"""
Tests for the Visualization Engine
-----------------------------------
Tests each pipeline stage independently with synthetic test images.

Run: python -m pytest tests/test_visualization.py -v
"""

import numpy as np
import cv2
import sys
import os
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ========================= FIXTURES =========================

@pytest.fixture
def sample_room_image():
    """Create a synthetic room image (480x640) with a floor region."""
    img = np.full((480, 640, 3), (200, 200, 210), dtype=np.uint8)  # Light walls

    # Draw darker floor in bottom half
    img[280:, :] = (140, 130, 120)  # Brown floor

    # Add some variation
    cv2.rectangle(img, (0, 0), (640, 60), (180, 180, 190), -1)  # Ceiling
    cv2.rectangle(img, (50, 100), (150, 280), (160, 140, 100), -1)  # Furniture

    return img


@pytest.fixture
def sample_tile_image():
    """Create a synthetic tile texture (100x100)."""
    tile = np.full((100, 100, 3), (180, 160, 140), dtype=np.uint8)

    # Add a simple pattern
    for i in range(0, 100, 20):
        cv2.line(tile, (i, 0), (i, 100), (160, 140, 120), 1)
        cv2.line(tile, (0, i), (100, i), (160, 140, 120), 1)

    return tile


@pytest.fixture
def sample_floor_mask():
    """Create a binary floor mask matching the room layout."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[280:, :] = 255  # Floor in bottom half
    return mask


@pytest.fixture
def sample_depth_map():
    """Create a synthetic depth map (closer at bottom, farther at top)."""
    h, w = 480, 640
    depth = np.zeros((h, w), dtype=np.float32)

    # Linear gradient: bottom (close/high value) to top (far/low value)
    for y in range(h):
        depth[y, :] = 1.0 - (y / h)  # Inverse depth: bottom = close = large

    return depth


# ========================= TILE RENDERER TESTS =========================

class TestTileRenderer:

    def test_straight_pattern(self, sample_tile_image):
        from visualization.tile_renderer import create_tile_canvas
        canvas = create_tile_canvas(sample_tile_image, (400, 400), pattern="straight")
        assert canvas.shape == (400, 400, 3)
        assert canvas.dtype == np.uint8

    def test_diagonal_pattern(self, sample_tile_image):
        from visualization.tile_renderer import create_tile_canvas
        canvas = create_tile_canvas(sample_tile_image, (400, 400), pattern="diagonal")
        assert canvas.shape == (400, 400, 3)

    def test_brick_pattern(self, sample_tile_image):
        from visualization.tile_renderer import create_tile_canvas
        canvas = create_tile_canvas(sample_tile_image, (400, 400), pattern="brick")
        assert canvas.shape == (400, 400, 3)

    def test_warp_canvas(self, sample_tile_image):
        from visualization.tile_renderer import create_tile_canvas, warp_tile_canvas
        canvas = create_tile_canvas(sample_tile_image, (400, 400))
        H = np.eye(3, dtype=np.float64)
        warped = warp_tile_canvas(canvas, H, (480, 640))
        assert warped.shape == (480, 640, 3)

    def test_apply_floor_mask(self, sample_tile_image, sample_floor_mask):
        from visualization.tile_renderer import create_tile_canvas, apply_floor_mask
        canvas = create_tile_canvas(sample_tile_image, (640, 480))
        masked = apply_floor_mask(canvas, sample_floor_mask)
        assert masked.shape == (480, 640, 3)
        # Top half should be zero (not floor)
        assert np.all(masked[:200, :] == 0)


# ========================= LIGHTING TESTS =========================

class TestLightingBlender:

    def test_brightness_map(self, sample_room_image, sample_floor_mask):
        from visualization.lighting_blender import extract_brightness_map
        bmap = extract_brightness_map(sample_room_image, sample_floor_mask)
        assert bmap.shape == sample_room_image.shape[:2]
        assert bmap.dtype == np.float32
        # Should be centered around 1.0
        floor_mean = np.mean(bmap[sample_floor_mask > 0])
        assert 0.5 < floor_mean < 2.0

    def test_feather_edges(self, sample_floor_mask):
        from visualization.lighting_blender import feather_edges
        feathered = feather_edges(sample_floor_mask)
        assert feathered.shape == sample_floor_mask.shape
        assert feathered.dtype == np.float32
        assert feathered.min() >= 0.0
        assert feathered.max() <= 1.0

    def test_composite(self, sample_room_image, sample_floor_mask):
        from visualization.lighting_blender import composite, feather_edges
        tile_region = np.full_like(sample_room_image, (100, 80, 60))
        alpha = feather_edges(sample_floor_mask)
        result = composite(sample_room_image, tile_region, alpha)
        assert result.shape == sample_room_image.shape
        assert result.dtype == np.uint8


# ========================= PLANE GEOMETRY TESTS =========================

class TestPlaneGeometry:

    def test_camera_intrinsics(self):
        from visualization.plane_geometry import estimate_camera_intrinsics
        K = estimate_camera_intrinsics((480, 640))
        assert K.shape == (3, 3)
        assert K[0, 0] > 0  # focal length
        assert K[1, 1] > 0
        assert K[2, 2] == 1.0

    def test_ransac_plane_fit(self):
        from visualization.plane_geometry import fit_plane_ransac
        # Create a perfect plane y = 0
        points = np.random.randn(100, 3)
        points[:, 1] = 0  # All y=0

        normal, d, inliers = fit_plane_ransac(points)
        assert normal.shape == (3,)
        assert np.sum(inliers) > 50  # Most should be inliers

    def test_homography_estimation(self, sample_floor_mask, sample_depth_map):
        from visualization.plane_geometry import estimate_homography
        H, canvas_size = estimate_homography(sample_floor_mask, sample_depth_map)
        assert H.shape == (3, 3)
        assert len(canvas_size) == 2


# ========================= DEPTH ESTIMATION TESTS =========================

class TestDepthEstimation:

    def test_normalize_depth(self, sample_depth_map):
        from visualization.depth_estimation import normalize_depth
        normalized = normalize_depth(sample_depth_map)
        assert normalized.shape == sample_depth_map.shape
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_depth_scale_map(self, sample_depth_map, sample_floor_mask):
        from visualization.depth_estimation import get_depth_scale_map
        scale_map = get_depth_scale_map(sample_depth_map, sample_floor_mask)
        assert scale_map.shape == sample_depth_map.shape
        # Scale should be within range
        floor_scales = scale_map[sample_floor_mask > 0]
        if len(floor_scales) > 0:
            assert floor_scales.min() >= 0.3
            assert floor_scales.max() <= 1.1

    def test_visualize_depth(self, sample_depth_map):
        from visualization.depth_estimation import visualize_depth
        vis = visualize_depth(sample_depth_map)
        assert vis.shape == (sample_depth_map.shape[0], sample_depth_map.shape[1], 3)
        assert vis.dtype == np.uint8


# ========================= FLOOR SEGMENTATION TESTS =========================

class TestFloorSegmentation:

    def test_clean_mask(self, sample_floor_mask):
        from visualization.floor_segmentation import clean_mask
        # Add noise
        noisy = sample_floor_mask.copy()
        noisy[50, 50] = 255  # Isolated pixel
        cleaned = clean_mask(noisy)
        assert cleaned.shape == noisy.shape
        assert cleaned.dtype == np.uint8

    def test_refine_mask_with_click(self, sample_floor_mask, sample_room_image):
        from visualization.floor_segmentation import refine_mask_with_click
        refined = refine_mask_with_click(
            sample_floor_mask, sample_room_image, (320, 400))
        assert refined.shape == sample_floor_mask.shape
        assert refined.dtype == np.uint8


# ========================= AI REALISTIC MODE TESTS =========================

class TestAIRealisticMode:

    def test_placeholder_disabled(self):
        from visualization.ai_realistic_mode import AIRealisticMode
        ai = AIRealisticMode()
        assert ai.enabled is False
        assert ai.is_available() is False

    def test_placeholder_raises(self):
        from visualization.ai_realistic_mode import AIRealisticMode
        ai = AIRealisticMode()
        with pytest.raises(NotImplementedError):
            ai.apply()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
