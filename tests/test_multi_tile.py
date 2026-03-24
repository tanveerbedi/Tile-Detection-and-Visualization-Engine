"""
Tests for Multi-Tile Comparative Visualization Engine
------------------------------------------------------
Uses synthetic images and mocked ML models — no GPU or downloaded
weights required.

Run:
    python -m pytest tests/test_multi_tile.py -v
"""

import sys
import os
import numpy as np
import cv2
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ========================= SHARED FIXTURES =========================

@pytest.fixture
def room_image():
    """480×640 synthetic room (walls + floor)."""
    img = np.full((480, 640, 3), (200, 200, 210), dtype=np.uint8)
    img[280:, :] = (140, 130, 120)
    return img


@pytest.fixture
def floor_mask():
    """Binary mask covering the bottom half of the image."""
    m = np.zeros((480, 640), dtype=np.uint8)
    m[280:, :] = 255
    return m


@pytest.fixture
def tile_a():
    """Solid beige tile texture."""
    t = np.full((100, 100, 3), (180, 160, 140), dtype=np.uint8)
    for i in range(0, 100, 20):
        cv2.line(t, (i, 0), (i, 100), (160, 140, 120), 1)
        cv2.line(t, (0, i), (100, i), (160, 140, 120), 1)
    return t


@pytest.fixture
def tile_b():
    """Darker marble-ish tile texture."""
    t = np.full((100, 100, 3), (80, 70, 65), dtype=np.uint8)
    cv2.circle(t, (50, 50), 20, (100, 90, 85), -1)
    return t


@pytest.fixture
def tile_c():
    """Light gray tile."""
    return np.full((100, 100, 3), (220, 220, 215), dtype=np.uint8)


@pytest.fixture
def synthetic_depth(room_image):
    """Simple linear depth gradient (no MiDaS required)."""
    h, w = room_image.shape[:2]
    depth = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        depth[y, :] = 1.0 - y / h
    return depth


# ========================= HELPERS =========================

def _make_mock_midas(depth_map):
    """
    Returns a context-manager patch that makes load_midas_model +
    estimate_depth return our pre-built synthetic depth map without
    actually loading the MiDaS neural network.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with patch(
            "visualization.multi_tile_engine.build_geometry_cache.__wrapped__"
            if hasattr(
                __import__("visualization.multi_tile_engine", fromlist=["build_geometry_cache"]).build_geometry_cache,
                "__wrapped__"
            )
            else "builtins.open",  # fallback key — never actually used
        ):
            yield

    # Simpler approach: patch the sub-module functions directly
    return depth_map


# ========================= GEOMETRY CACHE TESTS =========================

class TestBuildGeometryCache:

    def test_wall_surface_skips_depth(self, room_image, floor_mask):
        """Wall surface should NOT call MiDaS and should set norm_depth=None."""
        from visualization.multi_tile_engine import build_geometry_cache
        cache = build_geometry_cache(
            room_image, floor_mask,
            device="cpu", surface_type="wall"
        )
        assert cache.norm_depth is None
        assert cache.depth_colored is None
        assert cache.homography.shape == (3, 3)
        assert len(cache.canvas_size) == 2
        assert cache.feathered_mask.shape == floor_mask.shape
        assert cache.floor_stats is not None

    def test_floor_surface_with_mocked_depth(self, room_image, floor_mask, synthetic_depth):
        """Floor surface should produce a valid cache using injected depth map."""
        from visualization.multi_tile_engine import build_geometry_cache
        from visualization.depth_estimation import normalize_depth, visualize_depth

        norm = normalize_depth(synthetic_depth)
        colored = visualize_depth(synthetic_depth)

        with patch("visualization.multi_tile_engine.load_midas_model") as mock_load, \
             patch("visualization.multi_tile_engine.estimate_depth") as mock_est, \
             patch("visualization.multi_tile_engine.normalize_depth", return_value=norm), \
             patch("visualization.multi_tile_engine.visualize_depth", return_value=colored):

            mock_load.return_value = (MagicMock(), MagicMock(), "cpu")
            mock_est.return_value = synthetic_depth

            cache = build_geometry_cache(
                room_image, floor_mask,
                device="cpu", surface_type="floor"
            )

        assert cache.norm_depth is not None
        assert cache.norm_depth.shape == floor_mask.shape
        assert cache.homography.shape == (3, 3)
        assert cache.build_time_s >= 0.0

    def test_mask_resize(self, room_image, synthetic_depth):
        """Cache builder should resize a mask that doesn't match the image."""
        from visualization.multi_tile_engine import build_geometry_cache

        wrong_size_mask = np.zeros((240, 320), dtype=np.uint8)
        wrong_size_mask[140:, :] = 255

        cache = build_geometry_cache(
            room_image, wrong_size_mask,
            device="cpu", surface_type="wall"
        )
        # Feathered mask must match room_image shape
        assert cache.feathered_mask.shape == room_image.shape[:2]


# ========================= SINGLE TILE FROM CACHE TESTS =========================

class TestRenderSingleTileFromCache:

    def _build_wall_cache(self, room_image, floor_mask):
        from visualization.multi_tile_engine import build_geometry_cache
        return build_geometry_cache(
            room_image, floor_mask, device="cpu", surface_type="wall"
        )

    def test_result_shape(self, room_image, floor_mask, tile_a):
        """Result image must match room image dimensions."""
        from visualization.multi_tile_engine import render_single_tile_from_cache
        cache = self._build_wall_cache(room_image, floor_mask)
        result = render_single_tile_from_cache(
            room_image, tile_a, floor_mask, cache,
            tile_name="Tile A", similarity_score=0.85
        )
        assert result["result"].shape == room_image.shape
        assert result["result"].dtype == np.uint8

    def test_metadata_preserved(self, room_image, floor_mask, tile_a):
        """Name and score must pass through unchanged."""
        from visualization.multi_tile_engine import render_single_tile_from_cache
        cache = self._build_wall_cache(room_image, floor_mask)
        result = render_single_tile_from_cache(
            room_image, tile_a, floor_mask, cache,
            tile_name="BeigeMarbel", similarity_score=0.72
        )
        assert result["tile_name"] == "BeigeMarbel"
        assert pytest.approx(result["similarity_score"], abs=1e-6) == 0.72

    def test_timings_present(self, room_image, floor_mask, tile_a):
        """Timings dict must contain the expected keys."""
        from visualization.multi_tile_engine import render_single_tile_from_cache
        cache = self._build_wall_cache(room_image, floor_mask)
        result = render_single_tile_from_cache(
            room_image, tile_a, floor_mask, cache
        )
        assert "tile_rendering" in result["timings"]
        assert "lighting" in result["timings"]
        assert "compositing" in result["timings"]
        assert all(v >= 0 for v in result["timings"].values())

    def test_homography_matches_cache(self, room_image, floor_mask, tile_a):
        """Result homography must be identical to cache (not recomputed)."""
        from visualization.multi_tile_engine import render_single_tile_from_cache, build_geometry_cache
        cache = build_geometry_cache(room_image, floor_mask, device="cpu", surface_type="wall")
        result = render_single_tile_from_cache(room_image, tile_a, floor_mask, cache)
        np.testing.assert_array_equal(result["homography"], cache.homography)


# ========================= MULTI-TILE ORCHESTRATOR TESTS =========================

class TestRenderMultipleTiles:

    def _tiles(self, *imgs_and_names):
        """Build tile spec list from (img, name, score) triples."""
        return [
            {"image": img, "name": name, "score": score}
            for img, name, score in imgs_and_names
        ]

    def test_empty_tiles_returns_empty_list(self, room_image, floor_mask):
        """Calling with zero tiles must return []."""
        from visualization.multi_tile_engine import render_multiple_tiles
        results = render_multiple_tiles(
            room_image, [], floor_mask,
            device="cpu", surface_type="wall"
        )
        assert results == []

    def test_single_tile_returns_one_result(self, room_image, floor_mask, tile_a):
        """Exactly one result for one tile."""
        from visualization.multi_tile_engine import render_multiple_tiles
        tiles = self._tiles((tile_a, "Tile A", 0.9))
        results = render_multiple_tiles(
            room_image, tiles, floor_mask,
            device="cpu", surface_type="wall"
        )
        assert len(results) == 1
        assert results[0]["result"].shape == room_image.shape

    def test_multiple_tiles_correct_count(self, room_image, floor_mask, tile_a, tile_b, tile_c):
        """Three tiles → three results."""
        from visualization.multi_tile_engine import render_multiple_tiles
        tiles = self._tiles(
            (tile_a, "Beige", 0.80),
            (tile_b, "Marble", 0.91),
            (tile_c, "Gray", 0.74),
        )
        results = render_multiple_tiles(
            room_image, tiles, floor_mask,
            device="cpu", surface_type="wall"
        )
        assert len(results) == 3
        for r in results:
            assert r["result"] is not None
            assert r["result"].shape == room_image.shape

    def test_geometry_computed_once(self, room_image, floor_mask, tile_a, tile_b):
        """
        build_geometry_cache must be called EXACTLY ONCE even for multiple tiles.
        Verified by patching and counting invocations.
        """
        from visualization import multi_tile_engine

        call_count = {"n": 0}
        original_build = multi_tile_engine.build_geometry_cache

        def counting_build(*args, **kwargs):
            call_count["n"] += 1
            return original_build(*args, **kwargs)

        tiles = self._tiles(
            (tile_a, "Tile A", 0.8),
            (tile_b, "Tile B", 0.9),
        )

        with patch.object(multi_tile_engine, "build_geometry_cache", side_effect=counting_build):
            multi_tile_engine.render_multiple_tiles(
                room_image, tiles, floor_mask,
                device="cpu", surface_type="wall"
            )

        assert call_count["n"] == 1, (
            f"build_geometry_cache was called {call_count['n']} times — "
            "should be exactly 1 (geometry must be shared)."
        )

    def test_all_results_use_same_homography(self, room_image, floor_mask, tile_a, tile_b, tile_c):
        """All results must carry the SAME homography matrix (from shared cache)."""
        from visualization.multi_tile_engine import render_multiple_tiles
        tiles = self._tiles(
            (tile_a, "A", 0.8),
            (tile_b, "B", 0.9),
            (tile_c, "C", 0.7),
        )
        results = render_multiple_tiles(
            room_image, tiles, floor_mask,
            device="cpu", surface_type="wall"
        )
        H0 = results[0]["homography"]
        for r in results[1:]:
            np.testing.assert_array_equal(
                r["homography"], H0,
                err_msg="Homography must be identical across all tiles."
            )

    def test_names_and_scores_preserved(self, room_image, floor_mask, tile_a, tile_b):
        """Tile names and scores must pass through to result dicts."""
        from visualization.multi_tile_engine import render_multiple_tiles
        tiles = self._tiles(
            (tile_a, "CoolBeige", 0.88),
            (tile_b, "DarkMarble", 0.65),
        )
        results = render_multiple_tiles(
            room_image, tiles, floor_mask,
            device="cpu", surface_type="wall"
        )
        assert results[0]["tile_name"] == "CoolBeige"
        assert pytest.approx(results[0]["similarity_score"], abs=1e-6) == 0.88
        assert results[1]["tile_name"] == "DarkMarble"
        assert pytest.approx(results[1]["similarity_score"], abs=1e-6) == 0.65

    def test_progress_callback_called(self, room_image, floor_mask, tile_a, tile_b):
        """progress_callback must be called once per tile + once for geometry."""
        from visualization.multi_tile_engine import render_multiple_tiles

        calls = []

        def cb(current, total, msg):
            calls.append((current, total, msg))

        tiles = self._tiles(
            (tile_a, "A", 0.8),
            (tile_b, "B", 0.9),
        )
        render_multiple_tiles(
            room_image, tiles, floor_mask,
            device="cpu", surface_type="wall",
            progress_callback=cb,
        )
        # Expect: 1 call for geometry + 2 for tiles = 3 total
        assert len(calls) == 3
        assert calls[0][0] == 0   # geometry step: current=0
        assert calls[1][0] == 1   # tile 1
        assert calls[2][0] == 2   # tile 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
