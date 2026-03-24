"""
Multi-Tile Comparative Visualization Engine
-------------------------------------------
Enables rendering multiple tile textures onto the SAME room image
using a SHARED geometry pipeline (depth map, homography, plane data).

Critical performance principle:
  - GeometryCache: computed ONCE per room/mask combination
  - Per-tile loop: only tile canvas creation + warp + light + blend

Usage:
    from visualization.multi_tile_engine import (
        build_geometry_cache,
        render_multiple_tiles,
    )

    cache = build_geometry_cache(room_image, floor_mask, device="auto")
    results = render_multiple_tiles(
        room_image, tiles, floor_mask, device="auto",
        pattern="straight", tile_real_size=0.6,
        grout_width=2, grout_color=(60, 60, 60),
        color_match_strength=0.25, feather_radius=15,
        surface_type="floor",
    )
    # results[i]["result"]       → rendered image (BGR)
    # results[i]["tile_name"]    → name string
    # results[i]["similarity"]   → similarity score (0–1), if available
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .depth_estimation import (
    load_midas_model,
    estimate_depth,
    normalize_depth,
    visualize_depth,
)
from .plane_geometry import estimate_homography, estimate_wall_homography
from .lighting_blender import (
    extract_brightness_map,
    extract_color_statistics,
    feather_edges,
    apply_lighting,
    color_match_tile,
    composite,
)
from .tile_renderer import create_tile_canvas, warp_tile_canvas
from .realism import enhance_realism, add_tile_variation

logger = logging.getLogger(__name__)


# ========================= GEOMETRY CACHE =========================

@dataclass
class GeometryCache:
    """
    Immutable snapshot of all shared geometry computed from the room image
    and floor mask. This must be created ONCE and passed to every tile render.

    Attributes:
        norm_depth:      Normalized depth map (H, W), float32, [0, 1].
        depth_colored:   Colorized depth (H, W, 3) BGR for debug display.
        homography:      3×3 perspective transform (canvas → image).
        canvas_size:     (width, height) of the tile canvas in pixels.
        feathered_mask:  Soft alpha mask (H, W), float32, [0, 1].
        brightness_map:  Per-pixel brightness multiplier (H, W), float32.
        floor_stats:     Color statistics dict from the room floor region.
        surface_type:    'floor' or 'wall'.
        pixels_per_meter: Tile canvas scale factor.
        build_time_s:    Total wall-clock time to build the cache (seconds).
    """
    norm_depth: Optional[np.ndarray]
    depth_colored: Optional[np.ndarray]
    homography: np.ndarray
    canvas_size: Tuple[int, int]
    feathered_mask: np.ndarray
    brightness_map: np.ndarray
    floor_stats: dict
    surface_type: str = "floor"
    pixels_per_meter: float = 400.0
    build_time_s: float = 0.0


# ========================= GEOMETRY BUILDER =========================

def build_geometry_cache(
    room_image: np.ndarray,
    floor_mask: np.ndarray,
    device: str = "auto",
    surface_type: str = "floor",
    pixels_per_meter: float = 400.0,
    feather_radius: int = 15,
) -> GeometryCache:
    """
    Compute all expensive, tile-independent geometry from the room image
    and floor mask. This should be called ONCE and reused for all tiles.

    Steps:
        1. (floor only) MiDaS depth estimation
        2. Homography estimation (3D perspective for floor, 2D affine for wall)
        3. Lighting analysis — brightness map + color statistics
        4. Feathered edge mask

    Args:
        room_image:       Room photo (H, W, 3), BGR, uint8.
        floor_mask:       Brush mask (H, W), uint8, 0 or 255.
        device:           'cuda', 'cpu', or 'auto'.
        surface_type:     'floor' or 'wall'.
        pixels_per_meter: Canvas pixel density.
        feather_radius:   Gaussian blur radius for edge feathering.

    Returns:
        Populated GeometryCache ready for reuse across all tiles.
    """
    t_start = time.time()
    h, w = room_image.shape[:2]

    # Ensure mask dimensions match
    if floor_mask.shape[:2] != (h, w):
        floor_mask = cv2.resize(floor_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    logger.info(
        f"[MultiTileEngine] Building geometry cache. "
        f"Image: {w}×{h}, surface: {surface_type}, device: {device}"
    )

    # ── Step 1: Depth Estimation (floor only) ──────────────────────────────
    norm_depth = None
    depth_colored = None

    if surface_type == "floor":
        logger.info("[MultiTileEngine] Step 1/4: MiDaS depth estimation...")
        midas_model, midas_transform, midas_device = load_midas_model(device)
        raw_depth = estimate_depth(room_image, midas_model, midas_transform, midas_device)
        norm_depth = normalize_depth(raw_depth)
        depth_colored = visualize_depth(raw_depth)
        logger.info("[MultiTileEngine] Depth estimation done.")
    else:
        logger.info("[MultiTileEngine] Step 1/4: Skipping depth (wall surface).")


    # ── Step 2: Homography ─────────────────────────────────────────────────
    logger.info("[MultiTileEngine] Step 2/4: Computing homography...")
    if surface_type == "floor":
        homography, canvas_size = estimate_homography(
            floor_mask, norm_depth, pixels_per_meter=pixels_per_meter
        )
    else:
        homography, canvas_size = estimate_wall_homography(
            floor_mask, pixels_per_meter=pixels_per_meter
        )
    logger.info(f"[MultiTileEngine] Homography computed. Canvas: {canvas_size}")

    # ── Step 3: Lighting Analysis ──────────────────────────────────────────
    logger.info("[MultiTileEngine] Step 3/4: Lighting analysis...")
    brightness_map = extract_brightness_map(room_image, floor_mask)
    floor_stats = extract_color_statistics(room_image, floor_mask)
    logger.info("[MultiTileEngine] Lighting analysis done.")

    # ── Step 4: Feathered Mask ─────────────────────────────────────────────
    logger.info("[MultiTileEngine] Step 4/4: Building feathered mask...")
    feathered_mask = feather_edges(floor_mask, blur_radius=feather_radius)

    build_time = time.time() - t_start
    logger.info(f"[MultiTileEngine] Geometry cache built in {build_time:.2f}s")

    return GeometryCache(
        norm_depth=norm_depth,
        depth_colored=depth_colored,
        homography=homography,
        canvas_size=canvas_size,
        feathered_mask=feathered_mask,
        brightness_map=brightness_map,
        floor_stats=floor_stats,
        surface_type=surface_type,
        pixels_per_meter=pixels_per_meter,
        build_time_s=build_time,
    )


# ========================= SINGLE TILE FROM CACHE =========================

def render_single_tile_from_cache(
    room_image: np.ndarray,
    tile_image: np.ndarray,
    floor_mask: np.ndarray,
    cache: GeometryCache,
    pattern: str = "straight",
    tile_real_size: float = 0.6,
    grout_width: int = 2,
    grout_color: tuple = (60, 60, 60),
    color_match_strength: float = 0.25,
    tile_name: str = "",
    similarity_score: float = 0.0,
    # Realism enhancements
    realism_strength: float = 0.0,
    variation_amount: float = 0.05,
    depth_blur_strength: float = 0.4,
    reflection: bool = False,
    reflection_opacity: float = 0.08,
) -> dict:
    """
    Apply a single tile texture using pre-computed geometric cache.
    Only the per-tile steps run here (canvas + warp + light + blend).

    Args:
        room_image:           Room photo (H, W, 3), BGR, uint8.
        tile_image:           Tile texture (H, W, 3), BGR, uint8.
        floor_mask:           Binary brush mask (H, W), uint8, 0 or 255.
        cache:                GeometryCache built by build_geometry_cache().
        pattern:              'straight', 'diagonal', or 'brick'.
        tile_real_size:       Real-world tile edge length in meters.
        grout_width:          Grout line width in pixels.
        grout_color:          Grout BGR color tuple.
        color_match_strength: Tile color adaptation strength (0–1).
        tile_name:            Display name for the tile (included in result).
        similarity_score:     Similarity score [0, 1] for ranking.

    Returns:
        Result dict with keys:
            result, floor_mask, depth_map, depth_colored,
            warped_tiles, masked_tiles, homography, timings,
            canvas_size, brightness_map, tile_name, similarity_score
    """
    timings: dict = {}
    h, w = room_image.shape[:2]

    logger.info(
        f"[MultiTileEngine] Rendering tile '{tile_name}' "
        f"(pattern={pattern}, size={tile_real_size}m)"
    )

    # ── Step 1: Tile Canvas + Warp ─────────────────────────────────────────
    t0 = time.time()

    # Realism: tile variation pre-warp
    if realism_strength > 0 and variation_amount > 0:
        tile_image_render = add_tile_variation(
            tile_image,
            brightness_range=variation_amount,
            contrast_range=variation_amount,
            noise_std=3.0 * realism_strength,
        )
    else:
        tile_image_render = tile_image

    tile_canvas = create_tile_canvas(
        tile_image_render,
        cache.canvas_size,
        pattern=pattern,
        tile_real_size=tile_real_size,
        pixels_per_meter=cache.pixels_per_meter,
        grout_width=grout_width,
        grout_color=grout_color,
    )
    warped_tiles = warp_tile_canvas(tile_canvas, cache.homography, (h, w))
    masked_tiles = warped_tiles  # soft mask applied at composite stage
    timings["tile_rendering"] = time.time() - t0

    # ── Step 2: Lighting ───────────────────────────────────────────────────
    t0 = time.time()
    lit_tiles = apply_lighting(masked_tiles, cache.brightness_map, floor_mask)
    color_matched = color_match_tile(
        lit_tiles, floor_mask, cache.floor_stats, strength=color_match_strength
    )
    timings["lighting"] = time.time() - t0

    # ── Step 3: Composite ──────────────────────────────────────────────────
    t0 = time.time()
    result = composite(room_image, color_matched, cache.feathered_mask)

    # Realism: post-composite enhancements
    if realism_strength > 0:
        result = enhance_realism(
            result,
            room_image,
            floor_mask,
            norm_depth=cache.norm_depth,
            realism_strength=realism_strength,
            variation_amount=variation_amount,
            depth_blur_strength=depth_blur_strength,
            reflection=reflection,
            reflection_opacity=reflection_opacity,
        )

    timings["compositing"] = time.time() - t0

    total = sum(timings.values())
    logger.info(
        f"[MultiTileEngine] Tile '{tile_name}' rendered in {total:.2f}s"
    )

    return {
        "result": result,
        "floor_mask": floor_mask,
        "depth_map": cache.norm_depth,
        "depth_colored": cache.depth_colored,
        "warped_tiles": warped_tiles,
        "masked_tiles": masked_tiles,
        "homography": cache.homography,
        "timings": timings,
        "canvas_size": cache.canvas_size,
        "brightness_map": cache.brightness_map,
        "tile_name": tile_name,
        "similarity_score": similarity_score,
    }


# ========================= MULTI-TILE ORCHESTRATOR =========================

def render_multiple_tiles(
    room_image: np.ndarray,
    tiles: List[dict],
    floor_mask: np.ndarray,
    device: str = "auto",
    surface_type: str = "floor",
    pattern: str = "straight",
    tile_real_size: float = 0.6,
    grout_width: int = 2,
    grout_color: tuple = (60, 60, 60),
    color_match_strength: float = 0.25,
    feather_radius: int = 15,
    pixels_per_meter: float = 400.0,
    progress_callback=None,
    # Realism enhancements
    realism_strength: float = 0.0,
    variation_amount: float = 0.05,
    depth_blur_strength: float = 0.4,
    reflection: bool = False,
    reflection_opacity: float = 0.08,
) -> List[dict]:
    """
    Apply multiple tile textures to the same room image, reusing geometry.

    Each item in `tiles` is a dict with keys:
        - "image": np.ndarray (tile texture, BGR)
        - "name":  str (display name, optional)
        - "score": float (similarity score 0–1, optional)

    The expensive pipeline (depth, homography, lighting analysis) is
    computed EXACTLY ONCE via build_geometry_cache(), then reused for
    every tile in the loop.

    Args:
        room_image:           Room photo (H, W, 3), BGR, uint8.
        tiles:                List of tile dicts (see above). Max 5.
        floor_mask:           Binary brush mask (H, W), uint8, 0 or 255.
        device:               'cuda', 'cpu', or 'auto'.
        surface_type:         'floor' or 'wall'.
        pattern:              Tile layout pattern.
        tile_real_size:       Real-world tile size in meters.
        grout_width:          Grout line width pixels.
        grout_color:          Grout BGR color.
        color_match_strength: Color adaptation strength (0–1).
        feather_radius:       Edge feather blur radius.
        pixels_per_meter:     Canvas resolution.
        progress_callback:    Optional callable(current, total, tile_name)
                              for UI progress reporting.

    Returns:
        List of result dicts (same order as input tiles), each containing
        the rendered image and metadata. Returns [] if tiles is empty.
    """
    # ── Edge cases ──────────────────────────────────────────────────────────
    if not tiles:
        logger.warning("[MultiTileEngine] render_multiple_tiles called with empty tile list.")
        return []

    n = len(tiles)
    logger.info(f"[MultiTileEngine] Starting multi-tile render. {n} tile(s).")

    # ── Build shared geometry ONCE ──────────────────────────────────────────
    if progress_callback:
        progress_callback(0, n, "Computing geometry…")

    cache = build_geometry_cache(
        room_image=room_image,
        floor_mask=floor_mask,
        device=device,
        surface_type=surface_type,
        pixels_per_meter=pixels_per_meter,
        feather_radius=feather_radius,
    )

    # ── Render each tile using shared cache ─────────────────────────────────
    results: List[dict] = []
    for i, tile_spec in enumerate(tiles):
        tile_img = tile_spec.get("image")
        if tile_img is None:
            logger.error(f"[MultiTileEngine] Tile {i} has no 'image' key. Skipping.")
            continue

        name = tile_spec.get("name", f"Tile {i + 1}")
        score = tile_spec.get("score", 0.0)

        if progress_callback:
            progress_callback(i + 1, n, f"Rendering {name}…")

        try:
            result = render_single_tile_from_cache(
                room_image=room_image,
                tile_image=tile_img,
                floor_mask=floor_mask,
                cache=cache,
                pattern=pattern,
                tile_real_size=tile_real_size,
                grout_width=grout_width,
                grout_color=grout_color,
                color_match_strength=color_match_strength,
                tile_name=name,
                similarity_score=score,
                realism_strength=realism_strength,
                variation_amount=variation_amount,
                depth_blur_strength=depth_blur_strength,
                reflection=reflection,
                reflection_opacity=reflection_opacity,
            )
            results.append(result)
        except Exception as exc:
            logger.exception(f"[MultiTileEngine] Failed to render tile '{name}': {exc}")
            # Append a failure entry so callers can still show partial results
            results.append({
                "result": None,
                "tile_name": name,
                "similarity_score": score,
                "error": str(exc),
                "timings": {},
            })

    total_wall = cache.build_time_s + sum(
        sum(r.get("timings", {}).values()) for r in results
    )
    logger.info(
        f"[MultiTileEngine] All {len(results)} tile(s) rendered. "
        f"Total wall time: {total_wall:.2f}s "
        f"(geometry: {cache.build_time_s:.2f}s, "
        f"per-tile avg: {(total_wall - cache.build_time_s) / max(len(results), 1):.2f}s)"
    )

    return results
