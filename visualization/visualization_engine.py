"""
Visualization Engine — Orchestrator
------------------------------------
Combines all visualization sub-modules into a single pipeline:

    Room Image + Tile Image → Floor Mask → Depth Map → Homography
    → Tile Canvas → Warp → Light → Blend → Final Result

Usage:
    from visualization import apply_tile_to_room

    result = apply_tile_to_room(
        room_image=room_bgr,
        tile_image=tile_bgr,
        click_point=(300, 400),
        pattern='brick',
        device='auto'
    )
    final_image = result['result']
"""

import numpy as np
import cv2
import time
import logging

from .floor_segmentation import (
    load_segmentation_model,
    segment_floor,
    refine_mask_with_click,
    segment_floor_with_depth,
    clean_mask,
)
from .depth_estimation import (
    load_midas_model,
    estimate_depth,
    normalize_depth,
    get_depth_scale_map,
    visualize_depth,
)
from .plane_geometry import (
    estimate_camera_intrinsics,
    sample_floor_3d_points,
    fit_plane_ransac,
    estimate_homography,
    compute_vanishing_point,
)
from .tile_renderer import (
    create_tile_canvas,
    warp_tile_canvas,
    apply_floor_mask,
)
from .lighting_blender import (
    extract_brightness_map,
    apply_lighting,
    extract_color_statistics,
    color_match_tile,
    feather_edges,
    composite,
)
from .realism import enhance_realism, add_tile_variation

logger = logging.getLogger(__name__)


def apply_tile_to_room(
    room_image: np.ndarray,
    tile_image: np.ndarray,
    click_point: tuple = None,
    pattern: str = "straight",
    tile_real_size: float = 0.6,
    grout_width: int = 2,
    grout_color: tuple = (60, 60, 60),
    feather_radius: int = 15,
    color_match_strength: float = 0.25,
    device: str = "auto",
) -> dict:
    """
    Apply a tile texture onto the floor of a room image.

    Full pipeline:
    1. Estimate depth map (needed for both segmentation and perspective)
    2. Segment floor (color flood fill + depth refinement + object exclusion)
    3. Estimate homography from floor quad shape
    4. Generate tiled canvas with chosen pattern
    5. Warp canvas into room perspective
    6. Match lighting and blend

    Args:
        room_image: Room photo (H, W, 3), BGR, uint8.
        tile_image: Tile texture (H, W, 3), BGR, uint8.
        click_point: (x, y) where user clicked on the floor.
        pattern: 'straight', 'diagonal', or 'brick'.
        tile_real_size: Real-world tile size in meters.
        grout_width: Grout line width in pixels.
        grout_color: Grout BGR color.
        feather_radius: Edge feathering blur radius.
        color_match_strength: Tile color adaptation strength (0-1).
        device: 'cuda', 'cpu', or 'auto'.

    Returns:
        Dict with result image and all intermediate outputs.
    """
    timings = {}
    h, w = room_image.shape[:2]
    logger.info(f"Pipeline start. Image: {w}x{h}, pattern: {pattern}, device: {device}")

    # ==================== STEP 1: Depth Estimation ====================
    t0 = time.time()
    logger.info("Step 1/6: Depth estimation (MiDaS)...")

    midas_model, midas_transform, midas_device = load_midas_model(device)
    raw_depth = estimate_depth(room_image, midas_model, midas_transform, midas_device)
    norm_depth = normalize_depth(raw_depth)
    depth_colored = visualize_depth(raw_depth)

    timings["depth_estimation"] = time.time() - t0
    logger.info(f"  Depth estimated in {timings['depth_estimation']:.2f}s")

    # ==================== STEP 2: Floor Segmentation ====================
    t0 = time.time()
    logger.info("Step 2/6: Floor segmentation...")

    seg_model, seg_device = load_segmentation_model(device)

    # Use depth-assisted segmentation with click point
    floor_mask = segment_floor_with_depth(
        room_image, norm_depth, click_point, seg_model, seg_device
    )

    timings["floor_segmentation"] = time.time() - t0
    floor_coverage = np.sum(floor_mask > 0) / (h * w) * 100
    logger.info(f"  Floor coverage: {floor_coverage:.1f}%, time: {timings['floor_segmentation']:.2f}s")

    # Validate floor mask — if too small or too large, something is wrong
    if floor_coverage < 2.0:
        logger.warning("Floor mask too small! Check click point or image.")
    elif floor_coverage > 80.0:
        logger.warning("Floor mask covers >80% of image — likely incorrect. "
                        "Applying spatial constraint.")
        # Emergency spatial constraint: only keep bottom 65% max
        emergency = np.zeros_like(floor_mask)
        emergency[int(h * 0.35):, :] = 255
        floor_mask = cv2.bitwise_and(floor_mask, emergency)
        floor_mask = clean_mask(floor_mask)
        floor_coverage = np.sum(floor_mask > 0) / (h * w) * 100

    # ==================== STEP 3: Homography ====================
    t0 = time.time()
    logger.info("Step 3/6: Plane geometry & homography...")

    pixels_per_meter = 400.0
    homography, canvas_size = estimate_homography(
        floor_mask, norm_depth, pixels_per_meter=pixels_per_meter
    )

    timings["plane_geometry"] = time.time() - t0

    # ==================== STEP 4: Tile Grid ====================
    t0 = time.time()
    logger.info(f"Step 4/6: Tile grid ({pattern} pattern, canvas {canvas_size})...")

    tile_canvas = create_tile_canvas(
        tile_image, canvas_size, pattern=pattern,
        tile_real_size=tile_real_size,
        pixels_per_meter=pixels_per_meter,
        grout_width=grout_width,
        grout_color=grout_color
    )

    # Warp into perspective
    warped_tiles = warp_tile_canvas(tile_canvas, homography, (h, w))

    # We purposefully do NOT apply a hard bitwise_and floor mask here anymore!
    # The canvas is mathematically designed to be larger than the floor_mask
    # bounding box. We pass the fully warped tiles down to compositing,
    # which uses a feathered soft-alpha mask to prevent harsh jagged edges.
    masked_tiles = warped_tiles

    timings["tile_rendering"] = time.time() - t0
    logger.info(f"  Tile rendering done in {timings['tile_rendering']:.2f}s")

    # ==================== STEP 5: Lighting ====================
    t0 = time.time()
    logger.info("Step 5/6: Lighting matching...")

    brightness_map = extract_brightness_map(room_image, floor_mask)
    lit_tiles = apply_lighting(masked_tiles, brightness_map, floor_mask)

    floor_stats = extract_color_statistics(room_image, floor_mask)
    color_matched = color_match_tile(lit_tiles, floor_mask, floor_stats,
                                     strength=color_match_strength)

    timings["lighting"] = time.time() - t0

    # ==================== STEP 6: Compositing ====================
    t0 = time.time()
    logger.info("Step 6/6: Final compositing...")

    feathered_mask = feather_edges(floor_mask, blur_radius=feather_radius)
    result = composite(room_image, color_matched, feathered_mask)

    timings["compositing"] = time.time() - t0

    total = sum(timings.values())
    logger.info(f"Pipeline complete in {total:.2f}s")

    return {
        "result": result,
        "floor_mask": floor_mask,
        "depth_map": norm_depth,
        "depth_colored": depth_colored,
        "warped_tiles": warped_tiles,
        "masked_tiles": masked_tiles,
        "homography": homography,
        "timings": timings,
        "canvas_size": canvas_size,
        "brightness_map": brightness_map,
    }


def apply_tile_to_room_with_mask(
    room_image: np.ndarray,
    tile_image: np.ndarray,
    floor_mask: np.ndarray,
    pattern: str = "straight",
    tile_real_size: float = 0.6,
    grout_width: int = 2,
    grout_color: tuple = (60, 60, 60),
    feather_radius: int = 15,
    color_match_strength: float = 0.25,
    device: str = "auto",
    surface_type: str = "floor",
    # Realism enhancements (post-composite, does not touch geometry)
    realism_strength: float = 0.0,
    variation_amount: float = 0.05,
    depth_blur_strength: float = 0.4,
    reflection: bool = False,
    reflection_opacity: float = 0.08,
) -> dict:
    """
    Apply tile to room using a user-provided mask (from brush tool).

    Pipeline:
    1. Use the provided mask as-is
    2. Estimate depth map for perspective (if floor)
    3. Compute homography (3D if floor, 2D affine if wall)
    4. Generate + warp tile canvas
    5. Apply lighting and blend
    """
    timings = {}
    h, w = room_image.shape[:2]
    logger.info(f"Pipeline (brush mask). Image: {w}x{h}, pattern: {pattern}, surface: {surface_type}")

    # Ensure mask matches image size
    if floor_mask.shape[:2] != (h, w):
        floor_mask = cv2.resize(floor_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    floor_coverage = np.sum(floor_mask > 0) / (h * w) * 100
    logger.info(f"Brush mask coverage: {floor_coverage:.1f}%")

    # ==================== STEP 1: Depth (Optional) ====================
    t0 = time.time()
    
    pixels_per_meter = 400.0
    norm_depth = None
    depth_colored = None
    
    if surface_type == "floor":
        logger.info("Step 1/5: Depth estimation...")
        midas_model, midas_transform, midas_device = load_midas_model(device)
        raw_depth = estimate_depth(room_image, midas_model, midas_transform, midas_device)
        norm_depth = normalize_depth(raw_depth)
        depth_colored = visualize_depth(raw_depth)
    else:
        logger.info("Step 1/5: Skipping depth estimation for Wall surface.")

    timings["depth_estimation"] = time.time() - t0

    # ==================== STEP 2: Homography ====================
    t0 = time.time()
    logger.info("Step 2/5: Homography from shape...")

    if surface_type == "floor":
        from .plane_geometry import estimate_homography
        homography, canvas_size = estimate_homography(
            floor_mask, norm_depth, pixels_per_meter=pixels_per_meter
        )
    else:
        from .plane_geometry import estimate_wall_homography
        homography, canvas_size = estimate_wall_homography(
            floor_mask, pixels_per_meter=pixels_per_meter
        )

    timings["plane_geometry"] = time.time() - t0

    # ==================== STEP 3: Tile Grid ====================
    t0 = time.time()
    logger.info(f"Step 3/5: Tile grid ({pattern}, canvas {canvas_size})...")

    # ── Realism: tile variation (pre-warp, on source tile) ──────────────────
    if realism_strength > 0 and variation_amount > 0:
        tile_image_render = add_tile_variation(
            tile_image,
            brightness_range=variation_amount,
            contrast_range=variation_amount,
            noise_std=3.0 * realism_strength,
        )
        logger.info("Step 3/5 (realism): tile variation injected.")
    else:
        tile_image_render = tile_image

    tile_canvas = create_tile_canvas(
        tile_image_render, canvas_size, pattern=pattern,
        tile_real_size=tile_real_size,
        pixels_per_meter=pixels_per_meter,
        grout_width=grout_width,
        grout_color=grout_color
    )

    warped_tiles = warp_tile_canvas(tile_canvas, homography, (h, w))
    
    # We purposefully do NOT apply a hard bitwise_and floor mask here anymore!
    masked_tiles = warped_tiles

    timings["tile_rendering"] = time.time() - t0

    # ==================== STEP 4: Lighting ====================
    t0 = time.time()
    logger.info("Step 4/5: Lighting...")

    brightness_map = extract_brightness_map(room_image, floor_mask)
    lit_tiles = apply_lighting(masked_tiles, brightness_map, floor_mask)

    floor_stats = extract_color_statistics(room_image, floor_mask)
    color_matched = color_match_tile(
        lit_tiles, floor_mask, floor_stats, strength=color_match_strength
    )

    timings["lighting"] = time.time() - t0

    # ==================== STEP 5: Composite ====================
    t0 = time.time()
    logger.info("Step 5/5: Compositing...")

    feathered_mask = feather_edges(floor_mask, blur_radius=feather_radius)
    result = composite(room_image, color_matched, feathered_mask)

    # ── Realism: post-composite enhancements ────────────────────────────────
    if realism_strength > 0:
        t_r = time.time()
        result = enhance_realism(
            result,
            room_image,
            floor_mask,
            norm_depth=norm_depth,
            realism_strength=realism_strength,
            variation_amount=variation_amount,
            depth_blur_strength=depth_blur_strength,
            reflection=reflection,
            reflection_opacity=reflection_opacity,
        )
        timings["realism"] = time.time() - t_r
        logger.info(f"Step 5/5 (realism): enhancements applied in {timings['realism']:.2f}s")

    timings["compositing"] = time.time() - t0

    total = sum(timings.values())
    logger.info(f"Pipeline complete in {total:.2f}s")

    return {
        "result": result,
        "floor_mask": floor_mask,
        "depth_map": norm_depth,
        "depth_colored": depth_colored,
        "warped_tiles": warped_tiles,
        "masked_tiles": masked_tiles,
        "homography": homography,
        "timings": timings,
        "canvas_size": canvas_size,
        "brightness_map": brightness_map,
    }


def preload_models(device: str = "auto"):
    """Pre-load all heavy models into memory."""
    logger.info("Pre-loading models...")
    load_segmentation_model(device)
    load_midas_model(device)
    logger.info("All models loaded.")
