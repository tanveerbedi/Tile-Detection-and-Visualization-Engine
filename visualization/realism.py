"""
Realism Post-Processing Module
--------------------------------
Enhances tile compositing realism AFTER the core pipeline completes.
All functions are purely post-composite — they never touch homography,
depth maps, or floor masks.

Integration order (inside visualization_engine.py):
    composite()
    → add_tile_variation()   [called on source tile BEFORE create_tile_canvas]
    → depth_darkening()      [applied to composited result]
    → depth_blur()           [applied to composited result]
    → ambient_edge_shadow()  [applied to composited result]
    → apply_reflection()     [optional, applied to composited result]

All ops use vectorized NumPy / OpenCV — no Python-level pixel loops.
"""

import numpy as np
import cv2
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TILE VARIATION INJECTION  (pre-warp, called on source tile)
# ─────────────────────────────────────────────────────────────────────────────

def add_tile_variation(
    tile_image: np.ndarray,
    brightness_range: float = 0.05,
    contrast_range: float = 0.05,
    noise_std: float = 3.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Inject subtle random variation into a tile texture so that repeated copies
    in the grid don't look artificially identical.

    Applied BEFORE create_tile_canvas() on the source tile image.

    Args:
        tile_image:       Tile texture (H, W, 3), BGR, uint8.
        brightness_range: Max absolute brightness shift fraction (e.g. 0.05 = ±5%).
        contrast_range:   Max absolute contrast shift fraction.
        noise_std:        Standard deviation of additive Gaussian noise (pixel values).
        seed:             Optional RNG seed for reproducible results.

    Returns:
        Slightly varied tile texture (H, W, 3), BGR, uint8.
    """
    rng = np.random.default_rng(seed)

    img = tile_image.astype(np.float32)

    # Brightness: multiplicative shift
    brightness_factor = 1.0 + rng.uniform(-brightness_range, brightness_range)
    img = img * brightness_factor

    # Contrast: scale around mean
    if contrast_range > 0:
        mean_val = img.mean()
        contrast_factor = 1.0 + rng.uniform(-contrast_range, contrast_range)
        img = (img - mean_val) * contrast_factor + mean_val

    # Additive Gaussian noise
    if noise_std > 0:
        noise = rng.normal(0, noise_std, img.shape).astype(np.float32)
        img = img + noise

    return np.clip(img, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEPTH-BASED DARKENING  (post-composite)
# ─────────────────────────────────────────────────────────────────────────────

def depth_darkening(
    result: np.ndarray,
    floor_mask: np.ndarray,
    norm_depth: Optional[np.ndarray],
    factor: float = 0.15,
) -> np.ndarray:
    """
    Darken floor tiles in proportion to their distance from the camera
    (i.e. depth value). This simulates natural perspective darkening — objects
    farther away receive slightly less light.

    Effect is confined to the masked floor/wall region.

    Args:
        result:     Composited room image (H, W, 3), BGR, uint8.
        floor_mask: Binary mask (H, W), uint8, 0 or 255.
        norm_depth: Normalised depth map (H, W), float32, [0, 1].
                    If None the function returns result unchanged.
        factor:     Darkening strength. 0 = none, 0.15 = gentle, 0.4 = heavy.

    Returns:
        Darkened result image (H, W, 3), BGR, uint8.
    """
    if norm_depth is None or factor <= 0:
        return result

    h, w = result.shape[:2]

    depth = norm_depth
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_bool = floor_mask > 0
    if not np.any(mask_bool):
        return result

    # Darkening multiplier: near → 1.0, far → (1 - factor)
    darkening = 1.0 - depth * factor          # shape (H, W)
    darkening_3ch = np.stack([darkening] * 3, axis=-1)  # (H, W, 3)

    out = result.astype(np.float32)
    mask_3ch = np.stack([mask_bool] * 3, axis=-1)
    out[mask_3ch] = (out[mask_3ch] * darkening_3ch[mask_3ch])
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DEPTH-BASED BLUR (DOF simulation, post-composite)
# ─────────────────────────────────────────────────────────────────────────────

def depth_blur(
    result: np.ndarray,
    floor_mask: np.ndarray,
    norm_depth: Optional[np.ndarray],
    max_blur_radius: int = 3,
    n_zones: int = 3,
) -> np.ndarray:
    """
    Simulate camera depth-of-field by blurring far tiles more than near ones.

    Uses *layered* blurring (n_zones bands) rather than per-pixel kernels,
    making it fast and vectorized:
      - Zone 0 (near)  → no blur
      - Zone 1 (mid)   → small blur
      - Zone 2 (far)   → larger blur

    Effect is confined to the floor/wall mask region.

    Args:
        result:           Composited image (H, W, 3), BGR, uint8.
        floor_mask:       Binary mask (H, W), uint8, 0 or 255.
        norm_depth:       Normalised depth map (H, W), float32 [0, 1].
                          If None, returned unchanged.
        max_blur_radius:  Blur kernel half-size for the farthest zone (px).
                          Must be >= 1; kernel size = 2*radius+1.
        n_zones:          Number of depth bands. 3 is a good default.

    Returns:
        Depth-blurred result (H, W, 3), BGR, uint8.
    """
    if norm_depth is None or max_blur_radius < 1:
        return result

    h, w = result.shape[:2]

    depth = norm_depth
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_bool = floor_mask > 0
    if not np.any(mask_bool):
        return result

    out = result.copy()

    # Build zone boundaries: 0 = nearest
    thresholds = np.linspace(0, 1, n_zones + 1)

    for zone in range(1, n_zones):          # zone 0 = sharp, skip
        # Blur strength scales linearly with zone index
        radius = int(round(max_blur_radius * zone / (n_zones - 1)))
        if radius < 1:
            continue
        ksize = 2 * radius + 1

        lo, hi = thresholds[zone], thresholds[zone + 1]
        zone_mask = mask_bool & (depth >= lo) & (depth < hi)
        if not np.any(zone_mask):
            continue

        blurred = cv2.GaussianBlur(result, (ksize, ksize), 0)
        out[zone_mask] = blurred[zone_mask]

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. AMBIENT EDGE SHADOW  (post-composite)
# ─────────────────────────────────────────────────────────────────────────────

def ambient_edge_shadow(
    result: np.ndarray,
    floor_mask: np.ndarray,
    shadow_strength: float = 0.25,
    shadow_blur: int = 21,
    dilation_px: int = 5,
) -> np.ndarray:
    """
    Add a subtle contact shadow along the inner boundary of the tile mask.
    This simulates the natural shadowing where tiles meet walls or furniture,
    making the tiled floor look "grounded."

    Implementation:
      1. Dilate mask inward (erode), subtract to get boundary band
      2. Blur the band → soft shadow gradient
      3. Darken result pixels proportionally

    Args:
        result:          Composited image (H, W, 3), BGR, uint8.
        floor_mask:      Binary mask (H, W), uint8, 0 or 255.
        shadow_strength: 0 = none, 0.25 = subtle, 0.6 = heavy.
        shadow_blur:     Gaussian blur kernel size for shadow gradient (must be odd).
        dilation_px:     Width of the shadow band in pixels.

    Returns:
        Shadow-enhanced result (H, W, 3), BGR, uint8.
    """
    if shadow_strength <= 0:
        return result

    shadow_blur = shadow_blur | 1          # ensure odd

    # Erode mask to get interior; boundary = original - eroded
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
    )
    eroded = cv2.erode(floor_mask, kernel, iterations=1)
    boundary = cv2.subtract(floor_mask, eroded)           # uint8, 0 or 255

    # Blur boundary → smooth shadow gradient [0, 255]
    shadow_map = cv2.GaussianBlur(
        boundary.astype(np.float32),
        (shadow_blur, shadow_blur),
        shadow_blur // 3,
    )
    # Normalize to [0, 1]
    shadow_norm = shadow_map / (shadow_map.max() + 1e-6)

    # Darken: multiplier approaches (1 - shadow_strength) at shadow peak
    multiplier = 1.0 - shadow_norm * shadow_strength        # (H, W)
    multiplier_3ch = np.stack([multiplier] * 3, axis=-1)

    out = result.astype(np.float32) * multiplier_3ch
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 5. OPTIONAL GLOSSY REFLECTION  (post-composite)
# ─────────────────────────────────────────────────────────────────────────────

def apply_reflection(
    result: np.ndarray,
    room_image: np.ndarray,
    floor_mask: np.ndarray,
    opacity: float = 0.08,
    blur_radius: int = 25,
) -> np.ndarray:
    """
    Simulate a faint glossy reflection on the tile surface.

    A heavily blurred, vertically-flipped version of the upper portion
    of the room image is alpha-blended onto the floor region at low opacity.
    This recreates the way polished/glazed tiles subtly reflect the room above.

    Args:
        result:      Composited image (H, W, 3), BGR, uint8.
        room_image:  Original room image (H, W, 3), BGR, uint8.
        floor_mask:  Binary mask (H, W), uint8, 0 or 255.
        opacity:     Reflection blend weight (0.05–0.12 looks natural).
        blur_radius: Heavy blur applied to the reflection; must be odd.

    Returns:
        Reflection-blended result (H, W, 3), BGR, uint8.
    """
    if opacity <= 0:
        return result

    blur_radius = blur_radius | 1

    h, w = result.shape[:2]

    # Use top half of room as reflection source
    src = room_image[:h // 2, :]
    # Flip vertically so the "ceiling" appears near the camera
    src_flipped = cv2.flip(src, 0)
    # Stretch to full image height
    src_full = cv2.resize(src_flipped, (w, h), interpolation=cv2.INTER_LINEAR)
    # Heavy blur → painterly, non-distracting reflection
    reflection = cv2.GaussianBlur(src_full, (blur_radius, blur_radius), blur_radius // 3)

    mask_bool = floor_mask > 0

    out = result.astype(np.float32)
    ref = reflection.astype(np.float32)
    out[mask_bool] = out[mask_bool] * (1 - opacity) + ref[mask_bool] * opacity
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def enhance_realism(
    result: np.ndarray,
    room_image: np.ndarray,
    floor_mask: np.ndarray,
    norm_depth: Optional[np.ndarray] = None,
    realism_strength: float = 0.5,
    variation_amount: float = 0.05,   # kept here for docstring clarity; actual variation applied pre-warp
    depth_blur_strength: float = 0.4,
    reflection: bool = False,
    reflection_opacity: float = 0.08,
) -> np.ndarray:
    """
    Apply all active realism enhancements to the composited result image.

    Call this AFTER composite() in the visualization pipeline. This function
    never modifies homography, depth maps, or floor masks.

    Args:
        result:              Composited room image (H, W, 3), BGR, uint8.
        room_image:          Original (unmodified) room photo (H, W, 3), BGR, uint8.
        floor_mask:          Binary brush mask (H, W), uint8, 0 or 255.
        norm_depth:          Normalised depth map (H, W), float32 [0, 1], or None.
        realism_strength:    Global scale factor [0, 1].
                             0 = skip everything, 1 = full effect.
        variation_amount:    Tile brightness/contrast variation (documented here;
                             applied pre-warp via add_tile_variation()).
        depth_blur_strength: How many blur layers to use (fed into max_blur_radius).
        reflection:          If True, blend a faint glossy reflection.
        reflection_opacity:  Reflection alpha (only used if reflection=True).

    Returns:
        Realism-enhanced image (H, W, 3), BGR, uint8.
    """
    if realism_strength <= 0:
        return result

    s = realism_strength   # shorthand scale

    out = result

    # ── Step 1: Depth darkening ────────────────────────────────────────────
    out = depth_darkening(
        out,
        floor_mask,
        norm_depth,
        factor=0.15 * s,
    )

    # ── Step 2: Depth-of-field blur ────────────────────────────────────────
    max_blur = max(1, int(round(depth_blur_strength * 4 * s)))  # 1–4 px max
    out = depth_blur(
        out,
        floor_mask,
        norm_depth,
        max_blur_radius=max_blur,
        n_zones=3,
    )

    # ── Step 3: Ambient edge shadow ────────────────────────────────────────
    out = ambient_edge_shadow(
        out,
        floor_mask,
        shadow_strength=0.30 * s,
        shadow_blur=21,
        dilation_px=max(3, int(8 * s)),
    )

    # ── Step 4: Optional glossy reflection ────────────────────────────────
    if reflection:
        out = apply_reflection(
            out,
            room_image,
            floor_mask,
            opacity=reflection_opacity * s,
            blur_radius=25,
        )

    logger.info(
        f"[Realism] enhance_realism done. "
        f"strength={s:.2f}, depth_blur_max={max_blur}px, "
        f"reflection={reflection}"
    )
    return out
