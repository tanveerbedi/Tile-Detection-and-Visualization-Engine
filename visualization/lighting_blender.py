"""
Lighting & Blending Module
---------------------------
Matches the tile texture to the room's lighting conditions and
composites it seamlessly onto the original image.

Pipeline:
1. **Brightness Map Extraction**: Convert the original floor region to
   grayscale and normalize to capture lighting variations (shadows,
   highlights, ambient light gradients).

2. **Intensity Normalization**: Apply the brightness map to the tile
   texture via multiplication, so tiles in shadowy areas appear darker
   and tiles in well-lit areas appear brighter.

3. **Edge Feathering**: Apply Gaussian blur to the floor mask edges
   to create soft transitions between tiled and non-tiled regions.
   This prevents hard "pasted-on" edges.

4. **Compositing**: Alpha-blend the lit tile texture with the original
   image using the feathered mask:
       result = original × (1 - alpha) + lit_tile × alpha
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


def extract_brightness_map(
    image: np.ndarray,
    floor_mask: np.ndarray
) -> np.ndarray:
    """
    Extract a brightness/luminance map from the floor region of the original image.

    This captures the lighting conditions on the floor:
    - Shadow patterns
    - Light gradients (e.g., light from a window)
    - Overall brightness level

    The map is normalized so that the average brightness = 1.0,
    making it a multiplicative modifier for the tile texture.

    Args:
        image: Original BGR image (H, W, 3), uint8.
        floor_mask: Binary mask (H, W), uint8, 0 or 255.

    Returns:
        Brightness map (H, W), float32, centered around 1.0.
    """
    # Convert to LAB color space for perceptually uniform lightness
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)  # L channel, range 0-255

    # Normalize within floor region
    floor_pixels = floor_mask > 0

    if not np.any(floor_pixels):
        return np.ones(image.shape[:2], dtype=np.float32)

    # Mean brightness in floor region
    mean_brightness = np.mean(lightness[floor_pixels])

    if mean_brightness < 1e-6:
        return np.ones(image.shape[:2], dtype=np.float32)

    # Create brightness map normalized around 1.0
    brightness_map = lightness / mean_brightness

    # Smooth the brightness map to avoid noise artifacts
    brightness_map = cv2.GaussianBlur(brightness_map, (21, 21), 5)

    # Clamp to reasonable range
    brightness_map = np.clip(brightness_map, 0.3, 2.0)

    return brightness_map


def apply_lighting(
    tile_region: np.ndarray,
    brightness_map: np.ndarray,
    floor_mask: np.ndarray
) -> np.ndarray:
    """
    Apply the brightness map to the tile texture to match room lighting.

    Each pixel in the tile texture is multiplied by the corresponding
    brightness value. This makes the tiles look naturally lit - darker
    in shadows, brighter under lights.

    Args:
        tile_region: Tile image (H, W, 3), BGR, uint8.
        brightness_map: Brightness map (H, W), float32, centered at 1.0.
        floor_mask: Binary mask (H, W), uint8.

    Returns:
        Lit tile image (H, W, 3), BGR, uint8.
    """
    # Ensure same shape
    if brightness_map.shape[:2] != tile_region.shape[:2]:
        brightness_map = cv2.resize(brightness_map,
                                     (tile_region.shape[1], tile_region.shape[0]))

    # Convert tile to float for multiplication
    tile_float = tile_region.astype(np.float32)

    # Apply brightness per channel
    brightness_3ch = np.stack([brightness_map] * 3, axis=-1)
    lit_tile = tile_float * brightness_3ch

    # Clamp to valid range
    lit_tile = np.clip(lit_tile, 0, 255).astype(np.uint8)

    # We NO LONGER apply a hard bitwise_and mask here to preserve soft edges during compositing.
    return lit_tile


def extract_color_statistics(
    image: np.ndarray,
    floor_mask: np.ndarray
) -> dict:
    """
    Extract color statistics from the floor region for color matching.

    This helps adjust the tile's color temperature to better match
    the room's ambient color (warm/cool lighting).

    Args:
        image: Original BGR image (H, W, 3), uint8.
        floor_mask: Binary mask (H, W), uint8.

    Returns:
        Dict with 'mean_color' and 'std_color' in LAB space.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    floor_pixels = floor_mask > 0

    if not np.any(floor_pixels):
        return {"mean_color": np.array([128, 128, 128]), "std_color": np.array([30, 30, 30])}

    floor_lab = lab[floor_pixels]

    return {
        "mean_color": np.mean(floor_lab, axis=0),
        "std_color": np.std(floor_lab, axis=0)
    }


def color_match_tile(
    tile_region: np.ndarray,
    floor_mask: np.ndarray,
    floor_stats: dict,
    strength: float = 0.3
) -> np.ndarray:
    """
    Subtly adjust tile color temperature to match the room's lighting.

    Uses LAB color space transfer with a configurable strength parameter
    to avoid over-processing while ensuring color coherence.

    Args:
        tile_region: Tile image (H, W, 3), BGR, uint8.
        floor_mask: Binary mask (H, W), uint8.
        floor_stats: Dict from extract_color_statistics.
        strength: Blending strength (0.0 = no change, 1.0 = full match).

    Returns:
        Color-adjusted tile image (H, W, 3), BGR, uint8.
    """
    if not np.any(floor_mask > 0):
        return tile_region

    # Convert to LAB
    tile_lab = cv2.cvtColor(tile_region, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask_bool = floor_mask > 0

    # Get tile statistics in floor region
    tile_floor = tile_lab[mask_bool]
    if len(tile_floor) == 0:
        return tile_region

    tile_mean = np.mean(tile_floor, axis=0)
    tile_std = np.std(tile_floor, axis=0) + 1e-6

    floor_mean = floor_stats["mean_color"]
    floor_std = floor_stats["std_color"] + 1e-6

    # Apply partial color transfer (only a/b channels, keep L for brightness map)
    for ch in [1, 2]:  # a and b channels only
        tile_lab[:, :, ch] = (
            tile_lab[:, :, ch] * (1 - strength)
            + ((tile_lab[:, :, ch] - tile_mean[ch]) * (floor_std[ch] / tile_std[ch]) + floor_mean[ch]) * strength
        )

    tile_lab = np.clip(tile_lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(tile_lab, cv2.COLOR_LAB2BGR)

    # We NO LONGER apply a hard bitwise_and mask here to preserve soft edges during compositing.
    return result


def feather_edges(
    mask: np.ndarray,
    blur_radius: int = 15,
    edge_width: int = 20
) -> np.ndarray:
    """
    Create a feathered (soft-edged) version of the floor mask.

    The feathering creates a smooth gradient at the mask boundary,
    so the composite doesn't have sharp artificial edges.

    Process:
    1. Erode the mask slightly to create an inner boundary
    2. Compute the edge band (original - eroded)
    3. Apply Gaussian blur to the mask
    4. The result has full opacity inside, gradual fadeout at edges

    Args:
        mask: Binary mask (H, W), uint8, 0 or 255.
        blur_radius: Gaussian blur kernel radius (must be odd).
        edge_width: Width of the feathered edge in pixels.

    Returns:
        Feathered mask (H, W), float32, range [0, 1].
    """
    # Ensure odd kernel size
    blur_radius = blur_radius | 1

    # Convert to float
    mask_float = mask.astype(np.float32) / 255.0

    # Erode to create boundary region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width, edge_width))
    eroded = cv2.erode(mask_float, kernel, iterations=1)

    # Blend: full inside, gradual at edges
    # Use Gaussian blur on the original mask for smooth transition
    feathered = cv2.GaussianBlur(mask_float, (blur_radius, blur_radius), blur_radius // 3)

    # Combine: keep full opacity inside, use blurred at edges
    result = np.maximum(eroded, feathered)

    return np.clip(result, 0, 1).astype(np.float32)


def composite(
    original: np.ndarray,
    tile_region: np.ndarray,
    feathered_mask: np.ndarray
) -> np.ndarray:
    """
    Final compositing step: blend lit tile texture with original image.

    The compositing formula is:
        result = original × (1 - alpha) + tile × alpha

    Where alpha is the feathered mask. This produces:
    - Full tile visibility inside the floor
    - Smooth blending at floor boundaries
    - Original image preserved outside the floor

    Args:
        original: Original room image (H, W, 3), BGR, uint8.
        tile_region: Lit and masked tile image (H, W, 3), BGR, uint8.
        feathered_mask: Feathered mask (H, W), float32, [0, 1].

    Returns:
        Final composited image (H, W, 3), BGR, uint8.
    """
    # Ensure matching dimensions
    if tile_region.shape[:2] != original.shape[:2]:
        tile_region = cv2.resize(tile_region, (original.shape[1], original.shape[0]))
    if feathered_mask.shape[:2] != original.shape[:2]:
        feathered_mask = cv2.resize(feathered_mask, (original.shape[1], original.shape[0]))

    # Convert to float for blending
    orig_float = original.astype(np.float32)
    tile_float = tile_region.astype(np.float32)

    # 3-channel alpha
    alpha = np.stack([feathered_mask] * 3, axis=-1)

    # Handle zero tile pixels (outside the warped region) — keep original
    tile_nonzero = np.any(tile_float > 0, axis=-1)
    alpha[~tile_nonzero] = 0

    # Composite
    result = orig_float * (1 - alpha) + tile_float * alpha

    return np.clip(result, 0, 255).astype(np.uint8)
