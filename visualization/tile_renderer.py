"""
Tile Renderer Module
--------------------
Generates tiled texture canvases with configurable patterns,
applies perspective warping via homography, and masks to floor region.

Supported Tile Patterns:
    1. **Straight**: Standard grid layout, tiles aligned to axes
    2. **Diagonal**: 45° rotated diamond pattern
    3. **Brick**: Offset rows (every other row shifted by half tile width)

Depth-Aware Scaling:
    Tile size is modulated by the depth scale map. Tiles farther from
    the camera appear smaller, matching the natural perspective of the scene.
    This is achieved through the homography warp rather than per-tile scaling,
    which produces smoother results.
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


def create_tile_canvas(
    tile_img: np.ndarray,
    canvas_size: tuple,
    pattern: str = "straight",
    tile_real_size: float = 0.6,
    pixels_per_meter: float = 400.0,
    grout_width: int = 2,
    grout_color: tuple = (60, 60, 60)
) -> np.ndarray:
    """
    Create a large tiled canvas by repeating a tile image in the specified pattern.

    The canvas represents a bird's-eye (top-down) view of the tiled floor
    before perspective warping. The homography will transform this into
    the correct perspective.

    Args:
        tile_img: Tile texture image (H, W, 3), BGR, uint8.
        canvas_size: (width, height) of the output canvas.
        pattern: 'straight', 'diagonal', or 'brick'.
        tile_real_size: Real-world tile size in meters.
        pixels_per_meter: Scale mapping meters to canvas pixels.
        grout_width: Width of grout lines between tiles in pixels.
        grout_color: BGR color of the grout lines.

    Returns:
        Tiled canvas (canvas_h, canvas_w, 3), BGR, uint8.
    """
    canvas_w, canvas_h = canvas_size

    # Resize tile to the exact physical size specified by pixels_per_meter
    tile_pixel_size = max(10, int(tile_real_size * pixels_per_meter))
    tile = cv2.resize(tile_img, (tile_pixel_size, tile_pixel_size), interpolation=cv2.INTER_AREA)
    th, tw = tile.shape[:2]

    if pattern == "diagonal":
        return _create_diagonal_canvas(tile, canvas_w, canvas_h, grout_width, grout_color)
    elif pattern == "brick":
        return _create_brick_canvas(tile, canvas_w, canvas_h, grout_width, grout_color)
    else:
        return _create_straight_canvas(tile, canvas_w, canvas_h, grout_width, grout_color)


def _create_straight_canvas(
    tile: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    grout_width: int,
    grout_color: tuple
) -> np.ndarray:
    """
    Standard grid pattern — tiles aligned in a regular rectangular grid.

    Layout:
    ┌────┬────┬────┬────┐
    │    │    │    │    │
    ├────┼────┼────┼────┤
    │    │    │    │    │
    ├────┼────┼────┼────┤
    │    │    │    │    │
    └────┴────┴────┴────┘
    """
    th, tw = tile.shape[:2]
    step_x = tw + grout_width
    step_y = th + grout_width

    canvas = np.full((canvas_h, canvas_w, 3), grout_color, dtype=np.uint8)

    for y in range(0, canvas_h, step_y):
        for x in range(0, canvas_w, step_x):
            # Calculate placement region
            y_end = min(y + th, canvas_h)
            x_end = min(x + tw, canvas_w)
            tile_h = y_end - y
            tile_w = x_end - x

            canvas[y:y_end, x:x_end] = tile[:tile_h, :tile_w]

    return canvas


def _create_diagonal_canvas(
    tile: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    grout_width: int,
    grout_color: tuple
) -> np.ndarray:
    """
    Diagonal (diamond) pattern — tiles rotated 45°.

    Layout:
      ◇  ◇  ◇
    ◇  ◇  ◇  ◇
      ◇  ◇  ◇
    ◇  ◇  ◇  ◇
    """
    th, tw = tile.shape[:2]

    # Rotate tile 45 degrees
    center = (tw // 2, th // 2)
    M_rot = cv2.getRotationMatrix2D(center, 45, 1.0)

    # Calculate new bounding box for rotated tile
    cos_a = abs(M_rot[0, 0])
    sin_a = abs(M_rot[0, 1])
    new_w = int(th * sin_a + tw * cos_a)
    new_h = int(th * cos_a + tw * sin_a)

    M_rot[0, 2] += (new_w - tw) / 2
    M_rot[1, 2] += (new_h - th) / 2

    rotated = cv2.warpAffine(tile, M_rot, (new_w, new_h),
                              borderMode=cv2.BORDER_REFLECT)

    rh, rw = rotated.shape[:2]
    step_x = rw + grout_width
    step_y = rh + grout_width

    canvas = np.full((canvas_h, canvas_w, 3), grout_color, dtype=np.uint8)

    for row, y in enumerate(range(-rh, canvas_h + rh, step_y)):
        offset = (rw // 2 + grout_width // 2) if row % 2 else 0
        for x in range(-rw + offset, canvas_w + rw, step_x):
            y_start = max(0, y)
            x_start = max(0, x)
            y_end = min(y + rh, canvas_h)
            x_end = min(x + rw, canvas_w)

            tile_y_start = y_start - y
            tile_x_start = x_start - x
            tile_y_end = tile_y_start + (y_end - y_start)
            tile_x_end = tile_x_start + (x_end - x_start)

            if tile_y_end > rh or tile_x_end > rw:
                continue

            if y_end > y_start and x_end > x_start:
                canvas[y_start:y_end, x_start:x_end] = \
                    rotated[tile_y_start:tile_y_end, tile_x_start:tile_x_end]

    return canvas


def _create_brick_canvas(
    tile: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    grout_width: int,
    grout_color: tuple
) -> np.ndarray:
    """
    Brick (running bond) pattern — every other row offset by half tile width.

    Layout:
    ┌────────┬────────┬────────┐
    │        │        │        │
    ├────┬───┴────┬───┴────┬───┤
    │    │        │        │   │
    ├────┴───┬────┴───┬────┴───┤
    │        │        │        │
    └────────┴────────┴────────┘
    """
    th, tw = tile.shape[:2]
    step_x = tw + grout_width
    step_y = th + grout_width

    canvas = np.full((canvas_h, canvas_w, 3), grout_color, dtype=np.uint8)

    row_idx = 0
    for y in range(0, canvas_h, step_y):
        # Offset every other row by half tile width
        x_offset = (tw // 2 + grout_width // 2) if row_idx % 2 else 0

        for x in range(-tw + x_offset, canvas_w + tw, step_x):
            y_start = max(0, y)
            x_start = max(0, x)
            y_end = min(y + th, canvas_h)
            x_end = min(x + tw, canvas_w)

            tile_y_start = y_start - y
            tile_x_start = x_start - x
            tile_y_end = tile_y_start + (y_end - y_start)
            tile_x_end = tile_x_start + (x_end - x_start)

            if tile_y_end > th or tile_x_end > tw:
                continue

            if y_end > y_start and x_end > x_start:
                canvas[y_start:y_end, x_start:x_end] = \
                    tile[tile_y_start:tile_y_end, tile_x_start:tile_x_end]

        row_idx += 1

    return canvas


def warp_tile_canvas(
    canvas: np.ndarray,
    homography: np.ndarray,
    output_shape: tuple
) -> np.ndarray:
    """
    Warp the tile canvas using the homography matrix.

    This transforms the bird's-eye tile grid into the perspective
    view matching the room image. The homography encodes:
    - Tile convergence toward vanishing point
    - Size reduction with depth (farther tiles appear smaller)
    - Correct grid line perspective

    Args:
        canvas: Tiled canvas (H, W, 3), BGR, uint8.
        homography: 3x3 homography matrix.
        output_shape: (height, width) of the output image.

    Returns:
        Warped canvas (output_h, output_w, 3), BGR, uint8.
        Pixels outside the warp are set to 0 (black).
    """
    out_h, out_w = output_shape[:2]

    warped = cv2.warpPerspective(
        canvas,
        homography,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return warped


def apply_floor_mask(
    warped_canvas: np.ndarray,
    floor_mask: np.ndarray
) -> np.ndarray:
    """
    Clip the warped tile canvas to the floor region.

    Only pixels within the floor mask are kept; everything else
    is set to transparent (black).

    Args:
        warped_canvas: Warped tile image (H, W, 3), BGR, uint8.
        floor_mask: Binary mask (H, W), uint8, 0 or 255.

    Returns:
        Masked tile region (H, W, 3), BGR, uint8.
    """
    # Ensure same dimensions
    if warped_canvas.shape[:2] != floor_mask.shape[:2]:
        floor_mask = cv2.resize(floor_mask, (warped_canvas.shape[1], warped_canvas.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

    # Apply mask
    mask_3ch = cv2.merge([floor_mask, floor_mask, floor_mask])
    masked = cv2.bitwise_and(warped_canvas, mask_3ch)

    return masked
