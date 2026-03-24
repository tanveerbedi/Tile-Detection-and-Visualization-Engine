"""
Depth Estimation Module
-----------------------
Uses MiDaS (DPT-Large) to generate per-pixel depth maps.

Depth maps are used for:
1. Determining tile scale at different image positions (farther = smaller)
2. Estimating the floor plane via RANSAC in 3D space
3. Computing a homography for perspective-correct tile mapping

The depth from MiDaS is *inverse depth* (closer = larger value).
We normalize it to a 0-1 range for downstream use.

GPU/CPU Compatibility:
    - Automatically detects CUDA and uses GPU if available.
    - Falls back to CPU gracefully (slower but functional).
    - Model is loaded once and cached in memory.
"""

import numpy as np
import cv2
import torch
import logging

logger = logging.getLogger(__name__)

# ========================= MODEL CACHE =========================

_midas_model = None
_midas_transform = None
_midas_device = None


def load_midas_model(device: str = "auto"):
    """
    Load MiDaS DPT-Large model via torch.hub.

    MiDaS produces monocular depth estimation — a per-pixel depth
    value from a single RGB image. The DPT-Large variant uses a
    Vision Transformer backbone for highest accuracy.

    Args:
        device: 'cuda', 'cpu', or 'auto'.

    Returns:
        Tuple of (model, transform, device_string).
    """
    global _midas_model, _midas_transform, _midas_device

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _midas_model is not None and _midas_device == device:
        return _midas_model, _midas_transform, _midas_device

    logger.info(f"Loading MiDaS DPT-Large on {device}...")
    logger.info("This may take a moment on first run (downloading ~500MB)...")

    # Load DPT-Large — highest quality depth estimation
    _midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
    _midas_model.eval().to(device)

    # Load the corresponding transform
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    _midas_transform = midas_transforms.dpt_transform

    _midas_device = device

    logger.info("MiDaS model loaded successfully.")
    return _midas_model, _midas_transform, _midas_device


# ========================= DEPTH ESTIMATION =========================

def estimate_depth(
    image: np.ndarray,
    model=None,
    transform=None,
    device: str = "auto"
) -> np.ndarray:
    """
    Generate a per-pixel depth map from a single RGB image.

    MiDaS outputs *inverse depth* — larger values mean closer.
    We return the raw inverse depth map at the original image resolution.

    Args:
        image: BGR numpy array (H, W, 3).
        model: Pre-loaded MiDaS model (or None to auto-load).
        transform: Pre-loaded MiDaS transform.
        device: Device string.

    Returns:
        Depth map (H, W), float32, raw MiDaS output.
    """
    if model is None or transform is None:
        model, transform, device = load_midas_model(device)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply MiDaS transform
    input_batch = transform(img_rgb).to(device)

    # Inference
    with torch.no_grad():
        prediction = model(input_batch)

    # Resize to original image dimensions
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy().astype(np.float32)

    return depth_map


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """
    Normalize depth map to 0–1 range.

    After normalization:
    - 1.0 = closest to camera (largest inverse depth)
    - 0.0 = farthest from camera (smallest inverse depth)

    Args:
        depth_map: Raw depth map from MiDaS (H, W), float32.

    Returns:
        Normalized depth map (H, W), float32, range [0, 1].
    """
    d_min = depth_map.min()
    d_max = depth_map.max()

    if d_max - d_min < 1e-6:
        # Flat depth — return uniform 0.5
        return np.full_like(depth_map, 0.5)

    normalized = (depth_map - d_min) / (d_max - d_min)
    return normalized.astype(np.float32)


def get_depth_scale_map(
    depth_map: np.ndarray,
    floor_mask: np.ndarray,
    scale_range: tuple = (0.4, 1.0)
) -> np.ndarray:
    """
    Compute a per-pixel scale factor from depth, for tile size modulation.

    Tiles closer to the camera should be larger, tiles farther should be smaller.
    This creates a smooth scale map that's used during tile rendering.

    The depth gradient is key to perspective-correct tiling:
    - MiDaS gives inverse depth (closer = larger value)
    - We map this to a scale factor in [scale_range[0], scale_range[1]]
    - Closer pixels → scale ≈ 1.0 (full tile size)
    - Farther pixels → scale ≈ 0.4 (smaller tiles)

    Args:
        depth_map: Normalized depth map (H, W), float32, range [0, 1].
        floor_mask: Binary mask (H, W), uint8, 0 or 255.
        scale_range: (min_scale, max_scale) for tile sizing.

    Returns:
        Scale map (H, W), float32, values in scale_range.
    """
    # Extract depth only within floor region
    floor_depth = depth_map.copy()
    floor_pixels = floor_mask > 0

    if not np.any(floor_pixels):
        return np.ones_like(depth_map) * scale_range[1]

    # Normalize floor depth to 0-1 within the floor region
    floor_vals = floor_depth[floor_pixels]
    f_min, f_max = floor_vals.min(), floor_vals.max()

    if f_max - f_min < 1e-6:
        return np.ones_like(depth_map) * scale_range[1]

    # Map depth to scale: closer (high depth) → larger scale
    scale_map = np.zeros_like(depth_map)
    normalized_floor = (floor_depth - f_min) / (f_max - f_min)

    # Linear mapping: depth → scale
    min_scale, max_scale = scale_range
    scale_map = normalized_floor * (max_scale - min_scale) + min_scale

    # Apply floor mask
    scale_map[~floor_pixels] = 0

    return scale_map.astype(np.float32)


def visualize_depth(depth_map: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """
    Create a colorized visualization of the depth map for debugging.

    Args:
        depth_map: Depth map (H, W), float32.
        colormap: OpenCV colormap to use.

    Returns:
        Colorized depth visualization (H, W, 3), uint8, BGR.
    """
    normalized = normalize_depth(depth_map)
    depth_uint8 = (normalized * 255).astype(np.uint8)
    colorized = cv2.applyColorMap(depth_uint8, colormap)
    return colorized
