"""
Floor Segmentation Module
-------------------------
Detects floor regions in room images using a multi-strategy approach:

Strategy 1 (PRIMARY): Click-based flood fill in LAB color space.
    The user clicks on the floor, and we flood-fill in color space
    to find connected pixels with similar color/texture. This is the
    most reliable approach because DeepLabV3 PASCAL VOC classes don't
    have a specific "floor" label.

Strategy 2: Depth-assisted segmentation.
    Uses the depth map to identify horizontal surfaces (floor has a
    smooth depth gradient, walls have different gradients).

Strategy 3: DeepLabV3 semantic segmentation as a refinement filter
    to exclude object regions (furniture, people) from the floor mask.

Post-processing:
    - Morphological close/open to fill gaps and remove noise
    - Connected component analysis to keep largest region
    - Gaussian blur on edges for smooth boundary
"""

import numpy as np
import cv2
import torch
from torchvision import models, transforms
import logging

logger = logging.getLogger(__name__)

# ========================= MODEL LOADING =========================

_segmentation_model = None
_seg_device = None


def load_segmentation_model(device: str = "auto"):
    """
    Load DeepLabV3-ResNet101 pretrained model.
    Used as an auxiliary filter to exclude non-floor objects.
    """
    global _segmentation_model, _seg_device

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _segmentation_model is not None and _seg_device == device:
        return _segmentation_model, _seg_device

    logger.info(f"Loading DeepLabV3-ResNet101 on {device}...")
    _segmentation_model = models.segmentation.deeplabv3_resnet101(
        weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
    )
    _segmentation_model.eval().to(device)
    _seg_device = device
    logger.info("DeepLabV3 model loaded successfully.")
    return _segmentation_model, _seg_device


_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ========================= OBJECT MASK (what is NOT floor) =========================

def get_object_mask(image: np.ndarray, model=None, device: str = "auto") -> np.ndarray:
    """
    Get a binary mask of detected objects (NOT floor/background).
    
    DeepLabV3 with PASCAL VOC detects 20 object classes (person, chair, 
    sofa, table, etc.) plus background (class 0). We use this to EXCLUDE 
    objects from the floor mask, not to detect the floor itself.

    Returns:
        Binary mask (H, W), uint8, 255 = object pixel, 0 = non-object.
    """
    if model is None:
        model, device = load_segmentation_model(device)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = _preprocess(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)["out"]

    seg_map = output.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # Class 0 = background (includes floor + walls + ceiling)
    # Classes 1-20 = objects (person, bicycle, car, chair, sofa, etc.)
    # We want a mask of objects to EXCLUDE from floor
    object_mask = (seg_map > 0).astype(np.uint8) * 255

    # Resize to match original image
    if object_mask.shape != image.shape[:2]:
        object_mask = cv2.resize(object_mask, (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

    # Dilate slightly to ensure objects are fully covered
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    object_mask = cv2.dilate(object_mask, kernel, iterations=1)

    return object_mask


# ========================= FLOOR DETECTION STRATEGIES =========================

def segment_floor(image: np.ndarray, model=None, device: str = "auto") -> np.ndarray:
    """
    Automatic floor detection without a click point.
    
    Uses spatial + color heuristics since DeepLabV3 can't distinguish
    floor from walls (both are "background" class 0).

    Strategy:
    1. Get object mask to exclude furniture/people
    2. Focus on the bottom portion of the image
    3. Sample color from bottom-center and flood fill
    4. Clean and return

    Args:
        image: BGR numpy array (H, W, 3).
        model: Pre-loaded DeepLabV3 model.
        device: Device string.

    Returns:
        Binary floor mask (H, W), uint8, 0 or 255.
    """
    h, w = image.shape[:2]

    # Step 1: Get object exclusion mask
    try:
        object_mask = get_object_mask(image, model, device)
    except Exception as e:
        logger.warning(f"Object detection failed, using spatial fallback: {e}")
        object_mask = np.zeros((h, w), dtype=np.uint8)

    # Step 2: Use color-based floor detection from bottom-center
    # Sample a point at bottom-center (usually floor)
    sample_y = int(h * 0.85)
    sample_x = int(w * 0.5)

    # Make sure sample point isn't on an object
    if object_mask[sample_y, sample_x] > 0:
        # Try other bottom positions
        for sx in [w // 4, w * 3 // 4, w // 3, w * 2 // 3]:
            if object_mask[sample_y, int(sx)] == 0:
                sample_x = int(sx)
                break

    # Flood fill from sample point
    floor_mask = _color_flood_fill(image, (sample_x, sample_y), tolerance=30)

    # Step 3: Remove objects from floor mask
    floor_mask = cv2.bitwise_and(floor_mask, cv2.bitwise_not(object_mask))

    # Step 4: Apply spatial constraint — floor is mostly in bottom 70%
    spatial_weight = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        # Weight increases toward bottom of image
        ratio = y / h
        if ratio > 0.2:
            spatial_weight[y, :] = min(1.0, (ratio - 0.2) / 0.3)

    floor_float = floor_mask.astype(np.float32) / 255.0
    floor_float = floor_float * spatial_weight
    floor_mask = (floor_float > 0.5).astype(np.uint8) * 255

    return clean_mask(floor_mask)


def refine_mask_with_click(
    mask: np.ndarray,
    image: np.ndarray,
    click_point: tuple,
    tolerance: int = 30
) -> np.ndarray:
    """
    Primary floor detection strategy: flood-fill from user click.

    This is the most reliable approach. The user clicks on the floor,
    and we find all connected pixels with similar color in LAB space.

    LAB color space is used because:
    - L channel captures lightness independently from color
    - a,b channels capture color in a perceptually uniform way
    - Floor regions typically have consistent a,b values even with
      varying lighting (L changes but a,b stay similar)

    Args:
        mask: Initial mask (may be empty or from auto-detection).
        image: Original BGR image (H, W, 3).
        click_point: (x, y) where user clicked on the floor.
        tolerance: Color similarity tolerance (0-255).

    Returns:
        Binary floor mask (H, W), uint8, 0 or 255.
    """
    h, w = image.shape[:2]
    cx, cy = int(click_point[0]), int(click_point[1])
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))

    logger.info(f"Click-based floor detection at ({cx}, {cy}), tolerance={tolerance}")

    # Strategy A: Color-based flood fill in LAB space
    lab_mask = _color_flood_fill(image, (cx, cy), tolerance=tolerance)

    # Strategy B: Also try with slightly different tolerance for robustness
    lab_mask2 = _color_flood_fill(image, (cx, cy), tolerance=int(tolerance * 1.2))

    # Merge — use the one with more reasonable coverage
    coverage_a = np.sum(lab_mask > 0) / (h * w)
    coverage_b = np.sum(lab_mask2 > 0) / (h * w)

    # Ideal floor coverage is 15-60% of image
    if 0.10 < coverage_a < 0.65:
        combined = lab_mask
    elif 0.10 < coverage_b < 0.65:
        combined = lab_mask2
    else:
        # Use the more conservative one
        combined = lab_mask if coverage_a < coverage_b else lab_mask2

    logger.info(f"Flood fill coverage: {np.sum(combined > 0) / (h * w) * 100:.1f}%")

    return clean_mask(combined)


def _color_flood_fill(
    image: np.ndarray,
    seed_point: tuple,
    tolerance: int = 30
) -> np.ndarray:
    """
    Flood fill in LAB color space from a seed point.

    LAB space gives better results than RGB/HSV for floor detection
    because lighting variations primarily affect the L channel, while
    the floor's actual color is captured in a,b channels.

    Args:
        image: BGR image.
        seed_point: (x, y) seed pixel.
        tolerance: Color difference threshold.

    Returns:
        Binary mask (H, W), uint8, 0 or 255.
    """
    h, w = image.shape[:2]

    # Convert to LAB for perceptually uniform color comparison
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Blur slightly to reduce texture noise (floor textures can cause fragmentation)
    lab_blurred = cv2.GaussianBlur(lab, (11, 11), 3)

    # Flood fill with color tolerance
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Use different tolerances for L (lightness) vs a,b (color)
    # More tolerant on L (lighting varies), stricter on a,b (color should match)
    lo_diff = (tolerance * 1.5, tolerance * 0.8, tolerance * 0.8)
    hi_diff = (tolerance * 1.5, tolerance * 0.8, tolerance * 0.8)

    lo_diff = tuple(int(x) for x in lo_diff)
    hi_diff = tuple(int(x) for x in hi_diff)

    cv2.floodFill(
        lab_blurred.copy(),
        ff_mask,
        seed_point,
        newVal=(255, 255, 255),
        loDiff=lo_diff,
        upDiff=hi_diff,
        flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
    )

    # Extract mask (remove 1-pixel border added by floodFill)
    result = ff_mask[1:-1, 1:-1] * 255

    return result


def segment_floor_with_depth(
    image: np.ndarray,
    depth_map: np.ndarray,
    click_point: tuple = None,
    model=None,
    device: str = "auto"
) -> np.ndarray:
    """
    Enhanced floor segmentation using depth information.

    Depth-based floor detection:
    - Floor has a smooth, monotonically changing depth gradient
      (depth increases from bottom to top of floor)
    - Walls have a different depth gradient direction
    - Use depth gradient direction to separate floor from walls

    Args:
        image: BGR image.
        depth_map: Normalized depth map (H, W), float32, [0, 1].
        click_point: Optional (x, y) click for refinement.
        model: Pre-loaded segmentation model.
        device: Device string.

    Returns:
        Binary floor mask (H, W), uint8, 0 or 255.
    """
    h, w = image.shape[:2]

    # Get initial floor mask
    if click_point is not None:
        floor_mask = refine_mask_with_click(np.zeros((h, w), np.uint8), image, click_point)
    else:
        floor_mask = segment_floor(image, model, device)

    # Use depth to refine: floor pixels have a consistent depth relationship
    # (depth decreases smoothly from camera/bottom to back/top of floor)
    if depth_map is not None:
        # Compute depth gradient (vertical direction)
        depth_gy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)

        # Floor typically has a strong negative vertical gradient
        # (depth decreases as we go up in the image = farther from camera)
        # Walls have near-zero vertical gradient or positive gradient

        # Get the gradient characteristics at the click point or floor center
        floor_pixels = floor_mask > 0
        if np.any(floor_pixels):
            floor_grad = depth_gy[floor_pixels]
            median_grad = np.median(floor_grad)

            # If floor has a consistent gradient, use it to refine
            if abs(median_grad) > 0.001:
                # Pixels with similar gradient direction are likely floor
                grad_similar = np.abs(depth_gy - median_grad) < abs(median_grad) * 2
                depth_floor = grad_similar.astype(np.uint8) * 255

                # Intersect with color-based mask for safety
                floor_mask = cv2.bitwise_and(floor_mask, depth_floor)

    # Remove detected objects
    try:
        object_mask = get_object_mask(image, model, device)
        floor_mask = cv2.bitwise_and(floor_mask, cv2.bitwise_not(object_mask))
    except Exception:
        pass

    return clean_mask(floor_mask)


def clean_mask(mask: np.ndarray, min_area_ratio: float = 0.005) -> np.ndarray:
    """
    Post-process the floor mask: remove noise, smooth edges, fill holes.

    Operations:
    1. Morphological close (fill small gaps in the floor surface)
    2. Morphological open (remove isolated noise pixels)
    3. Keep only the largest connected component (the main floor)
    4. Fill interior holes (e.g., from furniture shadows)
    5. Gaussian blur edges for smooth boundary

    Args:
        mask: Binary mask (H, W), uint8, 0 or 255.
        min_area_ratio: Min area fraction to keep a component.

    Returns:
        Cleaned binary mask (H, W), uint8, 0 or 255.
    """
    if mask is None or np.sum(mask > 0) == 0:
        return mask if mask is not None else np.zeros((100, 100), dtype=np.uint8)

    h, w = mask.shape[:2]

    # Step 1: Morphological close — fill small gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    # Step 2: Morphological open — remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # Step 3: Keep only the LARGEST connected component
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    # Find the largest non-background component
    largest_label = 1
    largest_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i

    # Keep only the largest component
    cleaned = np.zeros_like(mask)
    cleaned[labels == largest_label] = 255

    # Step 4: Fill interior holes using contour filling
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(cleaned, contours, -1, 255, thickness=cv2.FILLED)

    # Step 5: Smooth edges with Gaussian blur + threshold
    blurred = cv2.GaussianBlur(cleaned.astype(np.float32), (7, 7), 3)
    cleaned = (blurred > 127).astype(np.uint8) * 255

    logger.info(f"Floor mask: {np.sum(cleaned > 0) / (h * w) * 100:.1f}% coverage, "
                f"largest component area: {largest_area}")

    return cleaned
