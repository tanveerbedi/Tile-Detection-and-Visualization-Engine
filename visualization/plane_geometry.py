"""
Plane Geometry Estimation Module
--------------------------------
Estimates the floor plane and computes a homography matrix to warp
a tile grid into the correct perspective.

Key Concepts:
    - **Floor Quadrilateral**: We find the bounding quadrilateral of the 
      floor mask. The shape of this quad encodes the perspective.
    - **Depth-Weighted Corners**: Depth values at the quad corners tell us
      which edges are closer/farther, determining tile scale.
    - **Homography**: Maps a rectangle (bird's-eye tile grid) to the
      floor quadrilateral (perspective view). This makes tiles:
        * Reduce in size as depth increases
        * Converge toward vanishing line
        * Follow natural perspective foreshortening

Why not assume flat front view?
    Real room photos have perspective. The floor recedes into the 
    distance, so we must warp the tile grid to match.
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


def estimate_camera_intrinsics(image_shape: tuple) -> np.ndarray:
    """Estimate approximate camera intrinsic matrix K."""
    h, w = image_shape[:2]
    fx = fy = w * 0.85
    K = np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=np.float64)
    return K


def sample_floor_3d_points(
    depth_map: np.ndarray,
    floor_mask: np.ndarray,
    K_intrinsic: np.ndarray,
    max_points: int = 5000
) -> np.ndarray:
    """Back-project floor pixels into 3D using depth + intrinsics."""
    ys, xs = np.where(floor_mask > 0)
    if len(ys) == 0:
        return np.zeros((0, 3))

    if len(ys) > max_points:
        idx = np.random.choice(len(ys), max_points, replace=False)
        ys, xs = ys[idx], xs[idx]

    depths = depth_map[ys, xs]
    valid = depths > 0.01
    xs, ys, depths = xs[valid], ys[valid], depths[valid]
    if len(xs) == 0:
        return np.zeros((0, 3))

    fx, fy = K_intrinsic[0, 0], K_intrinsic[1, 1]
    cx, cy = K_intrinsic[0, 2], K_intrinsic[1, 2]

    Z = 1.0 / (depths + 1e-6)
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    return np.stack([X, Y, Z], axis=1)


def fit_plane_ransac(
    points_3d: np.ndarray,
    n_iterations: int = 1000,
    distance_threshold: float = 0.02
) -> tuple:
    """Fit a plane to 3D points using RANSAC."""
    n = len(points_3d)
    if n < 3:
        return np.array([0, 1, 0], dtype=np.float64), 0.0, np.zeros(n, dtype=bool)

    best_normal, best_d, best_count = None, 0, 0
    best_mask = np.zeros(n, dtype=bool)

    for _ in range(n_iterations):
        idx = np.random.choice(n, 3, replace=False)
        p1, p2, p3 = points_3d[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        mag = np.linalg.norm(normal)
        if mag < 1e-10:
            continue
        normal /= mag
        d = -np.dot(normal, p1)
        distances = np.abs(np.dot(points_3d, normal) + d)
        inliers = distances < distance_threshold
        count = np.sum(inliers)
        if count > best_count:
            best_count = count
            best_normal, best_d, best_mask = normal, d, inliers

    if best_normal is None:
        best_normal = np.array([0, 1, 0], dtype=np.float64)

    logger.info(f"RANSAC: {best_count}/{n} inliers ({100*best_count/max(n,1):.1f}%)")
    return best_normal, best_d, best_mask


def estimate_homography(
    floor_mask: np.ndarray,
    depth_map: np.ndarray,
    pixels_per_meter: float = 400.0
) -> tuple:
    """
    Compute the true 3D perspective homography from a bird's-eye floor grid
    onto the image using the mathematical plane fit of the floor depth.

    Steps:
    1. Sample 3D points from depth map & fit floor plane via RANSAC.
    2. Raycast the floor mask boundary onto the 3D plane to find actual physical bounds.
    3. Construct a virtual 2D tile canvas large enough to cover these bounds.
    4. Compute homography mapping the virtual canvas to image pixels.

    Returns:
        Tuple of (homography_3x3, canvas_size):
            - H: 3x3 float64 matrix (from canvas to image)
            - canvas_size: (width, height) in pixels
    """
    h, w = floor_mask.shape[:2]
    K = estimate_camera_intrinsics((h, w))

    # 1. Back-project depth map pixels to 3D and fit floor plane
    logger.info("Sampling 3D points for floor plane fitting...")
    pts_3d = sample_floor_3d_points(depth_map, floor_mask, K)
    
    # 2. Fit plane equation: n_x*X + n_y*Y + n_z*Z + d = 0
    normal, d, _ = fit_plane_ransac(pts_3d)

    # Ensure normal points towards the camera (origin is 0,0,0)
    # A plane facing the camera will have d > 0 in this equation form.
    if d < 0:
        normal = -normal
        d = -d

    logger.info(f"Surface plane: normal={normal}, distance={d}")

    # 3. Create 2D coordinate system on the plane
    n = normal
    # Project camera Z axis (0, 0, 1) to the plane to get "forward/depth" direction
    Z_cam = np.array([0., 0., 1.], dtype=np.float64)
    v_forward = Z_cam - np.dot(Z_cam, n) * n
    
    # If the plane is perfectly facing the camera (a straight wall), projection fails. 
    # Fallback to the camera's Up vector (-Y).
    if np.linalg.norm(v_forward) < 1e-2:
        Up_cam = np.array([0., -1., 0.], dtype=np.float64)
        v_forward = Up_cam - np.dot(Up_cam, n) * n
        
    v_forward = v_forward / np.linalg.norm(v_forward)
    
    # "Right" direction
    u_right = np.cross(v_forward, n)
    u_right = u_right / np.linalg.norm(u_right)

    # Define the origin of our tile grid on the floor.
    # We raycast the center of the brush mask to keep tiles anchored to the brushed area.
    mask_ys, mask_xs = np.where(floor_mask > 0)
    if len(mask_xs) == 0:
        logger.error("Empty floor mask. Returning identity homography.")
        return np.eye(3, dtype=np.float64), (w, h)

    cx = (np.min(mask_xs) + np.max(mask_xs)) / 2.0
    cy = (np.min(mask_ys) + np.max(mask_ys)) / 2.0
    
    ray_c = np.array([(cx - K[0, 2]) / K[0, 0], (cy - K[1, 2]) / K[1, 1], 1.0], dtype=np.float64)
    t_c = -d / (np.dot(n, ray_c) + 1e-6)
    if t_c < 0:
        t_c = 1.0 # fallback
    T_anchor = t_c * ray_c

    # 4. Raycast mask contour to measure physical bounding box of the floor mask
    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.eye(3, dtype=np.float64), (w, h)
        
    largest = max(contours, key=cv2.contourArea)
    pts = largest.reshape(-1, 2)
    
    # Convert image pixels to unit rays
    rays = np.zeros((len(pts), 3), dtype=np.float64)
    rays[:, 0] = (pts[:, 0] - K[0, 2]) / K[0, 0]
    rays[:, 1] = (pts[:, 1] - K[1, 2]) / K[1, 1]
    rays[:, 2] = 1.0
    
    # Intersect with floor plane
    t_vals = -d / (np.dot(rays, n) + 1e-6)
    
    # Keep only points that are actually in front of the camera (t > 0) and not infinitely far (t < 50)
    valid = (t_vals > 0) & (t_vals < 50.0)
    if not np.any(valid):
        return np.eye(3, dtype=np.float64), (w, h)
        
    pts_3d_contour = rays[valid] * t_vals[valid][:, np.newaxis]
    
    # Project the physical 3D points onto the 2D floor basis (X_f, Y_f) relative to anchor
    diff = pts_3d_contour - T_anchor
    X_f = np.dot(diff, u_right)    # Real-world 'left/right' meters
    Y_f = np.dot(diff, v_forward)  # Real-world 'forward/back' meters
    
    min_X_f, max_X_f = np.min(X_f), np.max(X_f)
    min_Y_f, max_Y_f = np.min(Y_f), np.max(Y_f)
    
    # Add a generous half-meter padding so the canvas completely envelopes the mask
    pad = 0.5
    min_X_f -= pad
    max_X_f += pad
    min_Y_f -= pad
    max_Y_f += pad

    # Calculate required canvas pixel dimensions
    metric_w = max_X_f - min_X_f
    metric_h = max_Y_f - min_Y_f
    
    # Sanity bounds: if user painted up the wall, rays shoot to infinity.
    # We restrict the metric floor to a maximum of 15x15 meters around the closest point.
    if metric_w > 15.0:
        center_x = (min_X_f + max_X_f) / 2.0
        min_X_f = center_x - 7.5
        max_X_f = center_x + 7.5
        metric_w = 15.0
        
    if metric_h > 15.0:
        # Always anchor to the closest part of the floor (min_Y_f) and cut off the infinite horizon
        max_Y_f = min_Y_f + 15.0
        metric_h = 15.0
    
    canvas_w = int(np.ceil(metric_w * pixels_per_meter))
    canvas_h = int(np.ceil(metric_h * pixels_per_meter))
    
    canvas_w = max(100, canvas_w)
    canvas_h = max(100, canvas_h)
    
    # 5. Compute Homography mapping the Canvas corners to Image coordinates
    src_pts = np.array([
        [0, 0],
        [canvas_w - 1, 0],
        [canvas_w - 1, canvas_h - 1],
        [0, canvas_h - 1],
    ], dtype=np.float32)
    
    dst_pts = []
    for (cx_img, cy_img) in src_pts:
        # Convert canvas pixel (cx, cy) to floor physics meters (xf, yf)
        xf = cx_img / pixels_per_meter + min_X_f
        # Y is ascending down the canvas. Top of canvas (cy=0) should be FAR away (max_Y_f)
        yf = max_Y_f - (cy_img / pixels_per_meter)
        
        # 3D coordinate of the corner
        P = T_anchor + xf * u_right + yf * v_forward
        
        # Project back to image pixel
        u = P[0] * K[0, 0] / P[2] + K[0, 2]
        v = P[1] * K[1, 1] / P[2] + K[1, 2]
        dst_pts.append([u, v])
        
    dst_pts = np.array(dst_pts, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    if H is None:
        logger.error("Homography mathematically degenerate. Returning identity.")
        return np.eye(3, dtype=np.float64), (canvas_w, canvas_h)

    logger.info(f"Perfect 3D Homography computed. Metric canvas: {canvas_w}x{canvas_h}")
    return H.astype(np.float64), (canvas_w, canvas_h)


def estimate_wall_homography(
    wall_mask: np.ndarray,
    pixels_per_meter: float = 400.0
) -> tuple:
    """
    Computes a simple 2D homography for flat walls without 3D depth perspective.

    The canvas is sized exactly to the bounding box of the brush mask, 
    assuming the wall is parallel to the camera. This avoids the 3D raycasting
    which can fail on extreme perspectives.

    Returns:
        Tuple of (homography_3x3, canvas_size)
    """
    h, w = wall_mask.shape[:2]
    ys, xs = np.where(wall_mask > 0)
    
    if len(xs) == 0:
        logger.error("Empty wall mask. Returning identity homography.")
        return np.eye(3, dtype=np.float64), (w, h)
        
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    
    bw = max_x - min_x
    bh = max_y - min_y
    
    # We map 1 pixel of the wall bounding box to 1 pixel of the canvas.
    # Physical size scaling will be handled by the tile generation picking up `pixels_per_meter`.
    canvas_w = max(100, bw)
    canvas_h = max(100, bh)
    
    src_pts = np.array([
        [0, 0],
        [canvas_w - 1, 0],
        [canvas_w - 1, canvas_h - 1],
        [0, canvas_h - 1],
    ], dtype=np.float32)
    
    # Map the canvas to the bounding box of the mask in the image
    dst_pts = np.array([
        [min_x, min_y],
        [min_x + canvas_w - 1, min_y],
        [min_x + canvas_w - 1, min_y + canvas_h - 1],
        [min_x, min_y + canvas_h - 1],
    ], dtype=np.float32)
    
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    logger.info(f"Simple 2D Wall Homography computed. Canvas: {canvas_w}x{canvas_h}")
    return H.astype(np.float64), (canvas_w, canvas_h)



def _get_floor_quad_from_bbox(contour, img_h, img_w):
    """Fallback: create quad from bounding box of the floor contour."""
    x, y, bw, bh = cv2.boundingRect(contour)
    pts = np.array([
        [[x, y]],
        [[x + bw, y]],
        [[x + bw, y + bh]],
        [[x, y + bh]]
    ], dtype=np.int32)
    return pts


def _reduce_to_quad(approx_poly, img_h, img_w):
    """Reduce a polygon to 4 corners by keeping the most extreme points."""
    pts = approx_poly.reshape(-1, 2).astype(np.float32)

    # Use the convex hull and find 4 extreme points
    # Top-left: min x+y
    # Top-right: max x-y
    # Bottom-right: max x+y
    # Bottom-left: min x-y
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]

    tl = pts[np.argmin(sums)]
    br = pts[np.argmax(sums)]
    tr = pts[np.argmax(diffs)]
    bl = pts[np.argmin(diffs)]

    return np.array([[tl], [tr], [br], [bl]], dtype=np.int32)


def _order_points(pts):
    """
    Order 4 points as: top-left, top-right, bottom-right, bottom-left.

    Uses the sum and difference of coordinates:
    - Top-left has smallest sum (x+y)
    - Bottom-right has largest sum (x+y)
    - Top-right has largest difference (x-y)
    - Bottom-left has smallest difference (x-y)
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()

    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    rect[1] = pts[np.argmax(d)]   # top-right
    rect[3] = pts[np.argmin(d)]   # bottom-left

    return rect


def compute_vanishing_point(floor_mask, depth_map):
    """Estimate vanishing point from depth gradient in floor region."""
    h, w = floor_mask.shape[:2]
    gy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
    gx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
    fp = floor_mask > 0
    if not np.any(fp):
        return (w // 2, 0)
    mean_gx = np.mean(gx[fp])
    mean_gy = np.mean(gy[fp])
    floor_ys, floor_xs = np.where(fp)
    cy, cx = np.mean(floor_ys), np.mean(floor_xs)
    if abs(mean_gy) < 1e-6:
        return (w // 2, 0)
    scale = h / max(abs(mean_gy), 1e-6) * 0.5
    vx = int(cx - mean_gx * scale)
    vy = int(cy - mean_gy * scale)
    return (max(-w, min(2*w, vx)), max(-h, min(2*h, vy)))


def get_canvas_size(floor_mask, depth_map):
    """Determine optimal tile canvas size."""
    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return floor_mask.shape[1], floor_mask.shape[0]
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    return (max(int(bw * 2.0), 800), max(int(bh * 2.0), 800))
