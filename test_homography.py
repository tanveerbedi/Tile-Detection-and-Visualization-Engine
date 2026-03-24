import numpy as np
import cv2

def test_homography():
    print("Testing Homography Math")
    h, w = 500, 500
    fx = fy = w * 0.85
    K = np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=np.float64)

    # Fake floor plane: Normal pointing up (-Y), distance = 1.5m below camera
    n = np.array([0, -1, 0], dtype=np.float64)
    d = 1.5

    # Fake mask bounding box: bottom half of image
    mask_xs = np.array([100, 400, 400, 100])
    mask_ys = np.array([300, 300, 450, 450])
    
    # Floor spatial basis
    Z_cam = np.array([0, 0, 1], dtype=np.float64)
    v_forward = Z_cam - np.dot(Z_cam, n) * n
    if np.linalg.norm(v_forward) < 1e-4:
        v_forward = np.array([0, -1, 0], dtype=np.float64) - np.dot(np.array([0, -1, 0]), n) * n
    v_forward = v_forward / np.linalg.norm(v_forward)
    u_right = np.cross(n, v_forward)
    u_right = u_right / np.linalg.norm(u_right)
    
    print(f"Basis: n={n}, v={v_forward}, u={u_right}")

    # Anchor point: center of mask
    cx = np.mean(mask_xs)
    cy = np.mean(mask_ys)
    ray_c = np.array([(cx - K[0,2])/K[0,0], (cy - K[1,2])/K[1,1], 1.0])
    t_c = -d / (np.dot(n, ray_c) + 1e-6)
    T_anchor = t_c * ray_c
    print(f"T_anchor: {T_anchor}")

    # Raycast corners
    rays = np.zeros((4, 3))
    rays[:, 0] = (mask_xs - K[0,2]) / K[0,0]
    rays[:, 1] = (mask_ys - K[1,2]) / K[1,1]
    rays[:, 2] = 1.0
    
    t_vals = -d / (np.dot(rays, n) + 1e-6)
    pts_3d = rays * t_vals[:, np.newaxis]
    print(f"3D corners Z values: {pts_3d[:, 2]}")

    # Project to 2D
    diff = pts_3d - T_anchor
    X_f = np.dot(diff, u_right)
    Y_f = np.dot(diff, v_forward)
    print(f"X_f: {X_f}")
    print(f"Y_f: {Y_f}")

    pixels_per_meter = 100.0
    pad = 0.5
    min_X_f, max_X_f = np.min(X_f) - pad, np.max(X_f) + pad
    min_Y_f, max_Y_f = np.min(Y_f) - pad, np.max(Y_f) + pad
    
    metric_w = max_X_f - min_X_f
    metric_h = max_Y_f - min_Y_f
    
    cw = int(np.ceil(metric_w * pixels_per_meter))
    ch = int(np.ceil(metric_h * pixels_per_meter))
    print(f"Canvas size: {cw}x{ch}")

    # Canvas mapping check
    src_pts = np.array([
        [0, 0],
        [cw - 1, 0],
        [cw - 1, ch - 1],
        [0, ch - 1],
    ], dtype=np.float32)

    dst_pts = []
    # VERY IMPORTANT: Canvas Y = 0 is top of image, but Y_f = 0 is front. Let's map Y ascending.
    # If Y_f goes forward (away from camera), it maps to bottom of canvas or top?
    # Actually, we want top of canvas to be far away, bottom of canvas to be close.
    # Y_f is max when far away, min when close.
    # If cy = 0 (top of canvas), we want Y_f to be far away (max_Y_f).
    # If cy = ch-1 (bottom of canvas), we want Y_f to be close (min_Y_f).
    # So Y_f = max_Y_f - (cy / pixels_per_meter)
    for (cx_img, cy_img) in src_pts:
        xf = cx_img / pixels_per_meter + min_X_f
        yf = max_Y_f - (cy_img / pixels_per_meter)
        
        P = T_anchor + xf * u_right + yf * v_forward
        u = P[0] * K[0,0] / P[2] + K[0,2]
        v = P[1] * K[1,1] / P[2] + K[1,2]
        dst_pts.append([u, v])

    dst_pts = np.array(dst_pts, dtype=np.float32)
    print("Dst Pts on Image:")
    print(dst_pts)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    print("Homography:\n", H)

test_homography()