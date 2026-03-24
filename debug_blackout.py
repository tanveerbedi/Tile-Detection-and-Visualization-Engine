import cv2
import numpy as np
from visualization.visualization_engine import apply_tile_to_room_with_mask

def debug():
    print("Loading images...")
    room = cv2.imread(r"C:\Users\sarda\.gemini\antigravity\brain\6da33880-7281-4eaf-920d-e19a9bd83c09\living_room_1772177911441.png")
    tile = cv2.imread(r"F:\Netsmartz\Tile_visual\crops\crop_0.jpg")
    
    if room is None or tile is None:
        print("Error loading images!")
        return

    h, w = room.shape[:2]
    
    # Create a dummy brush mask (top half of the image - the Wall)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[50:int(h*0.4), 100:w-100] = 255
    cv2.imwrite("debug_mask.jpg", mask)
    
    print("Running pipeline...")
    res = apply_tile_to_room_with_mask(room, tile, mask, surface_type="wall")
    
    print("Saving intermediate outputs...")
    print(f"Canvas size: {res['canvas_size']}")
    print(f"Homography: {res['homography']}")
    if res.get('depth_map') is not None:
        cv2.imwrite("debug_depth.jpg", (res['depth_map']*255).astype(np.uint8))
    cv2.imwrite("debug_warped.jpg", res['warped_tiles'])
    
    # Re-run lit_tiles manually to see where it goes black
    from visualization.lighting_blender import extract_brightness_map, apply_lighting, extract_color_statistics, color_match_tile, composite, feather_edges
    
    bmap = extract_brightness_map(room, mask)
    cv2.imwrite("debug_bmap.jpg", (bmap * 127).clip(0, 255).astype(np.uint8))
    
    lit = apply_lighting(res['warped_tiles'], bmap, mask)
    cv2.imwrite("debug_lit.jpg", lit)
    
    stats = extract_color_statistics(room, mask)
    colored = color_match_tile(lit, mask, stats, 0.25)
    cv2.imwrite("debug_colored.jpg", colored)
    
    out_log = ""
    out_log += f"Canvas size: {res['canvas_size']}\n"
    out_log += f"Homography:\n{res['homography']}\n"
    
    final = res['result']
    out_log += f"Warped Tiles: max={res['warped_tiles'].max()}, min={res['warped_tiles'].min()}, non_zero={np.count_nonzero(res['warped_tiles'])}\n"
    out_log += f"Lit Tiles: max={lit.max()}, min={lit.min()}, non_zero={np.count_nonzero(lit)}\n"
    out_log += f"Colored Tiles: max={colored.max()}, min={colored.min()}, non_zero={np.count_nonzero(colored)}\n"
    out_log += f"Final Result: max={final.max()}, min={final.min()}, non_zero={np.count_nonzero(final)}\n"
    
    # Check if warped_tiles has color in the masked region
    # mask is single channel 0/255
    masked_warped = res['warped_tiles'][mask > 0]
    out_log += f"Warped inside mask: shape={masked_warped.shape}, max={masked_warped.max()}, mean={masked_warped.mean()}\n"
    
    masked_colored = colored[mask > 0]
    out_log += f"Colored inside mask: max={masked_colored.max()}, mean={masked_colored.mean()}\n"

    with open("debug_log.txt", "w") as f:
        f.write(out_log)

    print("Done. Check debug_*.jpg and debug_log.txt")

debug()