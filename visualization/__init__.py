"""
Visualization Module
--------------------
Depth-aware virtual tile visualization engine.

Sub-modules:
    - floor_segmentation: Semantic floor detection using DeepLabV3
    - depth_estimation: MiDaS-based per-pixel depth estimation
    - plane_geometry: RANSAC plane fitting + homography estimation
    - tile_renderer: Tile grid generation with pattern support
    - lighting_blender: Brightness matching + edge feathering
    - visualization_engine: Orchestrator combining all steps
    - realism: Post-composite realism enhancements
    - multi_tile_engine: Multi-tile comparison with geometry reuse
"""

from .visualization_engine import apply_tile_to_room, apply_tile_to_room_with_mask
from .multi_tile_engine import (
    GeometryCache,
    build_geometry_cache,
    render_multiple_tiles,
    render_single_tile_from_cache,
)
from .realism import enhance_realism, add_tile_variation

__all__ = [
    "apply_tile_to_room",
    "apply_tile_to_room_with_mask",
    "GeometryCache",
    "build_geometry_cache",
    "render_multiple_tiles",
    "render_single_tile_from_cache",
    "enhance_realism",
    "add_tile_variation",
]

