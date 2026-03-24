# 🏠 Depth-Aware Virtual Tile Visualization Engine

A comprehensive computer vision application that lets users virtually **re-tile floors and walls** in room photographs with photorealistic results. The system combines deep-learning object detection, visual similarity search, monocular depth estimation, 3D perspective geometry, and a **multi-tile comparison mode** — all through an interactive Streamlit UI.

---

## Table of Contents

- [Why This Project?](#why-this-project)
- [Features](#features)
- [🆕 Multi-Tile Comparison Mode](#-multi-tile-comparison-mode)
- [How It Works — End-to-End Pipeline](#how-it-works--end-to-end-pipeline)
- [Architecture Deep Dive](#architecture-deep-dive)
  - [1. Tile Detection (YOLOv8)](#1-tile-detection-yolov8)
  - [2. Visual Recommendation (ResNet50)](#2-visual-recommendation-resnet50--cosine-similarity)
  - [3. Floor/Wall Selection (Brush Tool)](#3-floorwall-selection-brush-tool)
  - [4. Depth Estimation (MiDaS DPT-Large)](#4-depth-estimation-midas-dpt-large)
  - [5. Plane Geometry & Homography (RANSAC)](#5-plane-geometry--homography-ransac)
  - [6. Tile Rendering (Pattern Engine)](#6-tile-rendering-pattern-engine)
  - [7. Lighting & Blending (LAB Color Space)](#7-lighting--blending-lab-color-space)
  - [8. Multi-Tile Engine (Geometry Reuse)](#8-multi-tile-engine-geometry-reuse)
  - [9. AI Realistic Mode (Future — ControlNet)](#9-ai-realistic-mode-future--controlnet)
- [Project Structure](#project-structure)
- [Tech Stack & Dependencies](#tech-stack--dependencies)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [Configuration Options](#configuration-options)
- [Performance Notes](#performance-notes)
- [API Reference (Legacy FastAPI)](#api-reference-legacy-fastapi)
- [Troubleshooting](#troubleshooting)

---

## Why This Project?

Traditional virtual staging tools require 3D room models, CAD software, or expensive commercial APIs. This project solves the problem with **a single photograph** — no 3D scanning, no room measurements, no manual polygon selection.

| Problem | Solution |
|---|---|
| Show clients tile options in their actual room | Upload room photo → brush floor → apply tile instantly |
| Flat tile overlays look fake | MiDaS depth + RANSAC plane fitting → perspective-correct warp |
| Manual region selection is tedious | Freedraw brush — just paint over the floor area |
| Hard to compare multiple tile options | **Comparison Mode** — apply 2–5 tiles at once, side-by-side |
| Repeating depth estimation for each tile wastes time | Geometry computed **once**, reused for all tiles |
| Tile catalog is hard to search visually | ResNet50 embeddings + cosine similarity finds top-K similar tiles |
| Overlay tiles look pasted on | LAB lighting transfer + Gaussian edge feathering blends seamlessly |

---

## Features

1. **Tile Detection (YOLOv8)** — Upload an image containing tiles. A pre-trained YOLO model detects and crops individual tile regions automatically.

2. **Visual Similarity Search (ResNet50)** — Select a tile to extract deep feature embeddings (2048-d). Cosine similarity finds the most visually similar tiles from a precomputed catalog.

3. **Interactive Floor/Wall Selection** — Upload a room photograph and use the built-in brush tool to paint over the target surface. No complex polygon tools.

4. **Depth-Aware 3D Perspective (MiDaS + RANSAC)** — Estimates a real-time monocular depth map, fits a 3D floor plane via RANSAC, and computes a homography matrix for perspective-correct tile mapping.

5. **Wall Mode** — For vertical surfaces, uses a simplified 2D affine homography (walls are parallel to camera — no depth needed).

6. **Dynamic Tile Layouts** — Three configurable tile patterns:
   - **Straight** — Standard rectangular grid
   - **Diagonal** — Diamond pattern (tiles rotated 45°)
   - **Brick** — Running bond (alternating row offsets)

7. **Realistic Lighting Compositing** — Extracts brightness/shadow maps from the original floor in LAB color space, transfers them onto the tile texture, and blends with Gaussian edge feathering.

8. **🆕 Multi-Tile Comparison Mode** — Select 2–5 tiles and see all of them applied to the same room side-by-side. Geometry is computed only once and shared across all tiles for efficiency.

9. **Before/After Comparison Slider** — Interactive drag slider to compare original room with the tiled result.

10. **Downloadable Results** — Export each result as PNG. In Comparison Mode every tile gets its own download button.

---

## 🆕 Multi-Tile Comparison Mode

> Added in the latest upgrade. Allows users to compare 2–5 tile options applied to the **identical room geometry** so comparisons are truly fair.

### How to Use

1. Click the **🔲 Comparison Mode (2–5 tiles)** radio at the top of the page
2. Add tiles using any of the tile sources:
   - Upload tile → click **Add to Compare**
   - YOLOv8 detection → checkboxes appear on each crop card
   - Similar Tiles catalog → **"+ Add to Compare"** button per result
3. Upload your room image and brush the floor/wall area
4. Click **🎯 Apply Selected Tiles (N)**
5. Review the side-by-side comparison grid:
   - Each tile shows its rendered result, name, and similarity score
   - The **⭐ Best Match** badge highlights the highest-scoring tile
6. Click **✅ Use Best Tile** to adopt the winner into Single Mode, or download any result individually

### Why It's Fast

The expensive pipeline steps run **exactly once**:

```
build_geometry_cache() — runs ONCE for all tiles
  ├─ MiDaS depth estimation         (~10–15 s saved per additional tile)
  ├─ RANSAC floor plane fitting      (shared)
  ├─ Homography computation          (identical matrix used by all tiles)
  └─ Lighting analysis               (brightness map, color stats, feathered mask)

render_single_tile_from_cache() — runs once PER tile (~1–2 s each)
  ├─ Create tile canvas with pattern
  ├─ Warp using cached homography
  ├─ Apply cached brightness map
  └─ Composite with cached feathered mask
```

For 3 tiles on CPU:
- **Old approach (no caching):** 3 × 16 s = ~48 s
- **New approach (shared geometry):** 16 s + 3 × 2 s = ~22 s

### Visual Consistency Guarantee

All tile renders use **identical** perspective, depth scaling, lighting, and tile scale baseline — making the visual comparison fair and trustworthy.

### Edge Cases

| Situation | Behavior |
|---|---|
| 0 tiles selected | Warning: "Please select at least one tile" |
| 1 tile selected | Falls back to single tile render automatically |
| > 5 tiles | Warning: "Maximum 5 tiles allowed — deselect some" |
| A tile render fails | Shows partial results; error tile marked in grid |

---

## How It Works — End-to-End Pipeline

### Single Tile Mode

```
┌──────────────┐     ┌──────────────┐     ┌───────────────┐
│  Upload Tile │────▶│ YOLOv8       │────▶│ Select Tile   │
│  Image       │     │ Detection    │     │ (or Direct)   │
└──────────────┘     └──────────────┘     └───────┬───────┘
                                                  │
                          ┌───────────────────────┘
                          ▼
                  ┌──────────────────┐
                  │ (Optional)       │
                  │ ResNet50 Catalog │
                  │ Similarity       │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐     ┌──────────────────┐
                  │ Upload Room      │────▶│ Brush Floor/Wall │
                  │ Image            │     │ (Canvas Tool)    │
                  └──────────────────┘     └────────┬─────────┘
                                                    │
                           ┌────────────────────────┘
                           ▼
              ┌────────────────────────┐
              │ apply_tile_to_room_    │
              │ with_mask()            │
              │                        │
              │  1. Depth (MiDaS)      │
              │  2. Homography (RANSAC)│
              │  3. Tile Canvas        │
              │  4. Perspective Warp   │
              │  5. Lighting (LAB)     │
              │  6. Composite          │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Before/After Slider   │
              │  + Download PNG        │
              └────────────────────────┘
```

### Comparison Mode

```
  Select 2–5 Tiles
         │
         ▼
  Upload Room + Brush Floor
         │
         ▼
  build_geometry_cache()          ← ONE TIME
  ┌──────────────────────┐
  │ MiDaS Depth          │
  │ RANSAC + Homography  │
  │ Brightness Map       │
  │ Feathered Mask       │
  └──────────┬───────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
  Tile 1           Tile 2  ...  Tile N
  render_          render_      render_
  single_tile_     single_tile_ single_tile_
  from_cache()     from_cache() from_cache()
    │                 │
    └────────┬────────┘
             ▼
  ┌────────────────────────┐
  │  Comparison Grid       │
  │  | T1 | T2 | T3 |      │
  │  ⭐ Best Match badge   │
  │  Per-tile Download     │
  │  Before/After Sliders  │
  └────────────────────────┘
```

---

## Architecture Deep Dive

### 1. Tile Detection (YOLOv8)

**Module:** `detection/detector.py`

Detects individual tiles in an uploaded image and crops them as separate images.

- Loads a custom-trained YOLOv8 model from `models/tile_yolov8_best.pt`
- Runs inference, filters class `"tiles"`, crops each detection by bounding box
- Saves crops as JPEG with UUID filenames to a temp directory

```python
model = YOLO("models/tile_yolov8_best.pt")
results = model(image_path)[0]
for box in results.boxes.xyxy:
    x1, y1, x2, y2 = map(int, box)
    crop = image[y1:y2, x1:x2]
```

---

### 2. Visual Recommendation (ResNet50 + Cosine Similarity)

**Modules:** `recommendation/embedding.py`, `recommendation/similarity.py`

**Embedding (`embedding.py`):**
- ResNet50 with final classification layer replaced by `Identity()` → 2048-d feature vector
- **Test-Time Augmentation (TTA):** generates 4 rotations (0°, 90°, 180°, 270°), averages embeddings → orientation-invariant features
- L2-normalized for cosine similarity

**Similarity search (`similarity.py`):**
- Loads `tile_embeddings.npy` (catalog) and `tile_labels.npy`
- Cosine similarity against all catalog embeddings → returns top-K with metadata from `df_balanced.csv`

```python
# TTA: 4-rotation averaging
features = model(torch.stack([transform(img.rotate(r)) for r in [0,90,180,270]]))
embedding = l2_normalize(features.mean(axis=0))

# Similarity
similarities = cosine_similarity([query_embedding], catalog_features)[0]
top_k = similarities.argsort()[::-1][:k]
```

---

### 3. Floor/Wall Selection (Brush Tool)

**Module:** `app.py` (uses `streamlit-drawable-canvas`)

- Displays the room image as a canvas background
- User paints with a configurable brush (size, opacity)
- Canvas RGBA alpha channel → binary mask (alpha > 10)
- Morphological close → open cleans gaps and noise
- Thresholded to strict binary (0 or 255)

In **Comparison Mode**, the same mask is reused for every tile — ensuring all tiles are applied to exactly the same region.

---

### 4. Depth Estimation (MiDaS DPT-Large)

**Module:** `visualization/depth_estimation.py`

Generates a per-pixel depth map from a single RGB room image.

- **MiDaS DPT-Large** via `torch.hub` — Vision Transformer backbone, highest accuracy
- Outputs *inverse depth* (closer = larger value)
- Key functions:
  - `load_midas_model(device)` — loads once, cached globally in `_midas_model`
  - `estimate_depth(image, ...)` → raw depth map
  - `normalize_depth(raw)` → [0, 1] float32
  - `visualize_depth(raw)` → INFERNO colormap for debug
  - `get_depth_scale_map(depth, mask)` → per-pixel tile scale factors

In Comparison Mode, `estimate_depth()` runs **once** inside `build_geometry_cache()` and the result is shared.

---

### 5. Plane Geometry & Homography (RANSAC)

**Module:** `visualization/plane_geometry.py`

Estimates the 3D floor plane from the depth map and computes a perspective homography to warp a flat tile grid into the correct camera view.

**Floor mode (`estimate_homography`):**
1. **Camera intrinsics:** Approximate focal length from image dimensions
2. **3D back-projection:** `X = (u−cx)·Z/fx`, `Y = (v−cy)·Z/fy` for all floor pixels
3. **RANSAC plane fit:** 1000 iterations, samples 3 points, fits plane `ax+by+cz+d=0`, keeps best inlier set
4. **Homography:** Projects tile canvas corners through the plane → `cv2.getPerspectiveTransform()`

**Wall mode (`estimate_wall_homography`):**
- Simplified 2D — maps canvas corners directly to bounding box of brush mask
- No depth needed — walls are parallel to camera

---

### 6. Tile Rendering (Pattern Engine)

**Module:** `visualization/tile_renderer.py`

Generates a large tile canvas in bird's-eye view, then warps it through the homography.

| Pattern | Method | Description |
|---|---|---|
| **Straight** | `_create_straight_canvas()` | Regular rectangular grid with grout lines |
| **Diagonal** | `_create_diagonal_canvas()` | Tiles rotated 45° in diamond layout |
| **Brick** | `_create_brick_canvas()` | Running bond — alternate rows offset by half tile width |

Each pattern draws configurable **grout lines** (width + color). The canvas is then warped with `cv2.warpPerspective()` using the homography matrix.

In Comparison Mode, only this step repeats per tile — it is fast (~0.5–1 s).

---

### 7. Lighting & Blending (LAB Color Space)

**Module:** `visualization/lighting_blender.py`

Makes the overlaid tiles look naturally lit and seamlessly blended.

1. **`extract_brightness_map()`** — Extracts floor luminance from LAB L-channel → ratio map centered at 1.0
2. **`apply_lighting()`** — Multiplies tile pixels by brightness ratio → tiles inherit room shadows/highlights
3. **`color_match_tile()`** — Partial LAB color transfer at configurable strength (default 0.25) → warm rooms get warm-looking tiles
4. **`feather_edges()`** — Gaussian-blurred eroded mask → smooth gradient at boundaries
5. **`composite()`** — `result = original × (1−alpha) + lit_tile × alpha`

In Comparison Mode, brightness map, color stats, and feathered mask are computed **once** in `build_geometry_cache()` and reused per tile.

---

### 8. Multi-Tile Engine (Geometry Reuse)

**Module:** `visualization/multi_tile_engine.py` *(new)*

This is the core of the Multi-Tile Comparison Mode upgrade.

#### `GeometryCache` dataclass

Holds all shared, tile-independent computed state:

```python
@dataclass
class GeometryCache:
    norm_depth: Optional[np.ndarray]      # Normalized depth map (H,W)
    depth_colored: Optional[np.ndarray]   # Debug colorized depth
    homography: np.ndarray                # 3×3 perspective matrix
    canvas_size: Tuple[int, int]          # (width, height) of tile canvas
    feathered_mask: np.ndarray            # Soft alpha mask (H,W) float32
    brightness_map: np.ndarray            # Per-pixel brightness multiplier
    floor_stats: dict                     # LAB color statistics
    surface_type: str                     # 'floor' or 'wall'
    pixels_per_meter: float               # Canvas resolution
    build_time_s: float                   # Build wall-clock time (seconds)
```

#### `build_geometry_cache(room_image, floor_mask, ...)`

Runs the expensive pipeline steps once:
1. MiDaS depth estimation (floor only)
2. Homography computation via RANSAC
3. Lighting analysis (brightness map + color stats)
4. Feathered edge mask

Returns a populated `GeometryCache`.

#### `render_single_tile_from_cache(room_image, tile_image, floor_mask, cache, ...)`

Applies a single tile using the pre-computed cache. Only runs:
- Tile canvas creation
- Perspective warp (from cache homography)
- Lighting application (from cache brightness map)
- Compositing (from cache feathered mask)

Returns the same result dict schema as `apply_tile_to_room_with_mask()`.

#### `render_multiple_tiles(room_image, tiles, floor_mask, ...)`

Orchestrates comparison rendering:
```python
# Tiles is a list of dicts: {image, name, score}
cache = build_geometry_cache(...)           # ONCE
results = [render_single_tile_from_cache(..., cache) for tile in tiles]
```

Supports an optional `progress_callback(current, total, message)` for UI progress bars.

---

### 9. AI Realistic Mode (Future — ControlNet)

**Module:** `visualization/ai_realistic_mode.py`

**Status:** 🔴 **NOT YET IMPLEMENTED** — Placeholder only.

Planned: MiDaS depth map as ControlNet conditioning + Stable Diffusion Inpainting for photorealistic material rendering (specular reflections, marble veins, grout shadows, global illumination). Will supplement (not replace) the deterministic CV pipeline.

---

## Project Structure

```
Tile_visual/
├── app.py                          # Streamlit UI (main entry point)
├── main.py                         # Legacy FastAPI API server
├── requirements.txt
├── README.md
│
├── detection/                      # Tile Detection Module
│   ├── __init__.py
│   └── detector.py                 # YOLOv8 inference + tile cropping
│
├── recommendation/                 # Visual Similarity Module
│   ├── __init__.py
│   ├── embedding.py                # ResNet50 feature extraction with TTA
│   └── similarity.py               # Cosine similarity search against catalog
│
├── visualization/                  # Core Rendering Engine
│   ├── __init__.py                 # Public API exports
│   ├── visualization_engine.py     # Single-tile pipeline orchestrator
│   ├── multi_tile_engine.py        # ★ NEW — Multi-tile comparison engine
│   ├── depth_estimation.py         # MiDaS DPT-Large depth mapping
│   ├── floor_segmentation.py       # DeepLabV3 floor detection (auto mode)
│   ├── plane_geometry.py           # RANSAC plane fitting + homography
│   ├── tile_renderer.py            # Tile canvas generation + perspective warp
│   ├── lighting_blender.py         # LAB color matching + edge feathering
│   └── ai_realistic_mode.py        # ControlNet placeholder (future)
│
├── utils/                          # Legacy utility wrappers
│   ├── detection.py
│   ├── embedding.py
│   └── similarity.py
│
├── models/                         # Pre-trained weights & catalog data
│   ├── tile_yolov8_best.pt         # Custom YOLOv8 tile detection weights
│   ├── tile_embeddings.npy         # Precomputed ResNet50 catalog embeddings
│   ├── tile_labels.npy             # Catalog image filenames
│   └── df_balanced.csv             # Catalog metadata (name, size, texture, color)
│
├── tests/
│   ├── test_visualization.py       # Unit tests for core pipeline (22 tests)
│   └── test_multi_tile.py          # ★ NEW — Unit tests for multi-tile engine (10 tests)
│
├── templates/                      # Legacy FastAPI HTML templates
├── static/                         # Static assets
├── crops/                          # YOLOv8 crop output directory
└── data/
```

---

## Tech Stack & Dependencies

| Component | Technology | Purpose |
|---|---|---|
| **UI** | Streamlit + streamlit-drawable-canvas | Interactive web UI with brush tool |
| **Tile Detection** | YOLOv8 (Ultralytics) | Object detection for tile extraction |
| **Feature Extraction** | ResNet50 (torchvision) | 2048-d embedding vectors for similarity |
| **Depth Estimation** | MiDaS DPT-Large (torch.hub) | Monocular per-pixel depth maps |
| **Floor Segmentation** | DeepLabV3-ResNet101 (torchvision) | Auxiliary segmentation for auto mode |
| **3D Geometry** | NumPy + OpenCV | RANSAC, homography, perspective warp |
| **Color Science** | OpenCV LAB conversion | Lighting transfer, color matching |
| **Similarity Search** | scikit-learn | Cosine similarity |
| **Multi-Tile Engine** | Pure NumPy / OpenCV | Geometry caching & comparison rendering |
| **Deep Learning** | PyTorch + torchvision + timm | Model infrastructure |
| **Legacy API** | FastAPI + Uvicorn | Headless detection/recommendation API |

---

## Installation

### Prerequisites

- Python 3.10+ (tested with Python 3.12.5)
- ~2 GB disk space for model downloads (MiDaS downloads ~500 MB on first run)
- GPU optional — CUDA-enabled GPU dramatically speeds up depth estimation

### Steps (Windows)

```powershell
# 1. Clone the repository
git clone <repository-url>
cd Tile_visual

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) GPU acceleration — PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Required Model Files in `models/`

| File | Purpose | Size |
|---|---|---|
| `tile_yolov8_best.pt` | YOLOv8 tile detection weights | ~6 MB |
| `tile_embeddings.npy` | Precomputed catalog embeddings | ~8.5 MB |
| `tile_labels.npy` | Catalog image filenames | ~180 KB |
| `df_balanced.csv` | Catalog metadata | ~150 KB |

> **Note:** MiDaS DPT-Large (~500 MB) and DeepLabV3-ResNet101 (~250 MB) download automatically via `torch.hub` / torchvision on first run.

---

## Running the Application

### Streamlit UI (Recommended)

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Run Tests

```powershell
# All tests — 32 total (22 existing + 10 new multi-tile)
.venv\Scripts\python.exe -m pytest tests/ -v

# Only multi-tile tests
.venv\Scripts\python.exe -m pytest tests/test_multi_tile.py -v
```

### Legacy FastAPI Server

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Usage Guide

### Single Tile Mode (Original)

1. **Upload a Tile Image** — choose *Direct texture* or *Detect tiles (YOLOv8)*
2. **Select Your Tile** — click *Use #N* on a crop, or browse *Find Similar Tiles*
3. **Upload a Room Image** — images > 1024px are auto-downscaled
4. **Brush the Target Surface** — paint over floor or wall area
5. **Configure Settings** in sidebar (surface type, pattern, grout, tile size…)
6. **Click 🚀 Apply Tile to Floor** — wait ~10–20 s (CPU) or ~2–5 s (GPU)
7. **Review** — Before/After slider, download PNG

### Comparison Mode (New)

1. **Toggle 🔲 Comparison Mode** at the top of the page
2. **Add tiles** from any source — upload, detected crops, or similar tiles catalog. The selection counter shows how many tiles are queued.
3. **Upload Room Image and brush floor** (same as Single Mode)
4. **Click 🎯 Apply Selected Tiles (N)** — a single progress bar tracks geometry + rendering
5. **View comparison grid** — all tile renders side by side, ⭐ best tile highlighted
6. **Download any tile** individually, or click **✅ Use Best Tile** to adopt the winner

---

## Configuration Options

| Setting | Range | Default | Description |
|---|---|---|---|
| **Device** | auto / cpu / cuda | auto | Inference device |
| **Surface Type** | floor / wall | floor | Floor = 3D depth perspective; Wall = flat 2D |
| **Layout Pattern** | straight / diagonal / brick | straight | Tile arrangement |
| **Grout Width** | 0–8 px | 2 | Width of grout lines |
| **Grout Color** | Any hex | #3C3C3C | Grout line color |
| **Brush Size** | 10–80 | 35 | Brush radius for painting |
| **Brush Opacity** | 0.2–1.0 | 0.45 | Canvas brush transparency |
| **Edge Feather** | 5–40 px | 15 | Gaussian blur at tile boundaries |
| **Color Match** | 0.0–1.0 | 0.25 | Strength of LAB color temperature transfer |
| **Tile Size** | 0.2–1.2 m | 0.6 | Physical tile dimension (affects perspective scale) |

---

## Performance Notes

### Single Tile Mode

| Component | CPU | GPU |
|---|---|---|
| MiDaS Depth Estimation | ~8–15 s | ~1–2 s |
| YOLOv8 Detection | ~2–5 s | <1 s |
| ResNet50 Embedding | ~1–2 s | <0.5 s |
| Tile Rendering + Warp | ~0.5–1 s | ~0.5 s |
| Lighting + Compositing | ~0.3–0.5 s | ~0.3 s |
| **Total** | **~12–25 s** | **~3–5 s** |

### Comparison Mode (Multi-Tile)

| Tiles | Without Cache | With Geometry Cache |
|---|---|---|
| 2 tiles | ~32 s | ~18 s |
| 3 tiles | ~48 s | ~20 s |
| 5 tiles | ~80 s | ~24 s |

> Geometry cache = MiDaS depth + homography + lighting (computed once). Each additional tile adds ~2 s render time.

---

## API Reference (Legacy FastAPI)

The `main.py` FastAPI server provides a lightweight headless API:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Home page (HTML upload form) |
| `/upload` | POST | Upload image → YOLOv8 → returns detected tile crops |
| `/select_tile` | POST | Select crop → ResNet50 → returns top-5 similar tiles |

> **Note:** The FastAPI server does **not** include the visualization or comparison engine. For full functionality use the Streamlit UI.

---

## Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| First run is very slow | MiDaS + DeepLabV3 download ~750 MB | Normal — subsequent runs use cached models |
| `CUDA out of memory` | GPU VRAM insufficient | Set Device to `cpu` in sidebar |
| Tiles look flat / no perspective | Surface type is "wall" | Switch to "floor" for 3D perspective |
| Mask coverage too low | Not enough area painted | Paint generously — threshold is 1% of image |
| "No tiles selected" warning in Comparison Mode | Mode toggled without adding tiles | Add tiles first via upload, crop, or similar tiles |
| Comparison grid only shows 1 result | Single tile was selected | Add 2+ tiles to get the comparison view |
| `No module named 'streamlit_drawable_canvas'` | Missing dependency | `pip install streamlit-drawable-canvas` |
| `ModuleNotFoundError: timm` | Missing dependency | `pip install timm` |
| Colors don't match the room | Color match strength low | Increase "Color Match" slider |

---

*Tile Visualization Engine • YOLOv8 + ResNet50 + MiDaS DPT-Large + RANSAC + Homography + Multi-Tile Comparison Mode*
