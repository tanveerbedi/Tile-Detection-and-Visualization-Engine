"""
Depth-Aware Virtual Tile Visualization Engine — Streamlit UI
============================================================

Full flow:
1. Upload Tile Image → (optional) YOLOv8 Detection → Select Tile
2. (Optional) Find Similar Tiles from catalog
3. Upload Room Image
4. BRUSH the floor area on the room image using a drawable canvas
5. Choose tile pattern (straight/diagonal/brick)
6. Apply Tile → Visualization Engine
7. Before/After Slider Comparison

The brush tool is the key interaction — user paints the floor region
directly, giving full control over where tiles are applied.
"""

import streamlit as st
import numpy as np
import cv2
import os
import uuid
import tempfile
import logging
import base64
from PIL import Image
from io import BytesIO

# ---- Compatibility shim for streamlit-drawable-canvas ----
# The package uses streamlit.elements.image.image_to_url() which was
# removed in newer Streamlit versions. We monkey-patch it back.
import streamlit.elements.image as _st_image
if not hasattr(_st_image, 'image_to_url'):
    try:
        from streamlit.elements.lib.image_utils import image_to_url as _real_image_to_url
        from streamlit.elements.lib.layout_utils import LayoutConfig
        def _image_to_url(image, width, clamp, channels, output_format, image_id):
            """Forward to the new Streamlit image_to_url with LayoutConfig."""
            return _real_image_to_url(
                image, 
                LayoutConfig(width=width), 
                clamp, 
                channels, 
                output_format, 
                image_id
            )
        _st_image.image_to_url = _image_to_url
    except ImportError:
        def _image_to_url_fallback(image, width, clamp, channels, output_format, image_id):
            """Fallback to base64 if internal API changes again."""
            buf = BytesIO()
            image.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/png;base64,{b64}"
        _st_image.image_to_url = _image_to_url_fallback
# ---- End compatibility shim ----

from streamlit_drawable_canvas import st_canvas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================= PAGE CONFIG =========================

st.set_page_config(
    page_title="Tile Visualization Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================= CUSTOM CSS =========================

st.markdown("""
<style>
    .main .block-container {
        max-width: 1300px;
        padding-top: 1.5rem;
    }
    .main-header {
        text-align: center;
        padding: 0.5rem 0 1.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 1rem;
    }
    .step-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.5rem;
    }
    .step-num {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        font-weight: bold;
        font-size: 14px;
        flex-shrink: 0;
    }
    .brush-help {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }
    .brush-help b { color: #1d4ed8; }
</style>
""", unsafe_allow_html=True)

# ========================= SIDEBAR =========================

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    device_option = st.selectbox(
        "Device", ["auto", "cpu", "cuda"], index=0,
        help="'auto' uses GPU if available, else CPU."
    )

    st.markdown("---")
    st.markdown("### 🎨 Tile Pattern")

    pattern = st.selectbox(
        "Layout Pattern", ["straight", "diagonal", "brick"], index=0,
    )

    grout_width = st.slider("Grout Width (px)", 0, 8, 2)
    grout_hex = st.color_picker("Grout Color", "#3C3C3C")
    grout_rgb = tuple(int(grout_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    grout_bgr = (grout_rgb[2], grout_rgb[1], grout_rgb[0])

    st.markdown("---")
    st.markdown("### 🖌️ Brush Settings")

    brush_size = st.slider("Brush Size", 10, 80, 35, help="Brush radius for painting floor area.")
    brush_opacity = st.slider("Brush Opacity", 0.2, 1.0, 0.45, 0.05)

    st.markdown("---")
    st.markdown("### 🔧 Rendering")

    st.radio("Surface Type", ["floor", "wall"], index=0, horizontal=True, 
             help="'floor' uses 3D perspective depth. 'wall' applies flat tiles.", key="surface_type")

    feather_radius = st.slider("Edge Feather", 5, 40, 15)
    color_match = st.slider("Color Match", 0.0, 1.0, 0.25)
    tile_real_size = st.slider("Tile Size (m)", 0.2, 1.2, 0.6)

    st.markdown("---")
    st.markdown("### ✨ Realism")

    realism_strength = st.slider(
        "Realism Strength", 0.0, 1.0, 0.5, 0.05,
        help="0 = classic rendering, 1 = full realism post-processing."
    )
    variation_amount = st.slider(
        "Tile Variation", 0.0, 0.15, 0.05, 0.01,
        help="Random brightness/contrast variation per tile (\u00b1% range)."
    )
    depth_blur_strength = st.slider(
        "Depth Blur", 0.0, 1.0, 0.4, 0.05,
        help="Blur far tiles slightly to simulate depth of field."
    )
    reflection = st.checkbox(
        "Glossy Reflection", value=False,
        help="Blend a faint reflection of the room onto tile surface (glazed tiles)."
    )

    st.markdown("---")
    st.markdown("### 📊 System")
    import torch
    gpu = torch.cuda.is_available()
    if gpu:
        st.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ CPU mode")

# ========================= HEADER =========================

st.markdown("""
<div class="main-header">
    <h1>🏠 Tile Visualization Engine</h1>
    <p>Brush the floor area, then apply tiles with depth-aware perspective rendering</p>
</div>
""", unsafe_allow_html=True)

# ========================= MODE TOGGLE =========================

st.markdown("""<style>
.mode-toggle label { font-weight: 600; font-size: 1rem; }
.compare-card {
    border: 2px solid #e5e7eb; border-radius: 10px;
    padding: 10px; text-align: center; background: #f9fafb;
    margin-bottom: 6px;
}
.compare-card.best {
    border-color: #22c55e; background: #f0fdf4;
}
.tile-badge {
    display: inline-block; background: #667eea; color: white;
    border-radius: 12px; padding: 2px 10px; font-size: 0.78rem;
    margin-bottom: 4px; font-weight: 600;
}
.tile-badge.best { background: #22c55e; }
</style>""", unsafe_allow_html=True)

_mode_label = st.radio(
    "Visualization Mode",
    ["🖼 Single Tile Mode", "🔲 Comparison Mode (2–5 tiles)"],
    horizontal=True,
    key="mode_radio",
    label_visibility="collapsed",
)
st.session_state.viz_mode = "comparison" if "Comparison" in _mode_label else "single"

if st.session_state.viz_mode == "comparison":
    n_sel = len(st.session_state.selected_tiles)
    st.info(
        f"**Comparison Mode** — select 2–5 tiles below, then click "
        f"**Apply Selected Tiles**. "
        + (f"📌 **{n_sel} tile(s) selected.**" if n_sel else "No tiles selected yet.")
    )
    if n_sel > 0:
        if st.button("🗑 Clear all selected tiles", key="clear_tiles"):
            st.session_state.selected_tiles = []
            st.session_state.comparison_results = []
            st.rerun()

# ========================= SESSION STATE =========================

defaults = {
    "selected_tile_img": None,
    "room_image": None,
    "floor_mask": None,
    "result_image": None,
    "debug_info": None,
    "detected_crops": [],
    "similar_tiles": [],
    "step": 1,
    # ---- Comparison Mode additions ----
    "viz_mode": "single",          # "single" | "comparison"
    "selected_tiles": [],          # list of {image, name, score}
    "comparison_results": [],      # list of result dicts from multi_tile_engine
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))


def load_img_url(url):
    if os.path.exists(url):
        return cv2.imread(url)
    try:
        import urllib.request
        with urllib.request.urlopen(url) as r:
            return cv2.imdecode(np.asarray(bytearray(r.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


# ========================= STEP 1: TILE UPLOAD =========================

st.markdown('<div class="step-header"><span class="step-num">1</span><b>Upload a Tile Image</b></div>',
            unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    tile_upload = st.file_uploader("Upload tile", type=["jpg","jpeg","png","bmp","webp"],
                                    key="tile_up", label_visibility="collapsed")

if tile_upload:
    tile_pil = Image.open(tile_upload).convert("RGB")
    with col2:
        st.image(tile_pil, caption="Tile Texture", width=220)

    mode = st.radio("Use tile as:", ["Direct texture", "Detect tiles (YOLOv8)"], horizontal=True)

    if mode == "Direct texture":
        st.session_state.selected_tile_img = pil_to_cv2(tile_pil)
        st.session_state.step = max(st.session_state.step, 3)
        st.success("✅ Tile selected!")
    else:
        if st.button("🔍 Detect Tiles", type="primary"):
            with st.spinner("Detecting..."):
                tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.jpg")
                tile_pil.save(tmp)
                cdir = os.path.join(tempfile.gettempdir(), "tile_crops")
                os.makedirs(cdir, exist_ok=True)
                try:
                    from detection.detector import detect_tiles
                    crops = detect_tiles(tmp, cdir)
                    st.session_state.detected_crops = [
                        (c, cv2.imread(os.path.join(cdir, c)))
                        for c in crops if cv2.imread(os.path.join(cdir, c)) is not None
                    ]
                    st.session_state.step = max(st.session_state.step, 2)
                except Exception as e:
                    st.error(f"Detection error: {e}")

# Show detected crops
if st.session_state.detected_crops:
    st.markdown("**Select a detected tile:**")
    cols = st.columns(min(len(st.session_state.detected_crops), 5))
    for i, (name, crop) in enumerate(st.session_state.detected_crops):
        with cols[i % len(cols)]:
            st.image(cv2_to_pil(crop), width=140)
            if st.session_state.viz_mode == "single":
                if st.button(f"Use #{i+1}", key=f"sel_{i}"):
                    st.session_state.selected_tile_img = crop
                    st.session_state.step = max(st.session_state.step, 3)
                    st.rerun()
            else:
                # Comparison Mode — checkbox to add/remove from comparison list
                already = any(t["name"] == name for t in st.session_state.selected_tiles)
                label = f"{'✅ ' if already else ''}Add #{i+1}"
                if st.button(label, key=f"cmp_sel_{i}", disabled=(
                    not already and len(st.session_state.selected_tiles) >= 5
                )):
                    if already:
                        st.session_state.selected_tiles = [
                            t for t in st.session_state.selected_tiles if t["name"] != name
                        ]
                    else:
                        st.session_state.selected_tiles.append(
                            {"image": crop, "name": f"Crop #{i+1}", "score": 0.0}
                        )
                    st.rerun()

# ========================= STEP 2: SIMILAR TILES (optional) =========================

if st.session_state.selected_tile_img is not None and st.session_state.step >= 2:
    with st.expander("🔎 Find Similar Tiles *(optional)*", expanded=False):
        if st.button("Find Similar"):
            with st.spinner("Computing embeddings..."):
                try:
                    from recommendation.embedding import get_embedding_from_path
                    from recommendation.similarity import get_top_k
                    tmp = os.path.join(tempfile.gettempdir(), f"s_{uuid.uuid4().hex}.jpg")
                    cv2.imwrite(tmp, st.session_state.selected_tile_img)
                    emb = get_embedding_from_path(tmp)
                    st.session_state.similar_tiles = get_top_k(emb, top_k=5)
                except Exception as e:
                    st.warning(f"Similarity failed: {e}")

        if st.session_state.similar_tiles:
            scols = st.columns(min(len(st.session_state.similar_tiles), 5))
            for i, m in enumerate(st.session_state.similar_tiles):
                with scols[i]:
                    img = load_img_url(m.get("image", ""))
                    if img is not None:
                        st.image(cv2_to_pil(img), width=110)
                        score = m.get('score', 0)
                        st.caption(f"{m.get('name','')}\n{score*100:.0f}%")
                        if st.session_state.viz_mode == "single":
                            if st.button("Use", key=f"sim_{i}"):
                                st.session_state.selected_tile_img = img
                                st.rerun()
                        else:
                            tile_name = m.get('name', f'Similar #{i+1}')
                            already = any(t["name"] == tile_name for t in st.session_state.selected_tiles)
                            btn_label = f"{'✅ Added' if already else '+ Add to Compare'}"
                            if st.button(btn_label, key=f"cmp_sim_{i}", disabled=(
                                not already and len(st.session_state.selected_tiles) >= 5
                            )):
                                if already:
                                    st.session_state.selected_tiles = [
                                        t for t in st.session_state.selected_tiles
                                        if t["name"] != tile_name
                                    ]
                                else:
                                    st.session_state.selected_tiles.append(
                                        {"image": img, "name": tile_name, "score": float(score)}
                                    )
                                st.rerun()

    st.session_state.step = max(st.session_state.step, 3)

# ========================= STEP 3: ROOM IMAGE UPLOAD =========================

if st.session_state.selected_tile_img is not None and st.session_state.step >= 3:
    st.markdown("---")
    st.markdown('<div class="step-header"><span class="step-num">2</span><b>Upload a Room Image</b></div>',
                unsafe_allow_html=True)

    room_upload = st.file_uploader("Upload room", type=["jpg","jpeg","png","bmp","webp"],
                                    key="room_up", label_visibility="collapsed")

    if room_upload:
        room_pil = Image.open(room_upload).convert("RGB")
        room_bgr = pil_to_cv2(room_pil)

        # Resize if too large
        max_dim = 1024
        h, w = room_bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            room_bgr = cv2.resize(room_bgr, (int(w * scale), int(h * scale)))

        st.session_state.room_image = room_bgr
        st.session_state.step = max(st.session_state.step, 4)

# ========================= STEP 4: BRUSH FLOOR AREA =========================

if st.session_state.room_image is not None and st.session_state.step >= 4:
    st.markdown("---")
    st.markdown('<div class="step-header"><span class="step-num">3</span>'
                '<b>Brush the Floor Area</b></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="brush-help">
        🖌️ <b>Paint over the floor area</b> where you want tiles applied.
        Use the brush to cover the entire floor surface.
        Don't paint on walls, furniture, or other non-floor areas.
    </div>
    """, unsafe_allow_html=True)

    room_bgr = st.session_state.room_image
    h, w = room_bgr.shape[:2]
    room_rgb = cv2.cvtColor(room_bgr, cv2.COLOR_BGR2RGB)
    room_pil = Image.fromarray(room_rgb)

    # Calculate display dimensions (fit within 700px width)
    display_w = min(700, w)
    display_h = int(h * display_w / w)

    # Drawing canvas with the room as background
    canvas_result = st_canvas(
        fill_color=f"rgba(255, 0, 0, {brush_opacity})",
        stroke_width=brush_size,
        stroke_color=f"rgba(255, 0, 0, {brush_opacity})",
        background_image=room_pil,
        drawing_mode="freedraw",
        height=display_h,
        width=display_w,
        key="floor_canvas",
    )

    # Extract mask from canvas drawing
    if canvas_result.image_data is not None:
        # The canvas returns RGBA — red channel has our brush strokes
        canvas_img = canvas_result.image_data  # (display_h, display_w, 4)

        # Any pixel that was drawn on will have alpha > 0
        alpha_channel = canvas_img[:, :, 3]
        drawn_mask = (alpha_channel > 10).astype(np.uint8) * 255

        # Resize mask to match original image dimensions
        floor_mask = cv2.resize(drawn_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Clean up the mask (fill gaps, smooth edges)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel_s, iterations=1)

        # BUG FIX: Force it back to binary 0 or 255 just in case interpolation touched it
        floor_mask = (floor_mask > 127).astype(np.uint8) * 255

        mask_coverage = np.sum(floor_mask > 0) / (h * w) * 100

        if mask_coverage > 1.0:
            st.session_state.floor_mask = floor_mask
            st.success(f"✅ Floor area marked: {mask_coverage:.1f}% of image")
            st.session_state.step = max(st.session_state.step, 5)
        elif mask_coverage > 0:
            st.info("Keep brushing — cover more of the floor area for better results.")
        else:
            st.info("👆 Start painting the floor area on the image above.")

# ========================= STEP 5: APPLY TILES =========================

if (st.session_state.room_image is not None and
    st.session_state.floor_mask is not None and
    st.session_state.step >= 5):

    st.markdown("---")
    st.markdown('<div class="step-header"><span class="step-num">4</span>'
                '<b>Apply Tiles</b></div>', unsafe_allow_html=True)

    # ---- Shared sidebar params ----
    _pattern = st.session_state.get('pattern', pattern)
    _tile_real_size = st.session_state.get('tile_real_size', tile_real_size)
    _grout_width = st.session_state.get('grout_width', grout_width)
    _feather_radius = st.session_state.get('feather_radius', feather_radius)
    _color_match = st.session_state.get('color_match', color_match)
    _device = st.session_state.get('device_option', device_option)
    _surface = st.session_state.get('surface_type', 'floor')

    # =====================================================================
    # SINGLE MODE — original flow
    # =====================================================================
    if st.session_state.viz_mode == "single":
        if st.session_state.selected_tile_img is None:
            st.info("Select a tile above to apply.")
        elif st.button("🚀 Apply Tile to Floor", type="primary", use_container_width=True):
            progress = st.progress(0, "Starting pipeline...")
            try:
                from visualization.visualization_engine import apply_tile_to_room_with_mask
                progress.progress(15, "Estimating depth...")
                result = apply_tile_to_room_with_mask(
                    room_image=st.session_state.room_image,
                    tile_image=st.session_state.selected_tile_img,
                    floor_mask=st.session_state.floor_mask,
                    pattern=_pattern,
                    tile_real_size=_tile_real_size,
                    grout_width=_grout_width,
                    grout_color=grout_bgr,
                    feather_radius=_feather_radius,
                    color_match_strength=_color_match,
                    device=_device,
                    surface_type=_surface,
                    realism_strength=realism_strength,
                    variation_amount=variation_amount,
                    depth_blur_strength=depth_blur_strength,
                    reflection=reflection,
                )
                progress.progress(100, "Done! ✅")
                st.session_state.result_image = result["result"]
                st.session_state.debug_info = result
                st.session_state.step = 6
                st.rerun()
            except Exception as e:
                logger.exception("Pipeline error")
                st.error(f"❌ Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    # =====================================================================
    # COMPARISON MODE — multi-tile engine
    # =====================================================================
    else:
        n_sel = len(st.session_state.selected_tiles)

        # Edge-case guards
        if n_sel == 0:
            st.warning("⚠️ No tiles selected. Use the checkboxes above to add tiles for comparison.")
        elif n_sel > 5:
            st.warning("⚠️ Maximum 5 tiles allowed. Please deselect some tiles.")
        else:
            # Show currently selected tile thumbnails
            st.markdown(f"**Selected tiles ({n_sel}):**")
            thumb_cols = st.columns(n_sel)
            for ti, tile_spec in enumerate(st.session_state.selected_tiles):
                with thumb_cols[ti]:
                    st.image(cv2_to_pil(tile_spec["image"]), caption=tile_spec["name"], width=100)

            if n_sel == 1:
                st.info("💡 Only 1 tile selected — falling back to Single Tile mode for this render.")

            apply_btn_label = (
                f"🎯 Apply Selected Tiles ({n_sel})"
                if n_sel > 1
                else "🚀 Apply Tile (Single)"
            )

            if st.button(apply_btn_label, type="primary", use_container_width=True):
                if n_sel == 1:
                    # Single-tile fallback
                    progress = st.progress(0, "Starting pipeline...")
                    try:
                        from visualization.visualization_engine import apply_tile_to_room_with_mask
                        progress.progress(15, "Estimating depth...")
                        result = apply_tile_to_room_with_mask(
                            room_image=st.session_state.room_image,
                            tile_image=st.session_state.selected_tiles[0]["image"],
                            floor_mask=st.session_state.floor_mask,
                            pattern=_pattern,
                            tile_real_size=_tile_real_size,
                            grout_width=_grout_width,
                            grout_color=grout_bgr,
                            feather_radius=_feather_radius,
                            color_match_strength=_color_match,
                            device=_device,
                            surface_type=_surface,
                            realism_strength=realism_strength,
                            variation_amount=variation_amount,
                            depth_blur_strength=depth_blur_strength,
                            reflection=reflection,
                        )
                        progress.progress(100, "Done! ✅")
                        st.session_state.result_image = result["result"]
                        st.session_state.debug_info = result
                        st.session_state.comparison_results = []
                        st.session_state.step = 6
                        st.rerun()
                    except Exception as e:
                        logger.exception("Pipeline error (single fallback)")
                        st.error(f"❌ Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                else:
                    # TRUE multi-tile path
                    progress = st.progress(0, "Computing geometry (once)…")
                    status_ph = st.empty()

                    def _progress_cb(current, total, msg):
                        frac = current / (total + 1)  # +1 to account for geometry step
                        progress.progress(frac, msg)
                        status_ph.caption(f"Step {current}/{total}: {msg}")

                    try:
                        from visualization.multi_tile_engine import render_multiple_tiles

                        all_results = render_multiple_tiles(
                            room_image=st.session_state.room_image,
                            tiles=st.session_state.selected_tiles,
                            floor_mask=st.session_state.floor_mask,
                            device=_device,
                            surface_type=_surface,
                            pattern=_pattern,
                            tile_real_size=_tile_real_size,
                            grout_width=_grout_width,
                            grout_color=grout_bgr,
                            color_match_strength=_color_match,
                            feather_radius=_feather_radius,
                            progress_callback=_progress_cb,
                            realism_strength=realism_strength,
                            variation_amount=variation_amount,
                            depth_blur_strength=depth_blur_strength,
                            reflection=reflection,
                        )

                        progress.progress(1.0, "Done! ✅")
                        status_ph.empty()

                        st.session_state.comparison_results = all_results
                        st.session_state.result_image = None  # clear single result
                        st.session_state.step = 6
                        st.rerun()

                    except Exception as e:
                        logger.exception("Multi-tile pipeline error")
                        st.error(f"❌ Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())

# ========================= STEP 6: RESULTS =========================

if st.session_state.step >= 6:
    st.markdown("---")

    # ==================================================================
    # COMPARISON RESULTS VIEW
    # ==================================================================
    if st.session_state.comparison_results:
        st.markdown("## 🔲 Comparison Results")

        cmp_results = st.session_state.comparison_results
        valid_results = [r for r in cmp_results if r.get("result") is not None]

        if not valid_results:
            st.error("All tile renders failed. Check the logs above for details.")
        else:
            # Find best tile by similarity score
            best_idx = max(
                range(len(valid_results)),
                key=lambda i: valid_results[i].get("similarity_score", 0.0)
            )

            room_rgb_cmp = cv2.cvtColor(st.session_state.room_image, cv2.COLOR_BGR2RGB)

            # --- Comparison grid ---
            grid_cols = st.columns(len(valid_results))
            for ci, (res, col) in enumerate(zip(valid_results, grid_cols)):
                is_best = (ci == best_idx)
                result_rgb = cv2.cvtColor(res["result"], cv2.COLOR_BGR2RGB)
                score = res.get("similarity_score", 0.0)
                name = res.get("tile_name", f"Tile {ci+1}")

                with col:
                    badge_cls = "best" if is_best else ""
                    label_html = (
                        f'<div class="compare-card {badge_cls}">'
                        + (f'<div class="tile-badge best">⭐ Best Match</div><br>' if is_best else "")
                        + f'<div class="tile-badge">{name}</div>'
                        + (f'<br><small>Score: {score*100:.0f}%</small>' if score > 0 else "")
                        + "</div>"
                    )
                    st.markdown(label_html, unsafe_allow_html=True)
                    st.image(result_rgb, use_container_width=True)

                    # Per-tile download
                    tile_buf = BytesIO()
                    Image.fromarray(result_rgb).save(tile_buf, format="PNG")
                    st.download_button(
                        "📥 Download",
                        data=tile_buf.getvalue(),
                        file_name=f"tile_{name.replace(' ', '_').lower()}.png",
                        mime="image/png",
                        key=f"dl_cmp_{ci}",
                        use_container_width=True,
                    )

            # --- Select Best Tile button ---
            st.markdown("---")
            best_name = valid_results[best_idx].get("tile_name", "Best")
            if st.button(
                f"✅ Use Best Tile — {best_name}",
                type="primary",
                key="use_best_tile",
            ):
                st.session_state.selected_tile_img = st.session_state.selected_tiles[best_idx]["image"]
                st.session_state.comparison_results = []
                st.session_state.result_image = valid_results[best_idx]["result"]
                st.session_state.debug_info = valid_results[best_idx]
                st.session_state.viz_mode = "single"
                st.rerun()

            # --- Side-by-side: original vs each result ---
            with st.expander("🔍 Per-Tile Before/After Sliders"):
                for ci, res in enumerate(valid_results):
                    is_best = (ci == best_idx)
                    name = res.get("tile_name", f"Tile {ci+1}")
                    result_rgb = cv2.cvtColor(res["result"], cv2.COLOR_BGR2RGB)
                    st.markdown(f"**{'⭐ ' if is_best else ''}{name}**")
                    slider = st.slider(
                        "Before/After", 0.0, 1.0, 0.5, 0.01,
                        key=f"cmp_slider_{ci}", label_visibility="collapsed"
                    )
                    h2, w2 = room_rgb_cmp.shape[:2]
                    sx = int(w2 * slider)
                    comp = result_rgb.copy()
                    comp[:, :sx] = room_rgb_cmp[:, :sx]
                    cv2.line(comp, (sx, 0), (sx, h2), (255, 255, 255), 3)
                    cv2.line(comp, (sx, 0), (sx, h2), (50, 50, 50), 1)
                    font2 = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(comp, "BEFORE", (max(10, sx-150), 35), font2, 0.9, (255,255,255), 3)
                    cv2.putText(comp, "BEFORE", (max(10, sx-150), 35), font2, 0.9, (0,0,0), 1)
                    cv2.putText(comp, "AFTER", (min(sx+15, w2-130), 35), font2, 0.9, (255,255,255), 3)
                    cv2.putText(comp, "AFTER", (min(sx+15, w2-130), 35), font2, 0.9, (0,0,0), 1)
                    st.image(comp, use_container_width=True)

    # ==================================================================
    # SINGLE RESULT VIEW (existing)
    # ==================================================================
    elif st.session_state.result_image is not None:
        st.markdown("## 🎨 Result")

        # Before / After slider
        st.markdown("### Before / After")
        slider_val = st.slider("Compare", 0.0, 1.0, 0.5, 0.01,
                               key="comp_slider", label_visibility="collapsed")

        room_rgb = cv2.cvtColor(st.session_state.room_image, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(st.session_state.result_image, cv2.COLOR_BGR2RGB)

        h, w = room_rgb.shape[:2]
        split_x = int(w * slider_val)

        comp = result_rgb.copy()
        comp[:, :split_x] = room_rgb[:, :split_x]
        cv2.line(comp, (split_x, 0), (split_x, h), (255, 255, 255), 3)
        cv2.line(comp, (split_x, 0), (split_x, h), (50, 50, 50), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comp, "BEFORE", (max(10, split_x - 150), 35), font, 0.9, (255,255,255), 3)
        cv2.putText(comp, "BEFORE", (max(10, split_x - 150), 35), font, 0.9, (0,0,0), 1)
        cv2.putText(comp, "AFTER", (min(split_x + 15, w - 130), 35), font, 0.9, (255,255,255), 3)
        cv2.putText(comp, "AFTER", (min(split_x + 15, w - 130), 35), font, 0.9, (0,0,0), 1)

        st.image(comp, use_container_width=True)

        # Side by side
        st.markdown("### Full Result")
        c1, c2 = st.columns(2)
        with c1:
            st.image(room_rgb, caption="Original Room", use_container_width=True)
        with c2:
            st.image(result_rgb, caption="With Tiles Applied", use_container_width=True)

        # Download
        buf = BytesIO()
        Image.fromarray(result_rgb).save(buf, format="PNG")
        st.download_button("📥 Download Result", data=buf.getvalue(),
                           file_name="tile_result.png", mime="image/png",
                           use_container_width=True)

        # Debug panel
        if st.session_state.debug_info:
            with st.expander("🔍 Debug / Pipeline Details"):
                debug = st.session_state.debug_info
                timings = debug.get("timings", {})
                total = sum(timings.values())

                st.markdown("#### ⏱️ Timings")
                for step_n, t in timings.items():
                    st.markdown(f"- **{step_n}**: {t:.2f}s ({t/total*100:.0f}%)")
                st.markdown(f"- **Total**: {total:.2f}s")

                st.markdown("#### 🖼️ Intermediate Results")
                dc = st.columns(3)
                with dc[0]:
                    if debug.get("floor_mask") is not None:
                        st.image(debug["floor_mask"], caption="Floor Mask", use_container_width=True)
                with dc[1]:
                    if debug.get("depth_colored") is not None:
                        st.image(cv2.cvtColor(debug["depth_colored"], cv2.COLOR_BGR2RGB),
                                caption="Depth Map", use_container_width=True)
                with dc[2]:
                    if debug.get("masked_tiles") is not None:
                        st.image(cv2.cvtColor(debug["masked_tiles"], cv2.COLOR_BGR2RGB),
                                caption="Warped Tiles", use_container_width=True)

                if debug.get("homography") is not None:
                    st.markdown("#### 📐 Homography Matrix")
                    st.code(np.array2string(debug["homography"], precision=4))

        # AI mode placeholder
        with st.expander("🤖 AI Realistic Mode (Coming Soon)"):
            st.info("Future: ControlNet Depth + SD Inpainting for photo-realistic rendering. **Currently disabled.**")

# ========================= FOOTER =========================

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#9ca3af; font-size:0.8rem'>"
    "Tile Visualization Engine • DeepLabV3 + MiDaS + RANSAC + Homography • Brush-to-Apply"
    "</div>", unsafe_allow_html=True
)
