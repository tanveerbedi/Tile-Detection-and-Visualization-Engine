from fastapi import FastAPI, File, UploadFile, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request, Form

import shutil
import os
import uuid
import logging

from utils.detection import detect_tiles
from utils.embedding import get_embedding_from_path
from utils.similarity import get_top_k

logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
CROP_DIR = "static/crops"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------- HOME ----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------- UPLOAD + DETECT ----------------
@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        img_path = f"{UPLOAD_DIR}/{file_id}.jpg"

        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        crops = detect_tiles(img_path, CROP_DIR)

        # if no tiles detected
        if not crops:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "No tile detected"}
            )

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "crops": crops
            }
        )
    except Exception as e:
        logger.exception("Error during upload/detection")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Processing error: {e}"}
        )


# ---------------- SELECT TILE → EMBEDDING ----------------
@app.post("/select_tile", response_class=HTMLResponse)
async def select_tile(request: Request, tile: str = Form(...)):
    try:
        tile_name = tile
        tile_path = os.path.join(CROP_DIR, tile_name)

        if not os.path.exists(tile_path):
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "Crop file not found. Please upload again."}
            )

        # embedding
        query_emb = get_embedding_from_path(tile_path)
        # similarity
        results = get_top_k(query_emb, top_k=5)

        return templates.TemplateResponse(
            "similar_result.html",
            {
                "request": request,
                "matches": results
            }
        )
    except Exception as e:
        logger.exception("Error during tile selection")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Processing error: {e}"}
        )



#.\.venv\Scripts\Activate.ps1
#cd tiles_Group4
#uvicorn main:app --reload