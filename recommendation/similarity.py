"""
Tile Similarity Module
---------------------
Loads precomputed catalog embeddings and metadata.
Returns top-K visually similar tiles for a given query embedding.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ================= LOAD DATA =================
_base_dir = os.path.dirname(os.path.dirname(__file__))
catalog_features = np.load(os.path.join(_base_dir, "models", "tile_embeddings.npy"))
image_names = np.load(os.path.join(_base_dir, "models", "tile_labels.npy"), allow_pickle=True)

metadata = pd.read_csv(os.path.join(_base_dir, "models", "df_balanced.csv"))

# ensure .jpg extension
metadata["img_name"] = (
    metadata["img_name"]
    .astype(str)
    .str.strip()
    .apply(lambda x: x if x.endswith(".jpg") else x + ".jpg")
)

image_names = np.array([str(x).strip() for x in image_names])


# ================= MAIN FUNCTION =================
def get_top_k(query_embedding: np.ndarray, top_k: int = 5) -> list:
    """
    Find the top-K most similar tiles from the catalog.

    Args:
        query_embedding: L2-normalized feature vector.
        top_k: Number of results to return.

    Returns:
        List of dicts with image URL, name, score, size, texture, color.
    """
    similarities = cosine_similarity([query_embedding], catalog_features)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for idx in top_indices:
        img_name = image_names[idx]
        sim_score = float(similarities[idx])

        row = metadata[metadata["img_name"] == img_name]
        if row.empty:
            continue

        row = row.iloc[0]

        results.append({
            "image": row["img_url"],
            "name": row["img_name"],
            "score": round(sim_score, 3),
            "size": row.get("size", ""),
            "texture": row.get("texture", ""),
            "color": row.get("color", "")
        })

    return results
