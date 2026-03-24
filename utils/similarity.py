import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ================= LOAD DATA =================
catalog_features = np.load("models/tile_embeddings.npy")
image_names = np.load("models/tile_labels.npy", allow_pickle=True)

metadata = pd.read_csv("models/df_balanced.csv")

# ensure .jpg
metadata["img_name"] = (
    metadata["img_name"]
    .astype(str)
    .str.strip()
    .apply(lambda x: x if x.endswith(".jpg") else x + ".jpg")
)

image_names = np.array([str(x).strip() for x in image_names])


# ================= MAIN FUNCTION =================
def get_top_k(query_embedding, top_k=5):

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
            "image": row["img_url"],              # image path for HTML
            "name": row["img_name"],
            "score": round(sim_score, 3),
            "size": row.get("size", ""),
            "texture": row.get("texture", ""),
            "color": row.get("color", "")
        })

    return results
