"""
tune_threshold.py
Replays all stored embeddings from the DB and shows how many unique
identities would be found at different similarity thresholds.
Run AFTER processing your video to find the best threshold for your scene.

Usage: python tune_threshold.py
"""

import sys, pickle, json
import numpy as np
sys.path.insert(0, ".")
from database.db_manager import DatabaseManager

config = json.load(open("config.json"))
db = DatabaseManager(config["database"]["path"])
records = db.get_all_embeddings()
db.close()

if not records:
    print("No embeddings in DB yet. Run main.py first.")
    sys.exit(0)

embeddings = []
for r in records:
    try:
        emb = pickle.loads(r["embedding"])
        embeddings.append((r["face_uuid"], emb))
    except Exception:
        pass

print(f"\nTotal registered face IDs in DB: {len(embeddings)}")
print(f"{'Threshold':>10} | {'Unique clusters':>15} | {'Recommendation'}")
print("-" * 55)

def cluster_count(embeddings, threshold):
    """Greedy clustering: count distinct identities at a given threshold."""
    centroids = []
    for uuid, emb in embeddings:
        matched = False
        for c in centroids:
            sim = float(np.dot(emb, c))
            if sim >= threshold:
                # Update centroid as running mean
                c[:] = (c + emb) / np.linalg.norm(c + emb)
                matched = True
                break
        if not matched:
            centroids.append(emb.copy())
    return len(centroids)

for t in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    count = cluster_count(embeddings, t)
    note = ""
    if t < 0.45:
        note = "← loose (may over-merge)"
    elif t > 0.65:
        note = "← strict (may over-split)"
    elif 0.50 <= t <= 0.60:
        note = "← recommended for crowds"
    print(f"{t:>10.2f} | {count:>15} | {note}")

print("\nSet 'similarity_threshold' in config.json to your chosen value.")
