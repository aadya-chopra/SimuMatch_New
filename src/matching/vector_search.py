import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/real_events", exist_ok=True)


print("Loading cleaned athlete data...")
df = pd.read_csv("data/processed/clean_athletes.csv")

# -------------------------
# 1. Vectorized text generation (NO apply)
# -------------------------
def safe(x): 
    return "" if pd.isna(x) else str(x)

df["profile_text"] = (
      df["name"].map(safe)
    + " | " + df["sex"].map(lambda x: "male" if x=="M" else ("female" if x=="F" else ""))
    + " | " + df["age"].map(lambda x: f"{int(x)} years old" if pd.notna(x) else "")
    + " | " + df["height"].map(lambda x: f"height {int(x)} cm" if pd.notna(x) else "")
    + " | " + df["weight"].map(lambda x: f"weight {int(x)} kg" if pd.notna(x) else "")
    + " | " + df["team"].map(lambda x: f"team {x}" if pd.notna(x) else "")
    + " | " + df["noc"].map(lambda x: f"noc {x}" if pd.notna(x) else "")
    + " | plays " + df["sport"].map(safe)
    + " | competes in " + df["event"].map(safe)
)

df["profile_text"] = df["profile_text"].str.replace(r"\s+", " ", regex=True)

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

texts = df["profile_text"].tolist()

print("Encoding athlete embeddings (batched)...")
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

df["emb"] = embeddings.tolist()

# -------------------------
# Save outputs
# -------------------------
np.save("data/athlete_vectors.npy", np.array(embeddings))
df.to_csv("data/processed/athletes_with_embeddings.csv", index=False)

print("Athlete embeddings generated!")
