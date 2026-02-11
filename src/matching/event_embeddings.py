import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/real_events", exist_ok=True)


print("Loading cleaned athlete data...")
df = pd.read_csv("data/processed/clean_athletes.csv")

# -------------------------
# Vectorized event metadata
# -------------------------
event_meta = (
    df.groupby("event")
    .agg({
        "sport": "first",
        "sex": lambda x: x.mode().iloc[0] if not x.mode().empty else None
    })
    .reset_index()
)

event_meta.rename(columns={"sex": "event_sex"}, inplace=True)

# vectorized event text
event_meta["event_text"] = (
    event_meta["sport"].fillna("") + " "
    + event_meta["event_sex"].map(lambda x: "men " if x=="M" else ("women " if x=="F" else "")) 
    + event_meta["event"]
)

event_meta["event_text"] = event_meta["event_text"].str.replace(r"\s+", " ", regex=True)

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

texts = event_meta["event_text"].tolist()

print("⚡ Encoding event embeddings (batched)...")
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

event_meta["emb"] = embeddings.tolist()

np.save("data/event_vectors.npy", np.array(embeddings))
event_meta.to_csv("data/processed/events_with_embeddings.csv", index=False)

print("Event embeddings saved!")