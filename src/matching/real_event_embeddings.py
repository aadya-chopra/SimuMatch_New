import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def build_event_text(row):
    """
    Convert event row to descriptive natural language text.
    This matches how we embedded Olympic events.
    """
    parts = []

    parts.append(f"Event name: {row['event_name']}")
    parts.append(f"Sport: {row['sport']}")
    parts.append(f"Event type: {row['event_type']}")
    parts.append(f"Distance: {row['distance_km']} km")
    parts.append(f"Location: {row['location']}")
    parts.append(f"Terrain: {row['terrain']}")
    parts.append(f"Gender category: {row['gender_category']}")
    parts.append(f"Expected duration: {row['expected_duration_hours']} hours")

    return " | ".join(parts)


def embed_real_events():
    input_path = "data/real_events/upcoming_events.csv"
    output_csv = "data/real_events/real_events_with_embeddings.csv"
    output_npy = "data/real_events/real_event_vectors.npy"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cannot find {input_path}")

    print(" Loading real events from:", input_path)
    df = pd.read_csv(input_path)

    # Build text descriptions
    df["text"] = df.apply(build_event_text, axis=1)

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

    print("Encoding real events...")
    embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True)

    # Save embeddings
    np.save(output_npy, embeddings)
    df.to_csv(output_csv, index=False)

    print(f"✅ Saved processed events to {output_csv}")
    print(f"✅ Saved embeddings to {output_npy}")
    print(" Real event embedding pipeline complete!")


if __name__ == "__main__":
    embed_real_events()
