import pandas as pd
import numpy as np
import sys, os

# Path from pages/ → SimuMatch-main/src/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_PATH = os.path.join(ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

def load_strava():
    return pd.read_csv("data/user_input/strava_activities_rows.csv")

def load_profile():
    return pd.read_csv("data/user_input/profiles_rows.csv").iloc[0]

def load_real_events():
    df = pd.read_csv("data/real_events/real_events_with_embeddings.csv")
    df["emb"] = list(np.load("data/real_events/real_event_vectors.npy", allow_pickle=True))
    return df
