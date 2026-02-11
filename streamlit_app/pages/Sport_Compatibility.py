import streamlit as st

import os, sys
print("CURRENT FILE:", __file__)
print("WORKING DIR:", os.getcwd())
print("PATHS:", sys.path[:3])
import sys
import sys
from pathlib import Path

# This file → streamlit_app/pages/xxx.py
# We want → SimuMatch-main (project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



from utils.load_data import load_profile, load_strava
from src.matching.match_engine import (
    recommend_cross_sport_from_strava,
    normalize_cross_sport_similarity,
    explain_sport_recommendation
)

st.title("⚖️ Cross-Sport Compatibility")

strava = load_strava()
profile = load_profile()

raw = recommend_cross_sport_from_strava(strava, profile, display_name="Mats")
norm = normalize_cross_sport_similarity(raw)

# Add explainability
explanation_list = []
for _, row in norm.iterrows():
    explanation_list.append(
        explain_sport_recommendation(
            sport=row["sport"],
            similarity=row["similarity"],
            normalized_fit=row["normalized_fit"],
            primary_sports=None,
            strava_df=strava
        )
    )

norm["why_this_sport"] = explanation_list

st.subheader("🔍 Best Sport Matches")
st.dataframe(norm)
