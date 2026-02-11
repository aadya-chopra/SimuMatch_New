import streamlit as st
from utils.load_data import load_profile, load_strava
from src.matching.match_engine import recommend_real_events_from_strava
import sys
import os
from pathlib import Path

# --------------------------------------------------
# Path setup (KEEP AS IS)
# --------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

print("CURRENT FILE:", __file__)
print("WORKING DIR:", os.getcwd())
print("PATHS:", sys.path[:3])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🌍 Real Event Recommendations")

# Load data
strava = load_strava()
profile = load_profile()

# --------------------------------------------------
# 🔹 ADD THIS BLOCK (NEW)
# --------------------------------------------------
st.subheader("⏳ Participation Timeline")

time_horizon = st.selectbox(
    "When do you want to participate?",
    options=[
        "Any time",
        "Within 3 months",
        "Within 6 months",
        "Within 1 year",
        "More than 1 year"
    ],
    index=2  # default = Within 6 months
)

# --------------------------------------------------
# Call engine WITH time_horizon
# --------------------------------------------------
events = recommend_real_events_from_strava(
    strava_df=strava,
    profile_row=profile,
    display_name="Mats",
    time_horizon=time_horizon
)

# --------------------------------------------------
# Display
# --------------------------------------------------
st.subheader("🏁 Top Real-World Event Matches")
for _, row in events.iterrows():
    st.markdown(f"### 🏁 {row['event_name']}")
    st.markdown(f"**Sport:** {row['sport']} | **League:** {row['league']}")
    st.markdown(f"**Date:** {row['date']} | **Location:** {row['location']}")
    st.markdown(f"**Readiness:** {row['readiness']}")
    st.markdown(f"🤖 Agent confidence: {row['agent_confidence']:.1f}%")
    st.markdown(f"**Why recommended:** {row['why_recommended']}")
    st.markdown(f"📝 *{row.get('description', '')}*")
    st.markdown(f"📈 Training feasibility score: {row['training_feasibility']:.2f}")

    st.divider()
