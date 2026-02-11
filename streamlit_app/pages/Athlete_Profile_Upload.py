import streamlit as st
from utils.load_data import load_strava, load_profile
import sys, os

# Path from pages/ → SimuMatch-main/src/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_PATH = os.path.join(ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


st.title("📤 Athlete Profile")

strava = load_strava()
profile = load_profile()

st.subheader("👤 Profile Information")
st.json(profile.to_dict())

st.subheader("📈 Strava Activity Summary")
st.write(strava.head())

st.success("Profile loaded successfully!")
