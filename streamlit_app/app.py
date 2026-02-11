import streamlit as st
import sys
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# This file → streamlit_app/pages/xxx.py
# We want → SimuMatch-main (project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


st.set_page_config(
    page_title="SimuMatch",
    page_icon="🏅",
    layout="wide"
)

st.title("🏅 SimuMatch – AI Athlete Event Matcher")
st.markdown("""
Welcome to **SimuMatch**, your AI-powered event recommendation system.

Use the navigation panel to:

### 1️ Upload athlete training data  
### 2️ View cross-sport compatibility  
### 3️ Discover real-world events matched to your profile  

---
""")
