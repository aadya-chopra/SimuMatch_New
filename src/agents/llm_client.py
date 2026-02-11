import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

def get_groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("❌ GROQ_API_KEY not found in environment")
        return None

    print("✅ Groq client initialized")
    return Groq(api_key=key)