import json
import re
import pandas as pd
from datetime import date


def _safe_parse_llm_json(text: str):
    try:
        text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        return json.loads(match.group())
    except Exception as e:
        print("DEBUG: JSON parsing failed:", e)
        return None


def llm_event_lookup(primary_sports, gender=None, location=None, time_horizon=None, llm=None):
    if llm is None:
        print("DEBUG: LLM client is None")
        return None

    today = date.today()

    SYSTEM_PROMPT = f"""
You are an AI sports event discovery agent.

Return ONLY valid JSON.
NO markdown. NO explanations.

RULES:
- ONLY participation endurance events
- Sports: running, cycling, swimming, triathlon
- Events MUST be AFTER {today.isoformat()}
- Prefer FUTURE events (6–36 months ahead)

JSON SCHEMA:
{{
  "events": [
    {{
      "event_name": "",
      "sport": "",
      "league": "",
      "date": "YYYY-MM-DD",
      "location": "",
      "description": ""
    }}
  ]
}}
"""

    user_prompt = f"""
Athlete primary sports: {primary_sports}
Gender: {gender or "any"}
Location: {location or "global"}
Target participation window: {time_horizon or "any time"}
"""

    try:
        response = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        raw = response.choices[0].message.content
        print("DEBUG LLM RAW OUTPUT:\n", raw)

        data = _safe_parse_llm_json(raw)
        if not data or "events" not in data:
            return None

        df = pd.DataFrame(data["events"])
        if df.empty:
            return None

        # ✅ CRITICAL FIX: convert date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()]
        df = df[df["date"].dt.date > today]

        print(f"DEBUG: LLM returned {len(df)} future events")
        return df.reset_index(drop=True)

    except Exception as e:
        print("DEBUG: LLM lookup failed:", e)
        return None
