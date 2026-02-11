import requests
import pandas as pd

BASE_URL = "https://www.thesportsdb.com/api/v1/json/3"

def fetch_endurance_events():
    """
    Fetch endurance-style events from TheSportsDB.
    Uses FREE public API key: 3
    """

    sports = ["Athletics", "Triathlon", "Cycling", "Swimming"]
    all_events = []

    for sport in sports:
        try:
            # Search events by sport name
            url = f"{BASE_URL}/searchevents.php"
            params = {"e": sport}

            r = requests.get(url, params=params, timeout=10)

            # ❗ important safety check
            if r.status_code != 200:
                print(f"API error for {sport}: HTTP {r.status_code}")
                continue

            # Sometimes API returns empty body
            if not r.text.strip():
                print(f"API error for {sport}: empty response")
                continue

            data = r.json()

            if not data or "event" not in data or data["event"] is None:
                continue

            for ev in data["event"]:
                all_events.append({
                    "event_name": ev.get("strEvent"),
                    "sport": sport,
                    "league": ev.get("strLeague"),
                    "date": ev.get("dateEvent"),
                    "location": ev.get("strCountry"),
                    "description": ev.get("strDescriptionEN"),
                })

        except Exception as e:
            print(f"API error for {sport}: {e}")

    if not all_events:
        return pd.DataFrame()

    return pd.DataFrame(all_events)
