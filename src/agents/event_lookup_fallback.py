import pandas as pd
from src.api.thesportsdb_api import fetch_endurance_events

def fallback_event_lookup(primary_sports, gender=None):
    """
    Deterministic fallback: API / CSV based
    """

    events_df = fetch_endurance_events()
    if events_df.empty:
        return events_df

    # Normalize
    events_df["sport"] = events_df["sport"].astype(str)
    events_df["league"] = events_df.get("league", "").astype(str)

    if "event_sex" not in events_df.columns:
        events_df["event_sex"] = ""

    # Remove spectator sports
    blocked_leagues = ["mlb", "nba", "nfl", "nhl"]
    blocked_sports = ["baseball", "basketball", "football", "rugby"]

    events_df = events_df[
        ~events_df["league"].str.lower().str.contains("|".join(blocked_leagues), na=False)
    ]
    events_df = events_df[
        ~events_df["sport"].str.lower().str.contains("|".join(blocked_sports), na=False)
    ]

    # Gender filtering
    if gender:
        g = gender.lower()
        if g.startswith("m"):
            events_df = events_df[events_df["event_sex"].isin(["", "M", "Men", "Male"])]
        elif g.startswith("f"):
            events_df = events_df[events_df["event_sex"].isin(["", "F", "Women", "Female"])]

    # Sport intent filtering
    sport_map = {
        "running": ["Athletics", "Marathon"],
        "cycling": ["Cycling"],
        "swimming": ["Swimming"],
    }

    allowed = []
    for s in primary_sports:
        allowed.extend(sport_map.get(s, []))

    if len(primary_sports) >= 3:
        allowed.append("Triathlon")

    if allowed:
        events_df = events_df[events_df["sport"].isin(allowed)]

    return events_df.reset_index(drop=True)
