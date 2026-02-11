import pandas as pd
from datetime import datetime

def compute_training_feasibility(sport, strava_df, event_date):
    """
    Returns a score [0–1] indicating if athlete can realistically prepare
    """

    if pd.isna(event_date):
        return 0.3

    today = pd.Timestamp.today()
    event_date = pd.to_datetime(event_date, errors="coerce")

    if event_date <= today:
        return 0.1

    weeks_to_event = max((event_date - today).days / 7, 1)

    # Athlete weekly volume (last 4 weeks)
    recent = strava_df.tail(28)
    weekly_volume = recent["distance"].sum() / 4

    # Rough readiness heuristic
    if weeks_to_event >= 24:
        return 0.9
    elif weeks_to_event >= 12:
        return 0.7
    elif weeks_to_event >= 6:
        return 0.5
    else:
        return 0.2
