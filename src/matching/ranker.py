# src/matching/ranker.py

import numpy as np


def compute_volume_score(strava_df, sport):
    """
    How much training volume matches this sport?
    """
    if strava_df is None or strava_df.empty:
        return 0.3

    sport_map = {
        "Cycling": ["Ride"],
        "Athletics": ["Run"],
        "Running": ["Run"],
        "Swimming": ["Swim"],
        "Triathlon": ["Run", "Ride", "Swim"],
        "Weightlifting": ["WeightTraining"],
    }

    activities = sport_map.get(sport, [])
    if not activities:
        return 0.2

    counts = strava_df["type"].value_counts()
    total = counts.sum()

    match_count = sum(counts.get(a, 0) for a in activities)

    if total == 0:
        return 0.2

    return round(match_count / total, 3)


def compute_sport_match_score(primary_sports, sport):
    """
    Whether this sport matches athlete's main sports
    """
    if not primary_sports:
        return 0.5

    if sport in primary_sports:
        return 1.0

    return 0.4


def compute_location_score(event_location, athlete_location=None):
    """
    Simple feasibility logic
    """
    if not athlete_location or not event_location:
        return 0.5

    if athlete_location.lower() in event_location.lower():
        return 1.0

    return 0.6


def compute_final_score(similarity, volume_score, sport_match, location_score):
    """
    Final compatibility score
    """
    final = (
        0.5 * similarity +
        0.2 * volume_score +
        0.2 * sport_match +
        0.1 * location_score
    )
    return round(float(final), 4)
