"""
Main runner for SimuMatch
-------------------------
Demonstrates:
1. Strava + profile ingestion
2. Cross-sport event matching
3. Cross-sport normalization
4. Explainable AI (Why this sport?)
"""

import pandas as pd

from src.matching.match_engine import (
    recommend_cross_sport_from_strava,
    normalize_cross_sport_similarity,
    explain_sport_recommendation,
    recommend_real_events_from_strava
)


def run_mats_demo():
    print("\n Running SimuMatch demo for athlete: Mats\n")

    # -----------------------------
    # 1. Load user input data
    # -----------------------------
    strava_df = pd.read_csv("data/user_input/strava_activities_rows.csv")
    profiles_df = pd.read_csv("data/user_input/profiles_rows.csv")

    profile_row = profiles_df.iloc[0]

    print(" Loaded Strava & profile data")
    print(f"   Activities: {len(strava_df)}")
    print(f"   Athlete gender: {profile_row.get('gender', 'unknown')}")
    print("-" * 50)

    # -----------------------------
    # 2. Cross-sport matching
    # -----------------------------
    cross_sport = recommend_cross_sport_from_strava(
        strava_df=strava_df,
        profile_row=profile_row,
        top_k_per_sport=1,
        display_name="Mats",
    )

    print(" Cross-sport best matches (raw similarity):")
    print(cross_sport)
    print("-" * 50)

    # -----------------------------
    # 3. Normalize across sports
    # -----------------------------
    normalized = normalize_cross_sport_similarity(cross_sport)

    print(" Normalized cross-sport compatibility:")
    print(normalized)
    print("-" * 50)

    # -----------------------------
    # 4. Explainable AI (Why this sport?)
    # -----------------------------
    explanations = []
    for _, row in normalized.iterrows():
        explanation = explain_sport_recommendation(
            sport=row["sport"],
            similarity=row["similarity"],
            normalized_fit=row["normalized_fit"],
            primary_sports=None,
            strava_df=strava_df,
        )
        explanations.append(explanation)

    normalized["why_this_sport"] = explanations

    print(" Explainable recommendations:")
    print(normalized)
    print("-" * 50)

    print(" Demo complete.\n")


def run_real_event_demo():
    print("\n Recommending REAL upcoming events for Mats\n")

    strava_df = pd.read_csv("data/user_input/strava_activities_rows.csv")
    profiles_df = pd.read_csv("data/user_input/profiles_rows.csv")
    profile_row = profiles_df.iloc[0]

    real_recs = recommend_real_events_from_strava(
        strava_df=strava_df,
        profile_row=profile_row,
        top_k=5,
        display_name="Mats",
    )

    print("Top compatible real-world events:")
    print(real_recs)
    print("-" * 50)

if __name__ == "__main__":
    run_mats_demo()
    run_real_event_demo()

if __name__ == "__main__":
    run_mats_demo()
