import numpy as np
import pandas as pd
from src.agents.llm_client import get_groq_client

from src.agents.event_lookup_agent import event_lookup_agent
from src.agents.compatibility_agent import dual_similarity_agent
from src.agents.insight_agent import build_why_recommended, compute_readiness
from src.matching.training_feasibility import compute_training_feasibility
from src.matching.event_difficulty import estimate_event_difficulty
from src.xai.why_recommended import build_why_recommended_v2

from sklearn.metrics.pairwise import cosine_similarity
from src.api.thesportsdb_api import fetch_endurance_events
from src.matching.ranker import (
    compute_volume_score,
    compute_sport_match_score,
    compute_location_score,
    compute_final_score
)

EVENT_TRAINING_WEEKS = {
    "marathon": 16,
    "half marathon": 10,
    "running": 12,
    "cycling": 12,
    "swimming": 10,
    "triathlon": 20,
    "ironman": 40,
}

# Load processed data
athlete_df = pd.read_csv("data/processed/athletes_with_embeddings.csv")
event_df = pd.read_csv("data/processed/events_with_embeddings.csv")

# Load embeddings
ath_embeddings = np.load("data/athlete_vectors.npy", allow_pickle=True)
event_embeddings = np.load("data/event_vectors.npy", allow_pickle=True)

# Attach embeddings to dataframes (aligned by row order)
athlete_df["emb"] = list(ath_embeddings)
event_df["emb"] = list(event_embeddings)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

import re
def build_profile_text_from_strava(strava_df: pd.DataFrame, profile_row: pd.Series, display_name: str = None) -> str:
    """
    Turn Strava + profile info into a single descriptive text string
    that we can embed with SentenceTransformer.
    """

    # ----- basic identity -----
    first_name = profile_row.get("first_name", "")
    last_name  = profile_row.get("last_name", "")
    gender     = profile_row.get("gender", "")
    dob        = profile_row.get("date_of_birth", "")

    name = display_name or f"{first_name} {last_name}".strip()

    # Rough age from date_of_birth if you want (optional – can leave as text)
    age_text = ""
    if isinstance(dob, str) and len(dob) >= 4:
        year = dob[:4]
        age_text = f"born in {year}"

    gender_text = ""
    if isinstance(gender, str) and gender:
        gender_text = gender.lower()

    # ----- separate by sport type -----
    runs  = strava_df[strava_df["type"] == "Run"]
    rides = strava_df[strava_df["type"] == "Ride"]
    swims = strava_df[strava_df["type"] == "Swim"]

    def km(x): 
        return float(x) / 1000.0

    # Running stats
    run_km_total = km(runs["distance"].sum()) if not runs.empty else 0.0
    run_km_long  = km(runs["distance"].max()) if not runs.empty else 0.0

    # Cycling stats
    ride_km_total = km(rides["distance"].sum()) if not rides.empty else 0.0
    ride_km_long  = km(rides["distance"].max()) if not rides.empty else 0.0

    # Swimming stats
    swim_km_total = km(swims["distance"].sum()) if not swims.empty else 0.0
    swim_km_long  = km(swims["distance"].max()) if not swims.empty else 0.0

    # Overall training load (very rough)
    total_hours = strava_df["moving_time"].sum() / 3600.0

    # ----- infer primary sports -----
    primary_sports = []
    if run_km_total > 0:
        primary_sports.append("running")
    if ride_km_total > 0:
        primary_sports.append("cycling")
    if swim_km_total > 0:
        primary_sports.append("swimming")

    # If all three present -> triathlete vibe
    if len(primary_sports) >= 3:
        sport_label = "triathlon and endurance multi-sport"
    elif len(primary_sports) == 2:
        sport_label = "endurance multi-sport"
    elif len(primary_sports) == 1:
        sport_label = primary_sports[0]
    else:
        sport_label = "general fitness"

    parts = []

    parts.append(name)
    if gender_text:
        parts.append(gender_text)
    if age_text:
        parts.append(age_text)

    parts.append(f"focuses on {sport_label}")

    # Add numeric flavour
    if run_km_total > 0:
        parts.append(f"total running distance {run_km_total:.1f} km")
        parts.append(f"longest run {run_km_long:.1f} km")
    if ride_km_total > 0:
        parts.append(f"total cycling distance {ride_km_total:.1f} km")
        parts.append(f"longest ride {ride_km_long:.1f} km")
    if swim_km_total > 0:
        parts.append(f"total swimming distance {swim_km_total:.1f} km")
        parts.append(f"longest swim {swim_km_long:.1f} km")

    parts.append(f"approximate total training time {total_hours:.1f} hours")

    # Final text (this is what we embed)
    profile_text = " | ".join(parts)
    return profile_text, primary_sports, gender_text

# Decide which column is the name column
NAME_COL = "name" if "name" in athlete_df.columns else "Name"

# Clean names and precompute first / last name for robust matching
def _split_first_last(full_name: str):
    full_name = str(full_name).lower()
    # keep only letters and spaces
    full_name = re.sub(r"[^a-z ]", " ", full_name)
    tokens = [t for t in full_name.split() if t]
    if not tokens:
        return None, None
    return tokens[0], tokens[-1]

firsts = []
lasts = []
for n in athlete_df[NAME_COL].astype(str):
    f, l = _split_first_last(n)
    firsts.append(f)
    lasts.append(l)

athlete_df["_first_name"] = firsts
athlete_df["_last_name"] = lasts



def find_athlete_index(name: str):
    """
    Robust athlete name matcher:
    1. exact match on full name
    2. match on first + last name (e.g. 'usain bolt' -> 'Usain St. Leo Bolt')
    3. full substring match
    4. ALL tokens must appear (AND)
    """
    name = name.lower().strip()

    name_col = "name" if "name" in athlete_df.columns else "Name"
    names = athlete_df[name_col].astype(str).str.lower()

    # 1) exact match (full string)
    exact = athlete_df[names == name]
    if len(exact) > 0:
        return exact.index[0]

    # Prepare query tokens
    import re
    cleaned = re.sub(r"[^a-z ]", " ", name)
    tokens = [t for t in cleaned.split() if t]

    # 2) first + last name match (best for cases like 'Usain Bolt')
    if len(tokens) >= 2:
        q_first, q_last = tokens[0], tokens[-1]
        mask_fl = (athlete_df["_first_name"] == q_first) & (athlete_df["_last_name"] == q_last)
        fl_matches = athlete_df[mask_fl]
        if len(fl_matches) > 0:
            return fl_matches.index[0]

    # 3) full substring
    contains = athlete_df[names.str.contains(name, na=False)]
    if len(contains) > 0:
        return contains.index[0]

    # 4) ALL tokens must be present (AND)
    if tokens:
        mask = pd.Series(True, index=names.index)
        for t in tokens:
            mask &= names.str.contains(t, na=False)
        all_tokens = athlete_df[mask]
        if len(all_tokens) > 0:
            return all_tokens.index[0]

    # nothing reasonable found
    return None
def is_participation_event(event_row):
    """
    Return True only for events where an athlete can participate,
    not spectator-only matches.
    """
    sport = str(event_row.get("sport", "")).lower()
    league = str(event_row.get("league", "")).lower()

    # Exclude team spectator sports
    blocked_leagues = ["mlb", "nba", "nfl", "nhl", "afl"]
    blocked_sports = ["baseball", "basketball", "football", "rugby"]

    if any(b in league for b in blocked_leagues):
        return False
    if any(b in sport for b in blocked_sports):
        return False

    return True

def recommend_events_from_strava(strava_df: pd.DataFrame,
                                 profile_row: pd.Series,
                                 top_k: int = 5,
                                 display_name: str = None):
    """
    Recommend events for an external athlete (not in 120-year dataset)
    using Strava + profile data.
    """

    # 1) Build profile text from Strava + profile
    profile_text, primary_sports, gender_text = build_profile_text_from_strava(
        strava_df,
        profile_row,
        display_name=display_name,
    )

    # 2) Embed this synthetic athlete profile
    athlete_vector = model.encode([profile_text])
    athlete_vector = athlete_vector.reshape(1, -1)

    # 3) Map primary_sports -> Olympic dataset sport names
    # You can tweak this mapping as you like
    sport_candidates = set()
    if "running" in primary_sports:
        sport_candidates.add("Athletics")
    if "cycling" in primary_sports:
        sport_candidates.add("Cycling")
    if "swimming" in primary_sports:
        sport_candidates.add("Swimming")
    if len(primary_sports) >= 3:
        sport_candidates.add("Triathlon")

    filtered = event_df.copy()

    # Filter by sport(s) if we have some
    if sport_candidates and "sport" in filtered.columns:
        filtered = filtered[filtered["sport"].isin(list(sport_candidates))]

    # Filter by sex (event_sex) if we can infer from profile gender
    if gender_text and "event_sex" in filtered.columns:
        # map 'male'/'female' -> 'M'/'F'
        g = gender_text.lower()
        if g.startswith("m"):
            filtered = filtered[filtered["event_sex"] == "M"]
        elif g.startswith("f"):
            filtered = filtered[filtered["event_sex"] == "F"]

    # Fallbacks if filter too strict
    if filtered.empty:
        filtered = event_df.copy()

    # 4) Compute similarity to all filtered events
    event_matrix = np.vstack(filtered["emb"].to_list())
    sims = cosine_similarity(athlete_vector, event_matrix)[0]

    top_indices = sims.argsort()[::-1][:top_k]
    top_events = filtered.iloc[top_indices].copy()
    top_events["similarity"] = sims[top_indices]

    # Show nice columns
    cols = [c for c in ["sport", "event", "event_sex", "similarity"] if c in top_events.columns]
    return top_events[cols].reset_index(drop=True)


def recommend_cross_sport_from_strava(
    strava_df: pd.DataFrame,
    profile_row: pd.Series,
    sports=None,
    top_k_per_sport: int = 1,
    display_name: str = None,
):
    """
    For a given Strava + profile athlete, show the top event in EACH sport
    (e.g. Athletics, Cycling, Swimming) so we can compare cross-sport similarity.

    Returns a DataFrame with columns:
      sport, event, event_sex, similarity
    """

    # 1) Build profile text and embed (re-use existing function + model)
    profile_text, _, gender_text = build_profile_text_from_strava(
        strava_df,
        profile_row,
        display_name=display_name,
    )
    athlete_vector = model.encode([profile_text]).reshape(1, -1)

    # 2) Which sports to consider?
    # If none given, use these common endurance sports
    if sports is None:
        sports = ["Athletics", "Cycling", "Swimming", "Triathlon", "Weightlifting"]

    results = []

    for sp in sports:
        # subset events in this sport
        subset = event_df[event_df["sport"] == sp].copy()
        if subset.empty:
            continue

        # filter by sex if we know gender
        if gender_text and "event_sex" in subset.columns:
            g = gender_text.lower()
            if g.startswith("m"):
                subset = subset[subset["event_sex"] == "M"]
            elif g.startswith("f"):
                subset = subset[subset["event_sex"] == "F"]

        if subset.empty:
            continue

        # build matrix and compute similarity
        event_matrix = np.vstack(subset["emb"].to_list())
        sims = cosine_similarity(athlete_vector, event_matrix)[0]

        # take top_k_per_sport for this sport
        top_idx = sims.argsort()[::-1][:top_k_per_sport]
        top_subset = subset.iloc[top_idx].copy()
        top_subset["similarity"] = sims[top_idx]
        top_subset["sport_rank"] = list(range(1, len(top_subset) + 1))

        results.append(top_subset[["sport", "event", "event_sex", "sport_rank", "similarity"]])

    if not results:
        return pd.DataFrame(columns=["sport", "event", "event_sex", "sport_rank", "similarity"])

    # concat all sports results
    return pd.concat(results, ignore_index=True)

def normalize_cross_sport_similarity(cross_sport_df: pd.DataFrame):
    """
    Normalize similarity scores across sports so that
    no single sport (e.g., cycling) dominates purely due
    to higher training volume.

    Input: output of recommend_cross_sport_from_strava
    Output: same dataframe + normalized_fit (0–1)
    """

    if cross_sport_df.empty:
        return cross_sport_df

    df = cross_sport_df.copy()

    min_sim = df["similarity"].min()
    max_sim = df["similarity"].max()

    if max_sim > min_sim:
        df["normalized_fit"] = (df["similarity"] - min_sim) / (max_sim - min_sim)
    else:
        df["normalized_fit"] = 0.0

    # Optional: sort by normalized score
    df = df.sort_values("normalized_fit", ascending=False)

    return df.reset_index(drop=True)

def explain_sport_recommendation(
    sport: str,
    similarity: float,
    normalized_fit: float,
    primary_sports: list,
    strava_df: pd.DataFrame,
):
    """
    Generate a human-readable explanation for why a sport
    was recommended for the athlete.
    """

    explanations = []

    # Volume-based reasoning
    if sport.lower() == "cycling":
        explanations.append(
            "High cycling compatibility due to significant cycling training volume and long-duration rides."
        )

    if sport.lower() == "athletics":
        explanations.append(
            "Running-based events align with endurance components observed in training history."
        )

    if sport.lower() == "swimming":
        explanations.append(
            "Swimming activity present, but comparatively lower volume reduces overall compatibility."
        )

    if sport.lower() == "triathlon":
        explanations.append(
            "Multi-sport training pattern (running, cycling, swimming) aligns with triathlon demands."
        )

    # Comparative reasoning
    if normalized_fit > 0.8:
        explanations.append(
            "This sport shows one of the highest relative compatibilities across all disciplines."
        )
    elif normalized_fit > 0.4:
        explanations.append(
            "This sport shows moderate compatibility compared to other disciplines."
        )
    else:
        explanations.append(
            "This sport ranks lower relative to other disciplines due to weaker alignment with training patterns."
        )

    return " ".join(explanations)

# -------------------------------------------------
# Explainability helpers
# -------------------------------------------------

def build_why_recommended(row, primary_sports):
    sport = str(row.get("sport", "")).lower()
    reasons = []

    if "running" in primary_sports and sport in ["athletics", "marathon"]:
        reasons.append("Strong alignment with your running training volume.")

    if "cycling" in primary_sports and sport == "cycling":
        reasons.append("Matches your high cycling endurance and long-ride history.")

    if "swimming" in primary_sports and sport == "swimming":
        reasons.append("Based on your consistent swimming sessions.")

    if len(primary_sports) >= 3 and sport == "triathlon":
        reasons.append("Your multi-sport training makes you well-suited for triathlon.")

    if not reasons:
        reasons.append("General endurance compatibility based on your activity profile.")

    return " ".join(reasons)


def compute_readiness(score):
    if score >= 0.65:
        return "Competition Ready"
    elif score >= 0.45:
        return "Almost Ready – short training phase needed"
    else:
        return "Not Ready Yet – build more base fitness"
    
def compute_training_feasibility(event_sport, event_date, strava_df):
    try:
        event_date = pd.to_datetime(event_date)
    except:
        return 0.0
    
    today = pd.Timestamp.today()
    weeks_available = max((event_date - today).days / 7, 0)

    sport_key = str(event_sport).lower()
    required_weeks = EVENT_TRAINING_WEEKS.get(sport_key, 12)

    weekly_hours = strava_df["moving_time"].sum() / 3600 / max(len(strava_df) / 7, 1)

    volume_factor = min(weekly_hours / 6, 1.0)  # 6h/week baseline

    feasibility = min((weeks_available / required_weeks) * volume_factor, 1.0)
    return round(feasibility, 2)

def compute_agent_confidence(
    similarity,
    sport_match,
    training_feasibility,
    location_score
):
    confidence = (
        0.35 * similarity +
        0.25 * sport_match +
        0.25 * training_feasibility +
        0.15 * location_score
    )
    return round(confidence * 100, 1)

    
def filter_by_time_horizon(df, time_horizon):
    if df.empty or "date" not in df.columns:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    today = pd.Timestamp.today()

    if time_horizon == "Within 3 months":
        cutoff = today + pd.DateOffset(months=3)
    elif time_horizon == "Within 6 months":
        cutoff = today + pd.DateOffset(months=6)
    elif time_horizon == "Within 1 year":
        cutoff = today + pd.DateOffset(years=1)
    else:
        return df  # Any time

    return df[(df["date"] >= today) & (df["date"] <= cutoff)]


def recommend_real_events_from_strava(
    strava_df,
    profile_row,
    top_k=5,
    display_name="User",
    time_horizon="Any time"
):
    """
    Recommend REAL upcoming endurance events using:
    - Agent-based event lookup
    - Dual cosine similarity
    - Hybrid rule-based ranking
    - XAI explanations
    """

    # --------------------------------------------------
    # 1️⃣ Build athlete profile FIRST (FIX)
    # --------------------------------------------------
    profile_text, primary_sports, gender_text = build_profile_text_from_strava(
        strava_df,
        profile_row,
        display_name
    )

    athlete_vector = model.encode([profile_text]).reshape(1, -1)

    # --------------------------------------------------
    # 2️⃣ Agent: Fetch upcoming events (uses primary_sports)
    # --------------------------------------------------
    llm = get_groq_client()
   

    events_df = event_lookup_agent(
        profile_row=profile_row,
        primary_sports=primary_sports,
        gender=gender_text,
        time_horizon=time_horizon,
        llm=llm
    )



    if events_df.empty:
        return pd.DataFrame({"message": ["No endurance events found"]})
    


    # --------------------------------------------------
    # 3️⃣ Prepare event text for embeddings
    # --------------------------------------------------
    if "description" not in events_df.columns:
        events_df["description"] = ""

    event_texts = (
        events_df["event_name"].fillna("") + " " +
        events_df["sport"].fillna("") + " " +
        events_df["description"].fillna("")
    ).tolist()

    real_event_vectors = model.encode(event_texts)

    # --------------------------------------------------
    # 4️⃣ Dual cosine similarity agent
    # (synthetic vector placeholder for now)
    # --------------------------------------------------
    _, sim_real = dual_similarity_agent(
        athlete_vector,
        np.zeros_like(athlete_vector),  # placeholder synthetic vector
        real_event_vectors
    )

    events_df["similarity"] = sim_real

    # --------------------------------------------------
    # 5️⃣ Hybrid ranker (ML + rules)
    # --------------------------------------------------
    final_scores = []
    training_scores = []
    confidence_scores = []

    for _, row in events_df.iterrows():
        volume_score = compute_volume_score(strava_df, row["sport"])
        sport_match = compute_sport_match_score(primary_sports, row["sport"])
        location_score = compute_location_score(
            row.get("location"),
            profile_row.get("location", None)
        )

        training_feasibility = compute_training_feasibility(
             row["sport"],
             row["date"],
             strava_df
        )
        
        confidence = compute_agent_confidence(
            similarity=row["similarity"],
            sport_match=sport_match,
            training_feasibility=training_feasibility,
            location_score=location_score
        )

        training_scores.append(round(training_feasibility, 3))
        confidence_scores.append(confidence)
    
        final = (
            0.30 * row["similarity"] +
            0.20 * training_feasibility +
            0.20 * sport_match +
            0.15 * volume_score +
            0.10 * (1 - abs(training_feasibility - estimate_event_difficulty(row))) +
            0.05 * location_score
        )
        
        final_scores.append(round(final, 4))

       

    events_df["training_feasibility"] = training_scores
    events_df["agent_confidence"] = confidence_scores
    events_df["compatibility_score"] = final_scores

    # --------------------------------------------------
    # 6️⃣ XAI layer
    # --------------------------------------------------
    events_df["why_recommended"] = events_df.apply(
        build_why_recommended_v2,
        axis=1
    )


    events_df["readiness"] = events_df["compatibility_score"].apply(
        compute_readiness
    )

    # --------------------------------------------------
    # 7️⃣ Return top-K results
    # --------------------------------------------------
    events_df = filter_by_time_horizon(events_df, time_horizon)
    if events_df.empty:
        return pd.DataFrame({"message": ["No events in selected time range"]})
    
    top_events = (
        events_df
        .sort_values("compatibility_score", ascending=False)
        .head(top_k)
    )

    cols = [
        "event_name",
        "sport",
        "league",
        "date",
        "location",
        "description",
        "compatibility_score",
        "training_feasibility",      
        "agent_confidence",
        "readiness",
        "why_recommended"
    ]

    return top_events[cols].reset_index(drop=True)





def recommend_events(name: str, top_k: int = 5):
    idx = find_athlete_index(name)
    if idx is None:
        return f"Athlete '{name}' not found."
    
    print("DEBUG matched athlete:", athlete_df.loc[idx, ["name", "Sport", "Sex"]].to_dict())

    athlete_row = athlete_df.loc[idx]

    # Embed
    athlete_vector = np.array(athlete_row["emb"]).reshape(1, -1)

    # Athlete sport & sex from original columns
    sport = athlete_row.get("Sport", None)
    sex = athlete_row.get("Sex", None)

    filtered = event_df.copy()

    # 1) Filter by sport if possible
    if pd.notnull(sport) and "sport" in filtered.columns:
        filtered = filtered[filtered["sport"] == sport]

    # 2) Filter by sex if possible
    if pd.notnull(sex) and "event_sex" in filtered.columns:
        filtered = filtered[filtered["event_sex"] == sex]

    # Fallbacks if filter is too strict
    if filtered.empty and pd.notnull(sport) and "sport" in event_df.columns:
        filtered = event_df[event_df["sport"] == sport]

    if filtered.empty:
        filtered = event_df.copy()

    # Build matrix for filtered events
    event_matrix = np.vstack(filtered["emb"].to_list())
    sims = cosine_similarity(athlete_vector, event_matrix)[0]

    top_indices = sims.argsort()[::-1][:top_k]
    top_events = filtered.iloc[top_indices].copy()
    top_events["similarity"] = sims[top_indices]

    return top_events[["sport", "event", "event_sex", "similarity"]].reset_index(drop=True)


if __name__ == "__main__":
    print(recommend_events("Usain St. Leo Bolt", top_k=5))
