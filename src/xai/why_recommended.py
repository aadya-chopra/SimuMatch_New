def build_why_recommended_v2(row):
    """
    Enhanced XAI explanation builder using multi-factor reasoning.
    """

    reasons = []

    sim = row.get("similarity", 0)
    feas = row.get("training_feasibility", 0)
    conf = row.get("agent_confidence_pct", 0)
    sport = str(row.get("sport", "")).lower()

    # --- similarity reasoning ---
    if sim > 0.65:
        reasons.append("High profile similarity match")
    elif sim > 0.45:
        reasons.append("Moderate profile similarity")

    # --- feasibility reasoning ---
    if feas > 0.7:
        reasons.append("Training timeline is realistic")
    elif feas < 0.4:
        reasons.append("Tight preparation window")

    # --- sport reasoning ---
    if sport in ["triathlon", "running", "cycling", "swimming"]:
        reasons.append("Matches your primary endurance sports")

    # --- confidence reasoning ---
    if conf > 75:
        reasons.append("Agent has high confidence in this recommendation")

    if not reasons:
        reasons.append("General endurance compatibility")

    return " | ".join(reasons)
