def build_why_recommended(row, primary_sports):
    sport = str(row.get("sport", "")).lower()
    reasons = []

    if "running" in primary_sports and sport in ["athletics", "marathon"]:
        reasons.append("Strong alignment with your running training.")

    if "cycling" in primary_sports and sport == "cycling":
        reasons.append("Matches your cycling endurance profile.")

    if "swimming" in primary_sports and sport == "swimming":
        reasons.append("Based on consistent swim training.")

    if len(primary_sports) >= 3 and sport == "triathlon":
        reasons.append("Your multi-sport training suits triathlon.")

    if not reasons:
        reasons.append("General endurance compatibility.")

    return " ".join(reasons)


def compute_readiness(score):
    if score >= 0.65:
        return "Competition Ready"
    elif score >= 0.45:
        return "Almost Ready – short training phase needed"
    else:
        return "Not Ready Yet – build more base fitness"
