def estimate_event_difficulty(event_row):
    """
    Rough difficulty score [0–1]
    """

    sport = event_row["sport"].lower()
    league = str(event_row.get("league", "")).lower()

    if "ironman" in league:
        return 0.9
    if "marathon" in league:
        return 0.8
    if "triathlon" in sport:
        return 0.75
    if "cycling" in sport:
        return 0.6
    if "swimming" in sport:
        return 0.5

    return 0.4
