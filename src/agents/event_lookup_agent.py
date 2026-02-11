from src.agents.event_lookup_llm import llm_event_lookup
from src.agents.event_lookup_fallback import fallback_event_lookup


def event_lookup_agent(
    profile_row,
    primary_sports,
    gender=None,
    location=None,
    time_horizon="Any time",
    llm=None,
):
    print("DEBUG: event_lookup_agent called")
    print("DEBUG: primary_sports =", primary_sports)
    print("DEBUG: time_horizon =", time_horizon)

    # 1️⃣ Try LLM FIRST (preferred)
    llm_df = llm_event_lookup(
        primary_sports=primary_sports,
        gender=gender,
        location=location,
        time_horizon=time_horizon,
        llm=llm,
    )

    if llm_df is not None and not llm_df.empty:
        print("DEBUG: USING LLM EVENTS")
        return llm_df

    # 2️⃣ Fallback ONLY if LLM fails
    print("DEBUG: FALLING BACK TO API / CSV")
    return fallback_event_lookup(
        primary_sports=primary_sports,
        gender=gender,
    )
