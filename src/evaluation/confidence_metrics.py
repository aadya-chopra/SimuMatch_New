import numpy as np

def compute_accuracy(df):
    return (df["true_fit"] == df["predicted_fit"]).mean()


def compute_precision(df):
    tp = ((df["predicted_fit"] == 1) & (df["true_fit"] == 1)).sum()
    fp = ((df["predicted_fit"] == 1) & (df["true_fit"] == 0)).sum()
    return tp / (tp + fp + 1e-9)


def compute_agent_confidence(similarity, sport_match, training_feasibility, location_score):
    score = (
        0.4 * float(similarity) +
        0.3 * float(training_feasibility) +
        0.2 * float(sport_match) +
        0.1 * float(location_score)
    )

    # clamp safety
    score = max(0.0, min(1.0, score))

    return round(score, 3)   # ✅ returns 0–1 only




