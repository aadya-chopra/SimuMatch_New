import pandas as pd
import numpy as np

def generate_synthetic_evaluation_data(n=300):
    """
    Synthetic data that mimics SimuMatch logic:
    - similarity
    - volume
    - sport match
    """

    np.random.seed(42)

    similarity = np.random.uniform(0, 1, n)
    volume = np.random.uniform(0, 1, n)
    sport_match = np.random.choice([0, 1], size=n, p=[0.3, 0.7])

    # Ground truth rule (what "should" be a good fit)
    true_fit = (
        (similarity > 0.6) &
        (volume > 0.5) &
        (sport_match == 1)
    ).astype(int)

    # Model prediction score (soft)
    predicted_score = (
        0.6 * similarity +
        0.25 * volume +
        0.15 * sport_match +
        np.random.normal(0, 0.05, n)
    )

    predicted_score = np.clip(predicted_score, 0, 1)
    predicted_fit = (predicted_score >= 0.65).astype(int)

    return pd.DataFrame({
        "similarity": similarity,
        "volume": volume,
        "sport_match": sport_match,
        "true_fit": true_fit,
        "predicted_score": predicted_score,
        "predicted_fit": predicted_fit
    })


if __name__ == "__main__":
    df = generate_synthetic_evaluation_data()
    print(df.head())
