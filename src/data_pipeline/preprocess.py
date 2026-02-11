import pandas as pd
import os

def preprocess_athlete_events(
    input_path="data/raw/athlete_events.csv",
    output_path="data/processed/clean_athletes.csv"
):
    print("Loading raw data...")
    df = pd.read_csv(input_path)

    # -----------------------------------
    # 1. Normalize column names
    # -----------------------------------
    df.columns = [c.strip().lower() for c in df.columns]

    # Kaggle columns → standardized names
    rename_map = {
        "id": "id",
        "name": "name",
        "sex": "sex",
        "age": "age",
        "height": "height",
        "weight": "weight",
        "team": "team",
        "noc": "noc",
        "games": "games",
        "year": "year",
        "season": "season",
        "city": "city",
        "sport": "sport",
        "event": "event",
        "medal": "medal"
    }
    df = df.rename(columns=rename_map)

    # -----------------------------------
    # 2. Drop rows with no name
    # -----------------------------------
    df = df.dropna(subset=["name"])

    # Clean name capitalization
    df["name"] = df["name"].str.title()

    # -----------------------------------
    # 3. Fix "sex" column values (M/F)
    # -----------------------------------
    df["sex"] = df["sex"].str.upper().replace({
        "W": "F", "FEMALE": "F", "MALE": "M"
    })

    # -----------------------------------
    # 4. Remove duplicates
    # Keep first occurrence per athlete ID
    # -----------------------------------
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])
    else:
        # fallback: dedupe by name
        df = df.drop_duplicates(subset=["name"])

    # -----------------------------------
    # 5. Ensure numeric columns are clean
    # -----------------------------------
    for col in ["age", "height", "weight", "year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------------
    # 6. Handle missing sport/event gracefully
    # -----------------------------------
    df["sport"] = df["sport"].fillna("Unknown Sport")
    df["event"] = df["event"].fillna("Unknown Event")

    # -----------------------------------
    # 7. Prepare folders & save clean dataset
    # -----------------------------------
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved → {output_path}")

    return df


if __name__ == "__main__":
    preprocess_athlete_events()
