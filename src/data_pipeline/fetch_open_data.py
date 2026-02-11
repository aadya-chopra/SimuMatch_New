import os, requests, pandas as pd

def fetch_world_athletics():
    url = "https://raw.githubusercontent.com/open-data-athletics/athletes/main/athletes.csv"
    df = pd.read_csv(url)
    df.to_csv("data/raw/athletes.csv", index=False)
    print("Fetched athletes")

if __name__ == "__main__":
    fetch_world_athletics()
