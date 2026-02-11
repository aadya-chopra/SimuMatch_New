import pandas as pd
from src.graph.neo4j_connection import get_driver

def push_athletes():
    df = pd.read_csv("data/processed/athletes_clean.csv")

    driver = get_driver()

    with driver.session() as session:
        for idx, row in df.iterrows():

            athlete_id = f"athlete_{idx}"  # auto ID instead of looking for 'id'

            session.run(
                """
                MERGE (a:Athlete {id:$id})
                SET a.name=$name, a.country=$country, a.sport=$sport
                """,
                id=athlete_id,
                name=row.get("Name", "Unknown"),
                country=row.get("NOC", "Unknown"),
                sport=row.get("Sport", "Unknown")
            )

    print("Graph nodes inserted!")

if __name__ == "__main__":
    push_athletes()
