from fastapi import FastAPI

app = FastAPI(title="SimuMatch API")

@app.get("/")
def root():
    return {"msg":"SimuMatch backend online"}

@app.get("/match/{athlete_id}")
def get_match(athlete_id: str):
    # TODO call ranking pipeline
    return {"athlete": athlete_id, "rec": ["event_1", "event_2"]}
