# SimuMatch

SimuMatch is an AI-powered system designed to match athletes to sports events based on their skill profile, performance attributes, and historical data patterns. The goal is to create a recommendation engine that can identify the most suitable events for athletes — useful for sports analytics, training guidance, scouting, and talent mapping.

---

## 🚀 Vision

Enable intelligent athlete-event matchmaking using machine learning + graph intelligence:

* Provide event recommendations based on athlete attributes
* Analyze athlete similarities & career paths
* Build a knowledge graph for explainability
* Create tools useful for coaches, analysts, and sports science research

---

## 📦 Data Used


We currently use **Olympic athlete & event datasets** (Kaggle / open-source) containing:

* Athlete ID, Name, Age
* Country / Team
* Sport & Event
* Physical data: Height, Weight

link: [https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results?resource=download](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)

We clean and process this data to generate **feature text** and embeddings.

### Matching Logic

We generate embeddings from text features like:

```
name + age + sex + country + sport + event
```

Then we compute **cosine similarity** between athletes & event vectors.

**Pipeline:**

1. Preprocess data
2. Create feature text for athletes & events
3. Generate embeddings (OpenAI model)
4. Save embeddings to disk
5. Build graph in Neo4j (athletes, sports, events relationships)
6. Recommend events by similarity search

---

## ⚙️ Local Setup Guide

Follow these steps to run SimuMatch on your machine.

### ✅ Requirements

* Python 3.9+
* Neo4j Desktop / Aura DB
* Virtual environment
* Git

---

### 🛠️ Setup

#### 1. Clone the repo

```bash
git clone https://github.com/shauryayay/SimuMatch.git
cd SimuMatch
```

#### 2. Create & activate virtual env

```bash
python3 -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate      # windows
```

#### 3. Install dependencies
Important: install PyTorch separately first

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

```

#### 4. Add your `.env` file

Create `.env`:

```
OPENAI_API_KEY=your_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
GROQ_API_KEY=your_key

```
#### 5. Neo4j Desktop
Download: https://neo4j.com/download/ 

Steps:
1. Create a new DB
2. Start it
3. Note URI + user + password
4. Put them in .env
---


### How to Run SimuMatch
#### Ensure data folders exist
```bash
mkdir -p data/raw
mkdir -p data/processed
```
#### Note:
data/processed/ is intentionally not stored in Git.
Everyone generates their own embedded data locally
Upload the Olympic dataset CSV inside:

```bash
data/raw/
```
#### Run Preprocess Pipeline
```bash
python src/data_pipeline/preprocess.py
```
This will generate:
```bash
data/processed/clean_athletes.csv
```

#### Generate embeddings

```bash
python src/matching/vector_search.py
```
```bash
python src/matching/event_embeddings.py
```
Generated files:

```bash
data/processed/athletes_with_embeddings.csv
data/processed/events_with_embeddings.csv
data/athlete_vectors.npy
data/event_vectors.npy
```
#### Generate Real Event Embeddings
```bash
python src/matching/real_event_embeddings.py
```
Expected output:

Encoding real events...

✅ Saved processed events to data/real_events/real_events_with_embeddings.csv
✅ Saved embeddings to data/real_events/real_event_vectors.npy

Real event embedding pipeline complete!

#### Add API Key

Create .env file:
```bash
GROQ_API_KEY=enter_api_key_created_in_groq
```
#### Run Streamlit App
```bash
streamlit run streamlit_app/app.py
```
#### Build graph

```bash
python -m src.graph.build_graph
```

or

```bash
python src/graph/build_graph.py
```

#### Run event recommender

```python
from src.matching.match_engine import recommend_events
print(recommend_events("Usain Bolt"))
```

---

## 🎯 Output

Example Response:

```
Top recommended events for Usain Bolt:
- 100m Sprint
- 200m Sprint
- 4×100m Relay
```



