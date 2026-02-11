import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# load environment
load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

if not uri:
    raise ValueError("NEO4J_URI not found. Did you create .env?")

driver = GraphDatabase.driver(uri, auth=(user, password))

def get_driver():
    return driver
