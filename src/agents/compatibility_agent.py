import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def dual_similarity_agent(
    athlete_vector,
    olympic_event_vectors,
    real_event_vectors
):
    sim_olympic = cosine_similarity(athlete_vector, olympic_event_vectors)[0]
    sim_real = cosine_similarity(athlete_vector, real_event_vectors)[0]

    return sim_olympic, sim_real
