"""Patient similarity search."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def find_similar_patients(state, database, k=5):
    sims = cosine_similarity(state.reshape(1, -1), database)
    return np.argsort(sims[0])[::-1][:k]
