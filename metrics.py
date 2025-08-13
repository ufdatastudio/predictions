import numpy as np

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity

class Metrics:

    def get_cosine_similarity(prediction_embeddings: np.array, observation_embeddings: np.array) -> list:
        assert len(prediction_embeddings) == len(observation_embeddings)

        model_scores = []
        for i in tqdm(range(len(prediction_embeddings))):

            # make them (1 × vector_dim) for sklearn
            pred_sent_embedding_reshaped = prediction_embeddings[i].reshape(1, -1)
            obser_sent_embedding_reshaped = observation_embeddings[i].reshape(1, -1)
            cos_sim = cosine_similarity(pred_sent_embedding_reshaped, obser_sent_embedding_reshaped)[0, 0]
            model_scores.append(cos_sim)
        
        return model_scores

