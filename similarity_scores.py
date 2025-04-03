import spacy

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

class SimilarityScores:
    
    def compute_similarity(self, prediction_observation_mapping: dict):
        """Compute the similarity between the prediction and observation sentences
        
        
        Parameters:
        -----------
        prediction_observation_mapping : `dict`
            A dictionary mapping prediction sentences to observation sentences.
            
        
        Returns:
        --------
        list
            A list of similarity scores between the prediction and observation sentences.
        """
        load_nlp_model = spacy.load("en_core_web_md")

        similarity_scores = []
        for prediction, observation in prediction_observation_mapping.items():
            pred_doc = load_nlp_model(prediction)
            obs_doc = load_nlp_model(observation)
            similarity = pred_doc.similarity(obs_doc)
            similarity_scores.append(similarity)
        
        return similarity_scores