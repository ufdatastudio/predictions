import spacy

import pandas as pd

from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer

from data_processing import DataProcessing


class FeatureExtractionFactory(ABC):
    """An abstract base class to create feature extraction classes."""

    def __init__(self, df_to_vectorize: pd.DataFrame, col_name_to_vectorize: str):
        self.df_to_vectorize = df_to_vectorize
        self.col_name_to_vectorize = col_name_to_vectorize
        self.vectorizer = None
    
    def __name__(self):
        return self.__class__.__name__
    
    def extract_text_to_vectorize(self):
        text_to_vectorize = DataProcessing.df_to_list(self.df_to_vectorize, self.col_name_to_vectorize)
        return text_to_vectorize

    def word_feature_extraction(self):
        pass

    def sentence_feature_extraction(self):
        pass

    def feature_scores(self):
        pass

class TfidfFeatureExtraction(FeatureExtractionFactory):
    """An extension of the abstract base class called FeatureExtractionFactory"""

    def __name__(self):
        return "TF x IDF Feature Extraction"

    def word_feature_extraction(self):
        """Vectorize the predictions DataFrame using a TfidfVectorizer for word features
        
        Returns:
        scipy.sparse._csr.csr_matrix
            A sparse matrix containing the vectorized word features
        """

        self.vectorizer = TfidfVectorizer(max_features=100)
        text_to_vectorize = self.extract_text_to_vectorize()
        vectorized_features = self.vectorizer.fit_transform(text_to_vectorize)
        
        return vectorized_features
    
    def feature_scores(self) -> pd.DataFrame:
        """Get the TF-IDF scores for the predictions"""

        vectorized_predictions = self.word_feature_extraction()
        # Convert the TF-IDF matrix to a dense matrix for easy viewing
        dense_matrix = vectorized_predictions.todense()

        # Get the feature names (terms) learned by the vectorizer
        feature_names = self.vectorizer.get_feature_names_out()

        # Create a DataFrame to visualize the TF-IDF scores
        tfidf_df = pd.DataFrame(dense_matrix, columns=feature_names)   

        return tfidf_df
    
class SpacyFeatureExtraction(FeatureExtractionFactory):
    """An extension of the abstract base class called FeatureExtractionFactory"""

    def __name__(self):
        return "Spacy Feature Extraction"
    
    def word_feature_extraction(self):
        """Extract word vector embeddings using Spacy
        
        Returns:
        list
            A list containing the word vector embeddings
        """
        text_to_vectorize = self.extract_text_to_vectorize()
        word_embeddings = []
        nlp = spacy.load("en_core_web_sm")

        for sentence in text_to_vectorize:
            doc = nlp(sentence)
            for token in doc:
                word_embeddings.append(token.vector)
        
        return word_embeddings

    def sentence_feature_extraction(self):
        """Extract sentence vector embeddings using Spacy
        
        Returns:
        list
            A list containing the sentence vector embeddings
        """
        text_to_vectorize = self.extract_text_to_vectorize()
        sent_embeddings = []
        nlp = spacy.load("en_core_web_sm")

        for sentence in text_to_vectorize:
            doc = nlp(sentence)
            for sent in doc.sents:
                sent_embeddings.append(sent.vector)
        
        return sent_embeddings
    
    def feature_scores(self):
        pass
