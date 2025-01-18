import pandas as pd

from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractionFactory(ABC):
    """An abstract base class to create feature extraction classes."""

    def __init__(self, df_to_vectorize: pd.DataFrame):
        self.df_df_to_vectorize = df_to_vectorize
        self.vectorizer = None
    
    def __name__(self):
        return self.__class__.__name__
    
    @abstractmethod
    def feature_extraction(self):
        pass

    def feature_scores(self):
        pass

class TfidfFeatureExtraction(FeatureExtractionFactory):
    """An extension of the abstract base class called FeatureExtractionFactory"""

    def __name__(self):
        return "TF x IDF Feature Extraction"

    def feature_extraction(self):
        """Vectorize the predictions DataFrame using a TfidfVectorizer
        
        Returns:
        scipy.sparse._csr.csr_matrix
            A sparse matrix containing the vectorized predictions
        """

        self.vectorizer = TfidfVectorizer(max_features=100)
        col_to_vectorize = self.df_df_to_vectorize.columns[0]
        vectorized_features = self.vectorizer.fit_transform(self.df_df_to_vectorize[col_to_vectorize])
        
        return vectorized_features
    
    def feature_scores(self) -> pd.DataFrame:
        """Get the TF-IDF scores for the predictions"""

        vectorized_predictions = self.feature_extraction()
        # Convert the TF-IDF matrix to a dense matrix for easy viewing
        dense_matrix = vectorized_predictions.todense()

        # Get the feature names (terms) learned by the vectorizer
        feature_names = self.vectorizer.get_feature_names_out()

        # Create a DataFrame to visualize the TF-IDF scores
        tfidf_df = pd.DataFrame(dense_matrix, columns=feature_names)   

        return  tfidf_df