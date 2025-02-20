import spacy

import numpy as np
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

    def word_feature_extraction(self, max_features: int = 100):
        """Vectorize the predictions DataFrame using a TfidfVectorizer for word features
        
        Returns:
        scipy.sparse._csr.csr_matrix
            A sparse matrix containing the vectorized word features
        """

        self.vectorizer = TfidfVectorizer(max_features=max_features)
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
    
    def __init__(self, df_to_vectorize: pd.DataFrame, col_name_to_vectorize: str):
        super().__init__(df_to_vectorize, col_name_to_vectorize)
        self.nlp = spacy.load("en_core_web_md")  # Load a SpaCy model with word vectors
    
    def extract_entities(self, data: pd.Series, disable_components: list, batch_size: int = 50):
        """
        Extract entities using the provided SpaCy NLP model.

        Parameters:
        -----------
        data : `pd.Series`
            A Series containing textual data for entity extraction.

        disable_components : `list`
            A list of components to disable in the SpaCy pipeline.
        
        batch_size : `int`
            The batch size for processing the data.

        Returns:
        --------
        tuple
            A tuple containing the POS tags, POS to word mappings, NER tags, and NER to word mappings.
        """
        tags = []
        all_pos_tags = set()

        entities = []
        all_ner_tags = set()

        label_counts = {}

        for doc in self.nlp.pipe(data, disable=disable_components, batch_size=batch_size):
            doc_tags = []
            for token in doc:
                doc_tags.append((token.text, token.pos_))
                all_pos_tags.add(token.pos_)
            tags.append(doc_tags)

            doc_entities = []
            for ent in doc.ents:
                label = ent.label_
                text = ent.text
                # updated_label = DataProcessing.update_ner(label, text)  # update the label
                
                count_key = f"{label}_{doc}"
                if count_key in label_counts:
                    label_counts[count_key] += 1
                else:
                    label_counts[count_key] = 1
                unique_label = f"{label}_{label_counts[count_key]}"

                doc_entities.append((text, unique_label))  # changed label to updated_label
                all_ner_tags.add(unique_label)

            entities.append(doc_entities)

        return all_pos_tags, tags, all_ner_tags, entities
   
    def word_feature_extraction(self):
        """Extract word vector embeddings using Spacy
        
        Returns:
        list
            A list containing the word vector embeddings
        """
        sentences = self.extract_text_to_vectorize()
        word_features = []

        for sentence in sentences:
                doc = self.nlp(sentence)
                vectors = [token.vector for token in doc if not token.is_stop and not token.is_punct and token.has_vector]
                if vectors:
                    mean_vector = np.mean(vectors, axis=0)
                else:
                    mean_vector = np.zeros((self.nlp.meta['vectors']['width'],), dtype=float)
                word_features.append(mean_vector)
            
        return np.array(word_features)  # Ensuring it returns a 2D array with consistent dimensions


    def sentence_feature_extraction(self):
        """Extract sentence vector embeddings using Spacy
        
        Returns:
        list
            A list containing the sentence vector embeddings
        """
        text_to_vectorize = self.extract_text_to_vectorize()
        sent_embeddings = []

        for sentence in text_to_vectorize:
            doc = self.nlp(sentence)
            for sent in doc.sents:
                sent_embeddings.append(sent.vector)
        
        return sent_embeddings
    
    def word_feature_scores(self):
        """Get the word vector embeddings for the predictions"""

        sentence_embeddings = self.word_feature_extraction()
        return sentence_embeddings
