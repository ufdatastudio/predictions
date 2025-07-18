import spacy
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm
from spacy import displacy
from collections import defaultdict
from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import RobertaTokenizer, RobertaForSequenceClassification

from data_processing import DataProcessing


class FeatureExtractionFactory(ABC):
    """An abstract base class to create feature extraction classes."""

    def __init__(self, df_to_vectorize: pd.DataFrame, col_name_to_vectorize: str = None, type_of_df: str = "Standard"):
        self.df_to_vectorize = df_to_vectorize
        self.col_name_to_vectorize = col_name_to_vectorize
        self.type_of_df = type_of_df
        self.vectorizer = None
    
    def __name__(self):
        return self.__class__.__name__
    
    def extract_text_to_vectorize(self):
        text_to_vectorize = DataProcessing.df_to_list(self.df_to_vectorize, self.col_name_to_vectorize, self.type_of_df)
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

    def word_feature_extraction(self, max_features: int):
        """Vectorize the predictions DataFrame using a TfidfVectorizer for word features
        
        Returns:
        scipy.sparse._csr.csr_matrix
            A sparse matrix containing the vectorized word features
        """

        self.vectorizer = TfidfVectorizer(max_features=max_features)
        text_to_vectorize = self.extract_text_to_vectorize()
        vectorized_features = self.vectorizer.fit_transform(text_to_vectorize)
        
        return vectorized_features
    
    def feature_scores(self, max_features: int) -> pd.DataFrame:
        """Get the TF-IDF scores for the predictions"""

        vectorized_predictions = self.word_feature_extraction(max_features)
        # Convert the TF-IDF matrix to a dense matrix for easy viewing
        dense_matrix = vectorized_predictions.todense()

        # Get the feature names (terms) learned by the vectorizer
        feature_names = self.vectorizer.get_feature_names_out()

        # Create a DataFrame to visualize the TF-IDF scores
        tfidf_df = pd.DataFrame(dense_matrix, columns=feature_names)
        
        # Add the actual sentences and prediction labels to the DataFrame
        sentence_label_tfidf_df = DataProcessing.include_sentence_and_label(tfidf_df, self.df_to_vectorize)

        return sentence_label_tfidf_df
    
class SpacyFeatureExtraction(FeatureExtractionFactory):
    """An extension of the abstract base class called FeatureExtractionFactory"""

    def __name__(self):
        return "Spacy Feature Extraction"
    
    def __init__(self, df_to_vectorize: pd.DataFrame, col_name_to_vectorize: str = None, type_of_df: str = "Standard"):
        super().__init__(df_to_vectorize, col_name_to_vectorize, type_of_df)
        self.nlp = spacy.load("en_core_web_lg")  # Load a SpaCy model with word vectors
    
    def update_features_count(self, label, label_counts):
        """
        Increment and return the count for a given label (NOUN, ORG) in this document. The purpose is so we can collect every feature,
        especially those features with same type (NOUN, ORG) instead of only one of them. For ex, NOUN_n corresponds to 
        n having that many number of NOUNs in at least one sentence. So, say sentence 7 has the maximum number of nouns (across 
        all sentences) to be three), then you’ll get NOUN_1, NOUN_2, NOUN_3.

        It's a helper function to extract_features().

        Parameters:
        -----------
        label : `str`
            The POS or NER tag
        label_counts : `dict`
            Dictionary mapping labels to their current count for this document
        
        Returns:
        --------
        int
            The updated count of how many times the label has been seen so far in this document;
            used as a positional suffix for column naming (e.g., 1 for NOUN_1, 2 for NOUN_2).
        """
        label_counts[label] += 1 # Increment the count for this label in this document.
        return label_counts[label]

    def extract_pos_features(self, disable_components: list, batch_size: int = 50, visualize: bool = False) -> tuple[list]:
        """
        Extract features (Part-of-Speech (POS) tags and Named Entities Recognition (NER)) using the provided SpaCy NLP model.

        Parameters:
        -----------
        disable_components : `list`
            A list of components to disable in the SpaCy pipeline.
        
        batch_size : `int`
            The batch size for processing the data.
          
        visualize : `bool`
            Show the entities using Spacy visualizations.

        Returns:
        --------
        tuple
            A tuple containing the POS tags, dict{POS : word}, NER tags, and dict{NER : word}.
        """
        # print(f"Pipeline: {self.nlp.pipe_names}")
        
        word_tag_mappings = []
        
        data = self.extract_text_to_vectorize()
        
        for doc_i, doc in tqdm(enumerate(self.nlp.pipe(data, disable=disable_components, batch_size=batch_size))):
            if doc_i <= 3:
                print(f"Spacy Doc ({doc_i}): ", doc)

                if visualize is True:
                    DataProcessing.visualize_spacy_doc(doc)

            """Extract POSs"""    
            words = []
            labels = []
            unique_labels = []
            lemmas = []
            dependencies = []
            is_stop_words = []
            pos_label_counts = defaultdict(int) # RESET for this doc!
            for token in doc:
                text = token.text # The original word text.
                label = token.pos_ # The simple UPOS part-of-speech tag.
                lemma = token.lemma_ # The base form of the word.
                dependency = token.dep_ # Syntactic dependency, i.e. the relation between tokens
                is_stop_word = token.is_stop
                new_count_for_label = self.update_features_count(label, pos_label_counts) # Update count
                unique_label = f"{label}_{new_count_for_label}" # Give label the new count (ie: noun_1, noun_2, etc)
                # doc_tags.append((text, label, unique_label, lemma, dependency, is_stop_word))
                words.append(text)
                labels.append(label)
                unique_labels.append(unique_label)
                lemmas.append(lemma)
                dependencies.append(dependency)
                is_stop_words.append(is_stop_word)                
            word_tag_mappings.append(words)
            word_tag_mappings.append(labels)
            word_tag_mappings.append(unique_labels)
            word_tag_mappings.append(lemmas)
            word_tag_mappings.append(dependencies)
            word_tag_mappings.append(is_stop_words)
            # if doc_i <= 2:
            #     print(word_tag_mappings)
            
        return word_tag_mappings
    
    def extract_ner_features(self, disable_components: list, batch_size: int = 50, visualize: bool = False) -> pd.DataFrame:
        """
        Extract features (Part-of-Speech (POS) tags and Named Entities Recognition (NER)) using the provided SpaCy NLP model.

        Parameters:
        -----------
        disable_components : `list`
            A list of components to disable in the SpaCy pipeline.
        
        batch_size : `int`
            The batch size for processing the data.
          
        visualize : `bool`
            Show the entities using Spacy visualizations.

        Returns:
        --------
        pd.DataFrame
            A dataframe containing the NER features: term, NER label, unique NER label, start character, and end character
        """
        # print(f"Pipeline: {self.nlp.pipe_names}")
        
        sentences = []
        words = []
        labels = []
        unique_labels = []
        start_chars = []
        end_chars = []

        ner_features_df = pd.DataFrame()
        data = self.extract_text_to_vectorize()
        ner_label_counts = defaultdict(int)

        for doc_i, doc in tqdm(enumerate(self.nlp.pipe(data, disable=disable_components, batch_size=batch_size))):
            if doc_i <= 3:
                print(f"Spacy Doc ({doc_i}): ", doc)

                if visualize is True:
                    DataProcessing.visualize_spacy_doc(doc)

            for ent in doc.ents:
                label = ent.label_
                text = ent.text
                start_char = ent.start_char
                end_char = ent.end_char
                new_count_for_label = self.update_features_count(label, ner_label_counts) # Update count
                unique_label = f"{label}_{new_count_for_label}" # Give label the new count (ie: person_1, person_2, etc)

                sentences.append(data[doc_i])
                words.append(text)
                labels.append(label)
                unique_labels.append(unique_label)
                start_chars.append(start_char)
                end_chars.append(end_char)

            # Add a free row with no entry for every new sentence
            sentences.append("")
            words.append("")
            labels.append("")
            unique_labels.append("")
            start_chars.append("")
            end_chars.append("")
        
        ner_features_df["Sentence"] = sentences
        ner_features_df["Term"] = words
        ner_features_df["NER Label"] = labels
        ner_features_df["Unique NER Label"] = unique_labels
        ner_features_df["Start Char"] = start_chars
        ner_features_df["End Char"] = end_chars
                
        return ner_features_df
    
    def extract_features(self, disable_components: list, batch_size: int = 50, visualize: bool = False) -> tuple[list]:
        """
        Extract features (Part-of-Speech (POS) tags and Named Entities Recognition (NER)) using the provided SpaCy NLP model.

        Parameters:
        -----------
        disable_components : `list`
            A list of components to disable in the SpaCy pipeline.
        
        batch_size : `int`
            The batch size for processing the data.
          
        visualize : `bool`
            Show the entities using Spacy visualizations.

        Returns:
        --------
        tuple
            A tuple containing the POS tags, dict{POS : word}, NER tags, and dict{NER : word}.
        """
        print(f"Pipeline: {self.nlp.pipe_names}")
        tags = []
        all_pos_tags = set()

        entities = []
        all_ner_tags = set()

        data = self.extract_text_to_vectorize()
        for doc_i, doc in tqdm(enumerate(self.nlp.pipe(data, disable=disable_components, batch_size=batch_size))):
            if doc_i <= 3:
                print(f"Spacy Doc ({doc_i}): ", doc)

                if visualize is True:
                    DataProcessing.visualize_spacy_doc(doc)

            """Extract POSs"""    
            doc_tags = []
            pos_label_counts = defaultdict(int) # RESET for this doc!
            for token in doc:
                label = token.pos_ # The simple UPOS part-of-speech tag.
                text = token.text # The original word text.
                lemma = token.lemma_ # The base form of the word.
                dependency = token.dep_ # Syntactic dependency, i.e. the relation between tokens
                # is_stop_word = token.is_stop
                # if doc_i <= 1:
                    # print(f" POS: {text}---{label}---{lemma}---{dependency}---{is_stop_word}")
                new_count_for_label = self.update_features_count(label, pos_label_counts) # Update count
                unique_label = f"{label}_{new_count_for_label}" # Give label the new count (ie: noun_1, noun_2, etc)
                doc_tags.append((text, unique_label))
                all_pos_tags.add(unique_label)
            tags.append(doc_tags)
            # if doc_i <= 1:
            #     print()
            
            """Extract NERs"""
            doc_entities = []
            ner_label_counts = defaultdict(int) # RESET for this doc!
            for ent in doc.ents:
                label = ent.label_
                text = ent.text
                # if doc_i <= 1:
                    # print(f" NER: {text}---{label}---{ent.start_char}---{ent.end_char}")
                new_count_for_label = self.update_features_count(label, ner_label_counts) # Update count
                unique_label = f"{label}_{new_count_for_label}" # Give label the new count (ie: person_1, person_2, etc)
                doc_entities.append((text, unique_label))
                all_ner_tags.add(unique_label)
            entities.append(doc_entities)
            # if doc_i <= 1:
            #     print()

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
        """Extract sentence (Doc) vector embeddings (sentence to numbers) using Spacy
        
        Returns:
        np.array(n_sentences, vector_dim=300)
            A np.array(n_sentences, vector_dim=300) containing the sentence vector embeddings
        """
        text_to_vectorize = self.extract_text_to_vectorize()
        sent_embeddings = []
        # count = 0
        for sentence in tqdm(text_to_vectorize[:3]):
            doc = self.nlp(sentence)
            # if count <= 2:
            #     print(f"Doc {count}: Tokens: {len(doc)}\n   Sentence: {doc}")
            #     count += 1
            sent_embeddings.append(doc.vector)            
        return np.array(sent_embeddings)
                
    def word_feature_scores(self):
        """Get the word vector embeddings for the predictions"""

        sentence_embeddings = self.word_feature_extraction()
        return sentence_embeddings
    
    def sentence_to_word_via_spacy(self):
        """Convert sentences to words"""
        
        words = []
        sentences = self.extract_text_to_vectorize()

        for sentence in tqdm(sentences):

            doc = self.nlp(sentence)
            for token in doc:
                # print(token.text)
                words.append(token.text)
            words.append(" ")
        return words
             
    def pre_sequence_labeling_coversion(self):
        
        words = self.sentence_to_word_via_spacy()
        
        words_df = pd.DataFrame(words, columns=['Word'])
        words_df['Word Label'] = np.where(words_df['Word'] == ' ', ' ', 'O')
        
        return words_df
    
    def split_words_in_sentence(self):
        """Convert sentences to split as words"""
        
        word_split_sentences = []
        sentences = self.extract_text_to_vectorize()

        for sentence in tqdm(sentences):
            words = []

            doc = self.nlp(sentence)
            for token in doc:
                # print(token.text)
                words.append(token.text)
            word_split_sentences.append(words)
        return word_split_sentences
    
class RobertaFeatureExtraction(FeatureExtractionFactory):
    """An extension of the abstract base class called FeatureExtractionFactory"""

    def __name__(self):
        return "Roberta Feature Extraction"
    
    def __init__(self):
        # super().__init__(df_to_vectorize, col_name_to_vectorize, type_of_df)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large-mnli')
        self.model = RobertaForSequenceClassification.from_pretrained('roberta-large-mnli')
    
    def extract_entailment_features(self, predictions: list, observations: list) -> pd.DataFrame:
        entailment_outcome = {}
        entailment_predictions = []
        entailment_observations = []
        entailment_labels = []

        self.model.eval()

        for prediction in tqdm(predictions[:3]):
            for observation in observations[:33]:

                # Tokenize the input pair and get the model's prediction
                with torch.no_grad():
                    inputs = self.tokenizer(prediction, observation, return_tensors='pt')
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # The model outputs logits for three classes. We take the one with the highest score.
                    predicted_class_id = torch.argmax(logits).item()

                # The model's config has a mapping from class ID to label
                # For roberta-large-mnli: 0 -> contradiction, 1 -> neutral, 2 -> entailment
                entailment_label = self.model.config.id2label[predicted_class_id]
                entailment_predictions.append(prediction)
                entailment_observations.append(observation)
                entailment_labels.append(entailment_label)
        
        # print(len(predictions), len(observations), len(entailment_labels))
        entailment_outcome["Prediction"] = entailment_predictions
        entailment_outcome["Observation"] = entailment_observations
        entailment_outcome["Entailment Label"] = entailment_labels
        entailment_df = pd.DataFrame(entailment_outcome)

        return entailment_df
