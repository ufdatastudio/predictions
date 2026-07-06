import spacy
import torch

import numpy as np
import pandas as pd

from tqdm import tqdm
from spacy import displacy
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Union

from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, BertModel

from data_processing import DataProcessing
from data_visualizing import DataVisualizing


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

    def sentence_embeddings_extraction(self):
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

    def __init__(self, df_to_vectorize: pd.DataFrame, col_name_to_vectorize: str = None, type_of_df: str = "Standard", embedding_model_name: str = "spacy_large"):
        super().__init__(df_to_vectorize, col_name_to_vectorize, type_of_df)
        self.embedding_model_name = embedding_model_name
        self.embedding_models = {
            'spacy_small': "en_core_web_sm",
            'spacy_medium': "en_core_web_md",
            'spacy_large': "en_core_web_lg",
            'spacy_transformer': "en_core_web_trf",
        }
        self.nlp = spacy.load(self.embedding_models[self.embedding_model_name])

    def update_features_count_old(self, label, label_counts):
        """
        Increment and return the count for a given label (NOUN, ORG) in this document. The purpose is so we can collect every feature,
        especially those features with same type (NOUN, ORG) instead of only one of them. For ex, NOUN_n corresponds to 
        n having that many number of NOUNs in at least one sentence. So, say sentence 7 has the maximum number of nouns (across 
        all sentences) to be three), then you'll get NOUN_1, NOUN_2, NOUN_3.

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

    def extract_pos_features_old(self, disable_components: list, batch_size: int = 50, visualize: bool = False) -> tuple[list]:
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
        sentences = []
        words = []
        labels = []
        unique_labels = []
        lemmas = []
        dependencies = []
        is_stop_words = []
        pos_features_df = pd.DataFrame()
        data = self.extract_text_to_vectorize()
        pos_label_counts = defaultdict(int) # RESET for this doc!

        for doc_i, doc in tqdm(enumerate(self.nlp.pipe(data, disable=disable_components, batch_size=batch_size))):
            if doc_i <= 3:
                print(f"Spacy Doc ({doc_i}): ", doc)

                if visualize is True:
                    DataProcessing.visualize_spacy_doc(doc)  

            for token in doc:
                text = token.text # The original word text.
                label = token.pos_ # The simple UPOS part-of-speech tag.
                lemma = token.lemma_ # The base form of the word.
                dependency = token.dep_ # Syntactic dependency, i.e. the relation between tokens
                is_stop_word = token.is_stop
                new_count_for_label = self.update_features_count(label, pos_label_counts) # Update count
                unique_label = f"{label}_{new_count_for_label}" # Give label the new count (ie: noun_1, noun_2, etc)
                sentences.append(doc)
                words.append(text)
                labels.append(label)
                unique_labels.append(unique_label)
                lemmas.append(lemma)
                dependencies.append(dependency)
                is_stop_words.append(is_stop_word)  
            # Add a free row with no entry for every new sentence
            sentences.append("")
            words.append("")
            labels.append("")
            unique_labels.append("")
            lemmas.append(lemma)
            dependencies.append(dependency)
            is_stop_words.append(is_stop_word)  

        pos_features_df["Sentence"] = sentences
        pos_features_df["Term"] = words
        pos_features_df["POS Label"] = labels
        pos_features_df["Unique POS Label"] = unique_labels
        pos_features_df["Lemmas"] = lemmas
        pos_features_df["Dependencies"] = dependencies
        pos_features_df["Stop Word"] = is_stop_words
        return pos_features_df

    def extract_ner_features_old(self, disable_components: list, batch_size: int = 50, visualize: bool = False) -> pd.DataFrame:
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

                sentences.append(doc)
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

    def extract_features_old(self, disable_components: list, batch_size: int = 50, visualize: bool = False) -> tuple[list]:
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

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def update_features_count(self, label: str, label_counts: dict[int]) -> int:
        """
        Parameters
        ----------
        label : str
            The POS or NER tag.
        label_counts : dict[int]
            Map from tag to current counter.

        Returns
        -------
        int
            Updated counter for the tag.
        """
        label_counts[label] += 1
        return label_counts[label]


    # ------------------------------------------------------------------
    # Morphological keys that should be split out into separate columns
    # ------------------------------------------------------------------
    _morph_keys = [
        "Case", "Number", "Person", "PronType", "Tense", "VerbForm",
        "Mood", "Voice", "Aspect", "Gender", "Definite", "Degree",
        "SubCat",
        "Animacy", "Acc", "Gen", "Loc", "Ins", "Parat", "Prep"
    ]

    # ------------------------------------------------------------------
    # POS+Morph extraction
    # ------------------------------------------------------------------
    def extract_pos_features(
        self,
        disable_components: list,
        batch_size: int = 50,
        visualize: bool = False,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        disable_components : list
            Components to disable in the SpaCy pipeline.
        batch_size : int, default 50
            How many documents to process per batch.
        visualize : bool, default False
            Show the first few docs if True.

        Notes
        -----
        The columns are ordered the same way they appear in SpaCy's
        token‑attribute reference (text → lemma_ → pos_ → tag_ → dep_
        → shape_ → is_alpha → is_stop → morph).  Every morphological
        feature defined in ``_morph_keys`` gets its own column.

        Returns
        -------
        pd.DataFrame
            One row per token.
        """
        print("\n" + "="*50)
        print("EXTRACTING POS FEATURES")
        print("="*50)
        # 1. Prepare containers -------------------------------------------------
        sentences, words, lemmas = [], [], []
        pos_labels, detailed_pos_labels = [], []
        dependencies, shape, is_alpha, is_stop_words = [], [], [], []

        # Morphological columns – one list per feature
        morph_dict = {k: [] for k in self._morph_keys}

        unique_pos_labels, unique_pos_detailed_labels = [], []

        data = self.extract_text_to_vectorize()
        pos_label_counts = defaultdict(int)
        detailed_pos_label_counts = defaultdict(int)

        # 2. Iterate over documents ---------------------------------------------
        for doc_i, doc in tqdm(
            enumerate(
                self.nlp.pipe(
                    data,
                    disable=disable_components,
                    batch_size=batch_size,
                )
            ),
            total=len(data),
            desc="Extracting POS Features",
            unit="doc",
            miniters=1
        ):
            if doc_i <= 3 and visualize:
                # DataProcessing.visualize_spacy_doc(doc)
                print(f"\n\t####### Sentence ({doc_i}): {doc} #######")
                DataVisualizing.spacy_pos_dep(doc, self.nlp)

            for token in doc:
                # baseline token attributes
                sentences.append(doc.text)
                words.append(token.text)
                lemmas.append(token.lemma_)
                pos_labels.append(token.pos_)
                detailed_pos_labels.append(token.tag_)
                dependencies.append(token.dep_)
                shape.append(token.shape_)
                is_alpha.append(token.is_alpha)
                is_stop_words.append(token.is_stop)

                # unique suffixes
                unique_pos_labels.append(
                    f"{token.pos_}_{self.update_features_count(token.pos_, pos_label_counts)}"
                )
                unique_pos_detailed_labels.append(
                    f"{token.tag_}_{self.update_features_count(token.tag_, detailed_pos_label_counts)}"
                )

                # --- Morphological features ------------------------------------
                for key in self._morph_keys:
                    morph_dict[key].append(token.morph.get(key) or "")

            # Optional empty row to separate documents
            sentences.append("")
            words.append("")
            lemmas.append("")
            pos_labels.append("")
            detailed_pos_labels.append("")
            dependencies.append("")
            shape.append("")
            is_alpha.append(False)
            is_stop_words.append("")
            unique_pos_labels.append("")
            unique_pos_detailed_labels.append("")
            for key in self._morph_keys:
                morph_dict[key].append("")

        # 3. Assemble the dataframe --------------------------------------------
        df_dict = {
            "Sentence": sentences,
            "Term": words,
            "Lemma": lemmas,
            "POS Label": pos_labels,
            "Detailed POS Label": detailed_pos_labels,
            "Dependency": dependencies,
            "Shape": shape,
            "Is Alpha": is_alpha,
            "Stop Word": is_stop_words,
            "Unique POS Label": unique_pos_labels,
            "Unique Detailed POS Label": unique_pos_detailed_labels,
        }

        # Add all morph columns
        df_dict.update(morph_dict)

        print("\n" + "="*50)
        print("DONE EXTRACTING POS FEATURES")
        print("="*50)
        return pd.DataFrame(df_dict)

    # ------------------------------------------------------------------
    # NER‑only
    # ------------------------------------------------------------------
    def extract_ner_features(
        self,
        disable_components: list,
        batch_size: int = 50,
        visualize: bool = False,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        disable_components : list
            Components to disable in the spaCy pipeline.
        batch_size : int, default 50
            How many documents to process per pipeline batch.
        visualize : bool, default False
            If True, visualise the first few docs.

        Notes
        -----
        Returns a DataFrame containing one row per entity.

        Returns
        -------
        pd.DataFrame
            Columns: Sentence, Term, NER Label, Unique NER Label,
            Start Char, End Char.
        """
        print("\n" + "="*50)
        print("EXTRACTING NER FEATURES")
        print("="*50)

        sentences, words, labels, unique_labels = [], [], [], []
        start_chars, end_chars = [], []

        data = self.extract_text_to_vectorize()
        ner_label_counts = defaultdict(int)

        for doc_i, doc in tqdm(
            enumerate(
                self.nlp.pipe(
                    data,
                    disable=disable_components,
                    batch_size=batch_size,
                )
            ),
            total=len(data),
            desc="Extracting NER Features",
            unit="doc",
            miniters=1
        ):
            if doc_i <= 3 and visualize:
                # DataProcessing.visualize_spacy_doc(doc)
                print(f"\n\t####### Sentence ({doc_i}): {doc} #######")
                DataVisualizing.spacy_ner_ent(doc, self.nlp)

            for ent in doc.ents:
                sentences.append(doc.text)
                words.append(ent.text)
                labels.append(ent.label_)
                unique_labels.append(
                    f"{ent.label_}_{self.update_features_count(ent.label_, ner_label_counts)}"
                )
                start_chars.append(ent.start_char)
                end_chars.append(ent.end_char)

            # Empty row to mark end of current document
            sentences.append("")
            words.append("")
            labels.append("")
            unique_labels.append("")
            start_chars.append("")
            end_chars.append("")

        print("\n" + "="*50)
        print("DONE EXTRACTING NER FEATURES")
        print("="*50)
        return pd.DataFrame(
            {
                "Sentence": sentences,
                "Term": words,
                "NER Label": labels,
                "Unique NER Label": unique_labels,
                "Start Char": start_chars,
                "End Char": end_chars,
            }
        )

    # ------------------------------------------------------------------
    # Unified extractor
    # ------------------------------------------------------------------
    def extract_features(
        self,
        disable_components: list,
        batch_size: int = 50,
        visualize: bool = False,
        mode: str = "both",
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Parameters
        ----------
        disable_components : list
            Components to disable in the spaCy pipeline.
        batch_size : int, default 50
            How many documents to process per pipeline batch.
        visualize : bool, default False
            If True, visualise the first few docs.
        mode : {'pos', 'ner', 'both'}, default 'both'
            Which feature set to extract.

        Notes
        -----
        The method simply forwards to the appropriate helper(s).

        Returns
        -------
        pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]
            * ``'pos'`` → POS DataFrame
            * ``'ner'`` → NER DataFrame
            * ``'both'`` → (POS DataFrame, NER DataFrame)
        """
        mode = mode.lower()
        if mode == "pos":
            return self.extract_pos_features(disable_components, batch_size, visualize)

        if mode == "ner":
            return self.extract_ner_features(disable_components, batch_size, visualize)

        # both
        pos_df = self.extract_pos_features(disable_components, batch_size, visualize)
        ner_df = self.extract_ner_features(disable_components, batch_size, visualize)
        return pos_df, ner_df

    # ------------------------------------------------------------------
    # Legacy alias (not strictly NumPy‑docstring compliant)
    # ------------------------------------------------------------------
    def extract_pos_ner_features(self, *args, **kwargs):
        """Alias to preserve backward compatibility. Returns as (pos_df, ner_df)"""
        return self.extract_features(*args, **kwargs)

    def word_feature_extraction(self):
        """Extract word vector embeddings using Spacy
        Returns:
        list
            A list containing the word vector embeddings
        """
        sentences = self.extract_text_to_vectorize()
        word_features = []

        for i, sentence in enumerate(tqdm(sentences, desc="Extracting word features")):
            if i < 7:
                print(f"{i+1}. {sentence}")
            doc = self.nlp(sentence)
            vectors = [token.vector for token in doc if not token.is_stop and not token.is_punct and token.has_vector]
            if vectors:
                mean_vector = np.mean(vectors, axis=0)
            else:
                mean_vector = np.zeros((self.nlp.meta['vectors']['width'],), dtype=float)
            word_features.append(mean_vector)
            
        return np.array(word_features)  # Ensuring it returns a 2D array with consistent dimensions

    def word_embeddings_extraction(
        self, 
        tokenized_words_with_metadata_df: pd.DataFrame = None,
        reorder_cols: list[str] = ["Base Sentence", "Word", "Word Embedding", "Ground Truth"]
    ) -> pd.DataFrame:

        if tokenized_words_with_metadata_df is None:
            tokenized_words_with_metadata_df = self.split_words_in_sentence()

        rows = []

        for _, row in tqdm(tokenized_words_with_metadata_df.iterrows(), 
                        total=len(tokenized_words_with_metadata_df),
                        desc="Embedding words"):

            word = row["Word"]
            doc = self.nlp(word)

            if len(doc) > 0 and doc[0].has_vector:
                embedding = doc[0].vector
            else:
                embedding = np.zeros(self.nlp.meta["vectors"]["width"], dtype=float)

            row_dict = row.to_dict()
            row_dict["Word Embedding"] = embedding

            rows.append(row_dict)

        df = pd.DataFrame(rows)

        if reorder_cols is not None:
            existing_reorder_cols = [c for c in reorder_cols if c in df.columns]
            remaining_cols = [c for c in df.columns if c not in existing_reorder_cols]
            df = df[existing_reorder_cols + remaining_cols]

        return df

    def sentence_embeddings_extraction(self, attach_to_df: bool = True):
        """Extract sentence (Doc) vector embeddings using SpaCy"""
        text_to_vectorize = self.extract_text_to_vectorize()
        sentence_embeddings = []
        
        # Check if using transformer model
        is_transformer = self.embedding_model_name == 'spacy_transformer'
        
        # Debug: Check what components are in the pipeline
        print(f"Pipeline components: {self.nlp.pipe_names}")
        
        for idx, sentence in enumerate(tqdm(text_to_vectorize)):
            if not isinstance(sentence, float) and pd.notna(sentence):
                doc = self.nlp(sentence)
                
                if is_transformer:
                    try:
                        # Method 1: Try using doc._.trf_data
                        if hasattr(doc._, 'trf_data') and doc._.trf_data is not None:
                            tensors = doc._.trf_data.tensors
                            if len(tensors) > 0 and tensors[0].numel() > 0:
                                embedding = tensors[0].mean(dim=0).cpu().numpy()
                                if idx < 3:  # Debug first few
                                    print(f"Method 1 success - embedding shape: {embedding.shape}")
                            else:
                                raise ValueError("Empty tensors")
                        else:
                            raise ValueError("No trf_data")
                            
                    except Exception as e1:
                        # Method 2: Manual token averaging
                        try:
                            if idx < 3:
                                print(f"Method 1 failed ({e1}), trying Method 2: manual token vectors")
                            
                            token_vectors = []
                            for token in doc:
                                if token.has_vector and token.vector.shape[0] > 0:
                                    token_vectors.append(token.vector)
                            
                            if token_vectors:
                                embedding = np.mean(token_vectors, axis=0)
                                if idx < 3:
                                    print(f"Method 2 success - embedding shape: {embedding.shape}")
                            else:
                                raise ValueError("No token vectors available")
                                
                        except Exception as e2:
                            if idx < 3:
                                print(f"Method 2 also failed ({e2}), using zeros")
                            embedding = np.zeros(768)
                else:
                    # For non-transformer models
                    embedding = doc.vector
                    if embedding.shape[0] == 0:
                        if idx < 3:
                            print(f"WARNING: Empty embedding for: {sentence[:50]}")
                
                sentence_embeddings.append(embedding)
            else:
                # For invalid sentences
                if is_transformer:
                    sentence_embeddings.append(np.zeros(768))
                else:
                    expected_size = self.nlp.meta.get('vectors', {}).get('width', 300)
                    sentence_embeddings.append(np.zeros(expected_size))

        if attach_to_df:
            self.df_to_vectorize[f"{self.col_name_to_vectorize} Embedding"] = sentence_embeddings
            return self.df_to_vectorize
        else:
            clean_embeddings = [emb for emb in sentence_embeddings if emb is not None]
            return np.array(clean_embeddings)

    def word_feature_scores(self):
        """Get the word vector embeddings for the predictions"""

        sentence_embeddings = self.word_feature_extraction()
        return sentence_embeddings

    def pre_sequence_labeling_coversion(self):
        words = self.sentence_to_word_via_spacy()
        words_df = pd.DataFrame(words, columns=['Word'])
        words_df['Word Label'] = np.where(words_df['Word'] == ' ', ' ', 'O')
        return words_df

    def split_words_in_sentence(self, columns_to_keep: list[str] = None) -> pd.DataFrame:
        """
        Convert sentences into DataFrame where each row = one word, preserving selected columns
        """
        rows = []

        data = self.df_to_vectorize

        # Default: keep all columns
        if columns_to_keep is None:
            columns_to_keep = list(data.columns)

        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Tokenizing sentences"):
            sentence = row[self.col_name_to_vectorize]

            doc = self.nlp(sentence)
            sentence_tokens = []

            for token in doc:
                word = token.text
                sentence_tokens.append(word)

                row_dict = {
                    self.col_name_to_vectorize: sentence,
                    "Word": word
                }

                # Keep additional columns
                for col in columns_to_keep:
                    if col not in row_dict:
                        row_dict[col] = row[col]

                rows.append(row_dict)
        tokenized_words_with_metadata_df = pd.DataFrame(rows)

        return tokenized_words_with_metadata_df
    
class SentenceTransformerFeatureExtraction(FeatureExtractionFactory):
    """An extension of the abstract base class called FeatureExtractionFactory"""

    def __name__(self):
        return "SentenceTransformer Feature Extraction"

    def __init__(
        self,
        df_to_vectorize: pd.DataFrame,
        col_name_to_vectorize: str = None,
        type_of_df: str = "Standard",
        embedding_model_name: str = "st_mini_lm"
    ):
        super().__init__(df_to_vectorize, col_name_to_vectorize, type_of_df)

        self.embedding_models = {
            'st_mpnet_base': 'sentence-transformers/all-mpnet-base-v2',
            'st_distilroberta': 'sentence-transformers/all-distilroberta-v1',
            'st_minilm_l12': 'sentence-transformers/all-MiniLM-L12-v2',
            'st_minilm_l6': 'sentence-transformers/all-MiniLM-L6-v2',
        }

        self.embedding_model_name = embedding_model_name
        model_name = self.embedding_models[self.embedding_model_name]

        # Load model
        self.st = SentenceTransformer(model_name)

    def sentence_embeddings_extraction(self, attach_to_df: bool = True):
        """
        Extract sentence embeddings using SentenceTransformer

        Parameters
        ----------
        attach_to_df : bool, default True
            If True, attach embeddings to dataframe. Otherwise return numpy array

        Returns
        -------
        pd.DataFrame or np.ndarray
        """

        text_to_vectorize = self.extract_text_to_vectorize()

        # ✅ Core operation (from docs)
        embeddings = self.st.encode(text_to_vectorize, show_progress_bar=True)

        if attach_to_df:
            self.df_to_vectorize[f"{self.col_name_to_vectorize} Embedding"] = list(embeddings)
            return self.df_to_vectorize
        else:
            return embeddings