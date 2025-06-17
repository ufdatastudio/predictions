import re
import numpy as np
import pandas as pd

from pathlib import Path
from spacy import displacy
from sklearn.model_selection import train_test_split

class DataProcessing:
    """A class to preprocess data"""

    def concat_dfs(dfs: list[pd.DataFrame], axis: int = 0, ignore_index=True) -> pd.DataFrame:
        """Concatenate multiple DataFrames"""
        df = pd.concat(dfs, axis=axis, ignore_index=ignore_index)
        return df
    
    def shuffle_df(df: pd.DataFrame):
        """Shuffle the data"""
        df = df.sample(frac=1).reset_index(drop=True)
        return df
    
    def split_data(vectorized_features, prediction_labels: pd.DataFrame):
        """Split the data into training and testing sets
        
        Parameters:
        -----------
        vectorized_features: Matrix
            A DataFrame containing the vectorized features
        
        prediction_labels: `pd.DataFrame`
            A DataFrame containing the prediction labels

        Returns:
        --------
        tuple
            A tuple containing the training and testing sets for the vectorized predictions and prediction labels
        """

        X_train, X_test, y_train, y_test = train_test_split(vectorized_features, prediction_labels, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def join_predictions_with_labels(df: pd.DataFrame, true_labels: pd.Series, y_predictions: pd.Series, model) -> pd.DataFrame:
        """Join the predictions with the true labels DF
        
        Parameters:
        -----------
        df: `pd.DataFrame`
            A DataFrame containing the data

        true_labels: `pd.DataFrame`
            A DataFrame containing the true labels

        y_predictions: `pd.DataFrame`
            A DataFrame containing the predictions from the model

        model: `object`
            An instance of the model
        
        Returns:
        --------
        `pd.DataFrame`
            A DataFrame containing the sentence, labels, and predictions
        """
        assert len(true_labels) == len(y_predictions), "The length of the true labels and predictions must be the same"

        N = len(df) - len(y_predictions)
        test_df = df.loc[N:].copy()
        model_name = model.__name__()
        model_col_name = f"{model_name} Prediction"
        test_df.loc[:, model_col_name] = y_predictions.values

        # Move "Prediction Label" column to second to last position
        cols = list(test_df.columns)
        cols.remove("Prediction Label")
        cols.insert(-1, "Prediction Label")
        test_df = test_df[cols]
        
        return test_df
    
    def join_predictions_with_sentences(df: pd.DataFrame, y_predictions: pd.Series, model) -> pd.DataFrame:
        """Join the model predictions with the sentences. In this case, no true labels are provided
        
        Parameters:
        -----------
        df: `pd.DataFrame`
            A DataFrame containing the data

        y_predictions: `pd.DataFrame`
            A DataFrame containing the predictions from the model

        model: `object`
            An instance of the model
        
        Returns:
        --------
        `pd.DataFrame`
            A DataFrame containing the sentence, labels, and predictions
        """
        assert len(df) == len(y_predictions), "The length of the true labels and predictions must be the same"

        joint_df = df.copy()
        model_name = model.__name__()
        model_col_name = f"{model_name} Prediction"
        joint_df.loc[:, model_col_name] = y_predictions.values
        
        return joint_df
    
    def reformat_df_with_template_number(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Reformat the DataFrame with the template number
        
        Parameters:
        -----------
        df: `pd.DataFrame`
            A DataFrame containing the data
        
        col_name: `str`
            The column name to use for the template number
        
        Returns:
        --------
        `pd.DataFrame`
            A DataFrame with the template number in the specified column
        """
        
        template_numbers = []
        reformat_predictions = []
        indices_to_keep = []

        for idx, prediction in enumerate(df[col_name].values):
            first_word = prediction.split()[0]
            if first_word == "T0:":
                template_numbers.append(0)
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T1:":
                template_numbers.append(1)
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T2:":
                template_numbers.append(2)
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T3:":
                template_numbers.append(3)
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T4:":
                template_numbers.append(4)
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T5:":
                template_numbers.append(5)
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T6:":
                template_numbers.append(6)
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            else:
                continue
        
        new_df = df.iloc[indices_to_keep].copy()
        new_df[col_name] = reformat_predictions
        new_df['Template Number'] = template_numbers
        return new_df
    
    def df_to_list(df: pd.DataFrame, col: str) -> list:
        """Convert a DataFrame to a list
        
        Parameters:
        -----------
        df: `pd.DataFrame`
            A DataFrame containing the data
        
        col: `str`
            The column to convert to a list

        Returns:
        --------
        list
            A list containing the data
        """
        return df[col].values.tolist()

    def drop_df_columns(df: pd.DataFrame, columns: list):
        """Drop columns
        
        Parameters:
        -----------
        df: `pd.DataFrame`
            The DataFrame to drop columns from
        
        columns: `list` 
            The list of columns to drop from the DataFrame
        
        Returns:
        --------
        df: `pd.DataFrame`
            The DataFrame with the columns dropped
        """
        df = df.drop(columns=columns)
        return df

    def encode_tags_entities_df(df: pd.DataFrame, sentence_and_label_df: pd.DataFrame) -> pd.DataFrame:
        """Encode the tags or entities in the DataFrame. Use 1 for the presence of the tag or entity and 0 if NaN
        
        Parameters:
        -----------
        df: `pd.DataFrame`
            The DataFrame to encode

        sentence_and_label_df: `pd.DataFrame`
            The DataFrame containing the sentence and prediction labels

        Returns:
        --------
        encoded_df: `pd.DataFrame`
            The DataFrame with the tags or entities encoded
        """
        bool_df = df.notnull() # Convert the DataFrame to boolean where if presence, place True and NaN place False
        encoded_df = bool_df.astype('int') # Convert the boolean DataFrame to integer where True is 1 and False is 0
        updated_encoded_df = DataProcessing.include_sentence_and_label(encoded_df, sentence_and_label_df)

        return updated_encoded_df

    def array_to_df(data: np.array) -> pd.DataFrame:
        """Convert a numpy array to a DataFrame
        
        Parameters:
        -----------
        data: `np.array`
            An array containing the data to convert to a DataFrame

        Returns:
        --------
        `pd.Series`
            A Series containing the data
        """
        return pd.Series(data)
    
    def convert_tags_entities_to_dataframe(keys_of_mappings: set, mappings: list[list]) -> pd.DataFrame:
        """
        Convert extracted features (Part-of-Speech (POS) tags and Named Entities Recognition (NER)) into a pandas DataFrame.
        
        Parameters:
        -----------
        entities : list
            A list of entities extracted from documents.
        
        all_ner_entities : list
            A list of all unique NER (Named Entity Recognition) tags.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing entities organized by NER tags.
        """
        df_ner = pd.DataFrame(columns=list(keys_of_mappings))
        for i, document_mapping in enumerate(mappings):
            for text, label in document_mapping:
                df_ner.at[i, label] = text
        return df_ner
    
    def include_sentence_and_label(df_to_update: pd.DataFrame, sentence_and_label_df: pd.DataFrame) -> pd.DataFrame:
        """Include the sentence and prediction labels in the DataFrame"""

        if len(df_to_update) == len(sentence_and_label_df):
            df_to_update.insert(0, 'Base Sentence', sentence_and_label_df['Base Sentence'].values)
            df_to_update.insert(1, 'Sentence Label', sentence_and_label_df['Sentence Label'].values)
            return df_to_update
        else:
            print(f"Error: The lengths of df_to_update ({len(df_to_update)}) and sentence_and_label_df ({len(sentence_and_label_df)}) do not match.")
            return df_to_update
        
    def convert_to_df(data, mapping=None):
        """Convert data to a DataFrame or Series.

        Parameters:
        -----------
        data: `np.array`, `list`, `dict`, or `set`
            An array, list, dictionary, or set containing the data to convert to a DataFrame or Series.
        
        mapping: `dict`, `optional`
            A dictionary containing mappings for tags/entities.

        Returns:
        --------
        `pd.DataFrame` or `pd.Series`
            A DataFrame or Series containing the data.
        """
        if isinstance(data, np.ndarray):
            return DataProcessing.array_to_df(data)
        elif isinstance(data, set) and isinstance(mapping, list):
            return DataProcessing.convert_tags_entities_to_dataframe(data, mapping)
        else:
            raise ValueError("Invalid input: data must be a numpy array, dictionary, list, or set with a mapping.")
    
# Functions to disregard
    def patterns(nlp):
        # ruler = nlp.add_pipe("entity_ruler", before="ner")
        # patterns = [
        #     {"label": "DATE", "pattern": [{"TEXT": {"REGEX": "20\\d{2}/\\d{2}/\\d{2}"}}]},
        #     {"label": "DATE", "pattern": [{"TEXT": {"REGEX": "20\\d{2}"}}, {"TEXT": {"REGEX": "Q[1-4]"}}]},
        #     {"label": "GPE", "pattern": [{"TEXT": {"REGEX": "LOC"}}, {"TEXT": {"REGEX": "LOC_\d"}}]},
        #     {"label": "DATE", "pattern": [{"TEXT": {"REGEX": "\d{1,2} [a-zA-Z]+ \d{4}"}}]} # NOT WORKING for ex 15 November 2021
        # ]
        # ruler.add_patterns(patterns)
        pass

    def update_ner(label, text):
        """Updates the NER label based on provided rules."""
        if label == "LOC" or label.startswith("LOC_"):
            return "GPE"
        elif label == "FAC":
            return "PERSON"
        elif label == "WORK_OF_ART":
            if re.match(r"Q\d of \d{4}|Quarter \d|\d Q\d", text):
                return "DATE"
            elif text in ["Weather Underground", "The Weather Channel]"]:
                return "ORG"
            else:
                return label  # Keep original label if no match
        elif label == "CARDINAL":
            if re.match(r"\d{1,2} [a-zA-Z]+ \d{4}", text):
                return "DATE"
            elif re.match(r"-\d+°[CF]|-\d+°[CF] to -\d+°[CF]|\d+°[CF] to \d+°[CF]|\d+°[CF]", text):
                return "TEMPERATURE"
            else: return label
        else:
            return label
    
    def select_pattern(template_number: int) -> str:
        """Select the pattern to use based on the template number
        
        Parameters:
        -----------
        template_number: `int`
           The template number that corresponds to the pattern to use

        Returns:
        --------
        `str`
            A string containing the pattern to use
        """

        if template_number == 1:
            pattern = r"On \[(.*?)\], \[(.*?)\] (.*?) that the \[(.*?)\] at \[(.*?)\] \[(.*?)\] \[(.*?)\] by \[(.*?)\] in \[(.*?)\]."
            return pattern
        elif template_number == 2:
            pattern = r"In \[(.*?)\], \[(.*?)\] from \[(.*?)\], (.*?) that the \[(.*?)\] \[(.*?)\] \[(.*?)\] from \[(.*?)\] to \[(.*?)\] in \[(.*?)\]."
            return pattern
        elif template_number == 3:
            pattern = r"\[(.*?)\] (.*?) on \[(.*?)\] that the \[(.*?)\] in \[(.*?)\] \[(.*?)\] \[(.*?)\] by \[(.*?)\] in \[(.*?)\]."
            return pattern
        elif template_number == 4:
            pattern = r"According to a \[(.*?)\] from \[(.*?)\], on \[(.*?)\], the \[(.*?)\] \[(.*?)\] \[(.*?)\] beyond \[(.*?)\] in the timeframe of \[(.*?)\]."
            return pattern
        elif template_number == 5:
            pattern = r"In \[(.*?)\], the \[(.*?)\] in \[(.*?)\] \[(.*?)\] \[(.*?)\] by \[(.*?)\], as (.*?) by \[(.*?)\] on \[(.*?)\]."
            return pattern
        else:
            raise ValueError("The template number is not recognized.")
    
    def sentence_to_df(df: pd.DataFrame) -> pd.DataFrame:
        """Convert an example sentence to a DataFrame

        NOTE: This function is specific to the template used in the example sentence
        
        Parameters:
        -----------
        sentence: `str`
            A sentence containing the variables to extract

        Returns:
        --------
        `pd.DataFrame`
            A DataFrame containing the extracted variables
        """
        template_numbers = df['Template Number'].values
        base_sentences = df['Base Sentence'].values
        
        extracted_data = []

        for template_number, sentence in zip(template_numbers, base_sentences):
            print(sentence)
            pattern = DataProcessing.select_pattern(template_number)
            print(pattern)
            
            match = re.match(pattern, sentence)
            
            if match:
                p_t, p_p, p_a, p_o, p_v, p_s, p_m, p_f = match.groups()
                data = {
                    'p_p': p_p,
                    'p_o': p_o,
                    'p_t': p_t,
                    'p_f': p_f,
                    'p_a': p_a,
                    'p_s': p_s,
                    'p_m': p_m,
                    'p_v': p_v,
                    'p_l': None  # Assuming y_l is not used in this template
                }
                extracted_data.append(data)
            else:
                print(sentence)
                raise ValueError("The sentence does not match the expected template.")
        
        return pd.DataFrame(extracted_data)
    
    def visualize_spacy_doc(doc, save_dir: str = "../paper/exSentence.png"):

        from PIL import Image
        import io
        import cairosvg

        options = {"compact": True, 
                   "bg": "#09a3d5", 
                   "font": "Source Sans Pro"}
        svg = displacy.render(doc, style="dep", jupyter=False, options=options)

            # Convert SVG to PNG
        png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(png_data))
        img.save(save_dir)