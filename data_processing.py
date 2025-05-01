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
    
    def convert_tags_entities_to_dataframe(keys_of_mappings: set, mappings: list[list]):
        """
        Convert extracted entities into a pandas DataFrame.
        
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
    