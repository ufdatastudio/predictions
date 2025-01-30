import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class DataProcessing:
    """A class to preprocess data"""

    def concat_dfs(dfs: list[pd.DataFrame]):
        """Concatenate multiple DataFrames"""
        df = pd.concat(dfs, ignore_index=True)
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
    
    def ex_sentence_to_df(sentence: str) -> pd.DataFrame:
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
        
        # Define the regex pattern to extract the variables from the sentence
        pattern = r"On \[(.*?)\], \[(.*?)\] predicts that the \[(.*?)\] at \[(.*?)\] \[(.*?)\] \[(.*?)\] by \[(.*?)\] in \[(.*?)\]"
        match = re.match(pattern, sentence)
        
        if match:
            y_t, y_p, y_a, y_o, y_v, y_s, y_m, y_f = match.groups()
            data = {
                'y_p': [y_p],
                'y_o': [y_o],
                'y_t': [y_t],
                'y_f': [y_f],
                'y_a': [y_a],
                'y_s': [y_s],
                'y_m': [y_m],
                'y_v': [y_v],
                'y_l': [None]  # Assuming y_l is not used in this template
            }
            return pd.DataFrame(data)
        else:
            raise ValueError("The sentence does not match the expected template.")
    
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