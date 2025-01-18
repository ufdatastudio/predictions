import pandas as pd

from sklearn.model_selection import train_test_split

class PreProcessing:
    """A class to preprocess data"""

    def concat_dfs(dfs: list[pd.DataFrame]):
        """Concatenate multiple DataFrames"""
        df = pd.concat(dfs)
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