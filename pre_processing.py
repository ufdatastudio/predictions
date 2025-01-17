import pandas as pd

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