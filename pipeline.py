import pandas as pd

from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer


from clean_predictions import PredictionDataCleaner
from text_generation_models import LlamaTextGenerationModel


class PipelineFactory(ABC):
    """An abstract base class to create pipelines."""

class BasePipeline(PipelineFactory):
    """An extension of the abstract base class called PipelineFactory"""

    def generate_predictions(self, text: str, label: int) -> pd.DataFrame:
        """Generate a prediction or non-prediction (general sentence) given the text and label
        
        Parameters:
        -----------
        text: `str`
            The text to generate a prediction or non-prediction from
        
        label: `int`
            An int that should be either 1 (prediction) or 0 (non-prediction)
        
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the generated prediction or non-prediction with the label
        """

        # Constants for model names
        LLAMA3_70B_INSTRUCT = "llama-3.1-70b-versatile"
        LLAMA3_8B_INSTRUCT = "llama3.1-8b-instant"
        DEFAULT_MODEL = LLAMA3_70B_INSTRUCT

        # Create an instance of the LlamaModel
        llama_model = LlamaTextGenerationModel(
            model_name=DEFAULT_MODEL,
            prompt_template=text,
            temperature=0.3, # Lower temperature for more deterministic output (so less random)
            top_p=0.9, # # Lower top_p to focus on high-probability words
        )

        df_col_names = ['Base Predictions']
        # Use the model to generate a prediction prompt and return it as a DataFrame
        predictions_df = llama_model.completion(df_col_names, label)
        # Display the DataFrame
        return predictions_df
    
    def clean_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Clean the predictions DataFrame by removing any empty rows"""
        cleaner = PredictionDataCleaner(predictions_df)
        predictions_col = predictions_df.columns[0]

        cleaner.lower_case(predictions_col)
        cleaner.remove_html_and_urls(predictions_col)
        cleaner.remove_contractions(predictions_col)
        # cleaner.remove_non_alphabetical_characters(predictions_col) # May need to keep so we don't remove numbers, percentages, etc.
        cleaner.remove_extra_spaces(predictions_col)

        return cleaner.df

    def tfidf_vectorize_predictions(self, predictions_df: pd.DataFrame, visualize: bool = False) -> pd.DataFrame:
        """Vectorize the predictions DataFrame using a TfidfVectorizer"""
        vectorizer = TfidfVectorizer()
        predictions_col = predictions_df.columns[0]
        vectorized_predictions = vectorizer.fit_transform(predictions_df[predictions_col])

        if visualize:
            # Convert the TF-IDF matrix to a dense matrix for easy viewing
            dense_matrix = vectorized_predictions.todense()

            # Get the feature names (terms) learned by the vectorizer
            feature_names = vectorizer.get_feature_names_out()

            # Create a DataFrame to visualize the TF-IDF scores
            tfidf_df = pd.DataFrame(dense_matrix, columns=feature_names)

            # Print the shape of the complete TF-IDF DataFrame; # Display the first few rows of the DataFrame
            print(f"Shape {tfidf_df.shape}:\n   {tfidf_df.shape[0]} rows\n   {tfidf_df.shape[1]} unique words\n\nVisualize TF-IDF scores:\n{tfidf_df.head()}")

        return vectorized_predictions
