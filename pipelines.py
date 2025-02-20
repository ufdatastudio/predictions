import pandas as pd

from abc import ABC, abstractmethod

from data_processing import DataProcessing
from clean_predictions import PredictionDataCleaner
from feature_extraction import TfidfFeatureExtraction
from text_generation_models import LlamaTextGenerationModel
from classification_models import PerceptronModel, EvaluationMetric


class PipelineFactory(ABC):
    """An abstract base class to create pipelines."""

    def __init__(self):
        super().__init__()

class BasePipeline(PipelineFactory):
    """An extension of the abstract base class called PipelineFactory"""

    def generate_predictions(self, text: str, label: int, domain: str) -> pd.DataFrame:
        """Generate a prediction or non-prediction (general sentence) given the text and label
        
        Parameters:
        -----------
        text: `str`
            The text to generate a prediction or non-prediction from
        
        label: `int`
            An int that should be either 1 (prediction) or 0 (non-prediction)
        
        domain: `str`
            The domain of the text, e.g., financial, weather, health, etc.
        
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the generated prediction or non-prediction with the label
        """

        # models: https://console.groq.com/docs/models

        # Constants for model names
        LLAMA3_70B_INSTRUCT = "llama-3.1-70b-versatile"
        LLAMA3_8B_INSTRUCT = "llama3.1-8b-instant"

        LLAMA3_70B_INSTRUCT = "llama-3.3-70b-versatile"
        DEFAULT_MODEL = LLAMA3_70B_INSTRUCT

        # Create an instance of the LlamaModel
        llama_model = LlamaTextGenerationModel(
            model_name=DEFAULT_MODEL,
            prompt_template=text,
            temperature=0.3, # Lower temperature for more deterministic output (so less random)
            top_p=0.9, # Lower top_p to focus on high-probability words
        )

        df_col_names = ['Base Sentence']
        # Use the model to generate a prediction prompt and return it as a DataFrame
        df = llama_model.completion(df_col_names, label, DEFAULT_MODEL, domain)

        return df

    def pre_process_data(self, df) -> pd.DataFrame:
        """Pre-process the predictions DataFrame by removing any empty rows"""
        df = DataProcessing.concat_dfs(df)
        df = DataProcessing.shuffle_df(df)
        return df
    
    def tfidf_features(self, df, feature_scores: bool = False) -> pd.DataFrame:
        """Vectorize the predictions DataFrame using a TfidfVectorizer"""
        tf_idf_feature_extractor = TfidfFeatureExtraction(df)
        tfidf_vectorized_features = tf_idf_feature_extractor.feature_extraction()
        
        if feature_scores:
            scores = tf_idf_feature_extractor.feature_scores()
            return tfidf_vectorized_features, scores

        return tfidf_vectorized_features

    def train_and_predict(self, classifier, X_train, y_train, X_test):
        """Split the data into training and testing sets
        
        Parameters:
        -----------
        classifier: `object`
            An instance of the selected model
            
        vectorized_features_df: `pd.DataFrame`
            A DataFrame containing the vectorized features
        
        prediction_labels: `pd.DataFrame`
            A DataFrame containing the prediction labels

        Returns:
        --------
        """

        classifier.train_model(X_train, y_train)
        y_prediction = classifier.predict(X_test)
        
        return y_prediction
    
    def evaluation_metrics(self, y_true, y_prediction, default_metrics: bool):
        """Evaluate the model using accuracy and precision"""

        eval_metric = EvaluationMetric()

        if default_metrics:
            metrics = eval_metric.eval_classification_report(y_true, y_prediction)
            print(metrics)
            return None

        accuracy = eval_metric.eval_accuracy(y_true, y_prediction)
        precision = eval_metric.eval_precision(y_true, y_prediction)
        recall = eval_metric.eval_recall(y_true, y_prediction)
        f1 = eval_metric.eval_f1_score(y_true, y_prediction)

        metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        return metrics_dict
