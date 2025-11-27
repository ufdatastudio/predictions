import numpy as np

from tqdm import tqdm

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from classification_models import SkLearnModelFactory

class EvaluationMetric:

    def eval_accuracy(self, y_true, y_prediction):
        return accuracy_score(y_true, y_prediction)

    def eval_precision(self, y_true, y_prediction):
        return precision_score(y_true, y_prediction)

    def eval_recall(self, y_true, y_prediction):
        return recall_score(y_true, y_prediction)

    def eval_f1_score(self, y_true, y_prediction):
        return f1_score(y_true, y_prediction)
    
    def custom_evaluation_metrics(self, y_true, y_prediction):
        """Evaluate the model using accuracy and precision"""

        accuracy = self.eval_accuracy(y_true, y_prediction)
        precision = self.eval_precision(y_true, y_prediction)
        recall = self.eval_recall(y_true, y_prediction)
        f1 = self.eval_f1_score(y_true, y_prediction)

        metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        return metrics_dict

    def eval_classification_report(self, y_true, y_prediction):
        # print(classification_report(y_true, y_prediction))

        return classification_report(y_true, y_prediction)
    
    def get_cohens_kappa(self, df, rater_1_col_name, rater_2_col_name):
        frequency_table = pd.crosstab(df[rater_1_col_name], df[rater_2_col_name])
        return cohens_kappa(frequency_table)
    
    def calculate_pairwise_cohens_kappa(self, df: pd.DataFrame, model_names: list, 
                                    model_type: str = 'Model', print_bool: bool = False) -> pd.DataFrame:
        """
        Calculate Cohen's Kappa for all unique pairs of models.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing model predictions
        model_names : list
            List of column names for models to compare
        model_type : str, optional
            Description for printing (e.g., 'ML', 'LLM'). Default is 'Model'.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with all pairwise Cohen's Kappa scores
        """
        from itertools import combinations
        
        cohens_kappa_scores = []
        
        if print_bool == True:
            print(f"\n{'='*60}")
            print(f"{model_type} Models - Cohen's Kappa Scores")
            print(f"{'='*60}")
        
        for model_1, model_2 in combinations(model_names, 2):
            scores = self.get_cohens_kappa(df, model_1, model_2)
            kappa_value = round(scores.kappa, 2)
            
            cohens_kappa_scores.append({
                'Model 1': model_1,
                'Model 2': model_2,
                "Cohen's Kappa": kappa_value
            })
            if print_bool == True:
                print(f"Cohen's Kappa ({model_1} x {model_2}): {kappa_value}")
    
        return pd.DataFrame(cohens_kappa_scores)

    def train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a model.
        
        Parameters:
        -----------
        model_name: `str`
            The name of the model to be trained and evaluated.
        X_train: `pd.DataFrame`
            The training features.
        y_train: `pd.Series`
            The training labels.
        X_test: `pd.DataFrame`
            The test features.
        y_test: `pd.Series`
            The test labels.
        
        Returns:
        --------
        dict
            A dictionary containing the model name and its evaluation metrics.
        pd.Series
            The predictions made by the model.
        """
        model = SkLearnModelFactory.select_model(model_name)
        model.train_model(X_train, y_train)
        predictions = model.predict(X_test)
        

        metrics = EvaluationMetric.custom_evaluation_metrics(y_true=y_test, y_prediction=predictions)
        metrics['Model'] = model.get_model_name()
        
        return metrics, predictions
    
    def get_cosine_similarity(prediction_embeddings: np.array, observation_embeddings: np.array) -> list:
        assert len(prediction_embeddings) == len(observation_embeddings)

        model_scores = []
        for i in tqdm(range(len(prediction_embeddings))):

            # make them (1 × vector_dim) for sklearn
            pred_sent_embedding_reshaped = prediction_embeddings[i].reshape(1, -1)
            obser_sent_embedding_reshaped = observation_embeddings[i].reshape(1, -1)
            cos_sim = cosine_similarity(pred_sent_embedding_reshaped, obser_sent_embedding_reshaped)[0, 0]
            model_scores.append(cos_sim)
        
        return model_scores
