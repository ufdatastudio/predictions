# Before this, run python3 create_combined_dataset.py to create dataset

import os
import sys
import shap
import joblib
import argparse
import matplotlib
matplotlib.use('Agg')  # Prevent GUI windows from opening

import numpy as np
import pandas as pd

from datetime import datetime

# Get the current working directory of the script
script_dir = os.getcwd()
# Add the parent directory to the system path
sys.path.append(os.path.join(script_dir, '../'))

from data_visualizing import DataPlotting, DataVisualizing
from feature_extraction import SpacyFeatureExtraction
from classification_models import SkLearnModelFactory

class Explainability:

    def explain_model(
        X_train_df: pd.DataFrame,
        embeddings_col_name: str,
        ml_model,
        model_name: str,
        save_path: str,
        include_version: bool = False
        ):
        """
        Generate SHAP explanations for a trained model.
        
        Parameters
        ----------
        X_train_df : pd.DataFrame
            Training features dataframe
        embeddings_col_name : str
            Name of embeddings column
        ml_model : SkLearnModelFactory
            Trained model instance
        model_name : str
            Name of the model (used in filenames)
        save_path : str
            Directory path to save SHAP plots
        include_version : bool, default False
            If True, appends version suffix to filenames
        
        Notes
        -----
        Uses TreeExplainer for tree-based models (fast),
        LinearExplainer for linear models (fast),
        KernelExplainer for others (slow - skipped by default).
        
        Tree-based: random_forest, gradient_boosting, decision_tree, xgboost
        Linear: logistic_regression, ridge_classifier, sgd_classifier
        Skipped: perceptron, support_vector_machine (too slow)
        """
        
        print(f"\n{'='*50}")
        print(f"SHAP EXPLAINABILITY: {model_name}")
        print(f"{'='*50}")
        
        # Extract embeddings as numpy array
        X_train_array = np.array(X_train_df[embeddings_col_name].to_list())
        
        # Select explainer based on model type
        tree_based = ['random_forest_classifier', 'decision_tree_classifier']
        linear_based = ['logistic_regression', 'ridge_classifier', 'sgd_classifier']
        skip_models = ['perceptron', 'support_vector_machine_classifier', 'gradient_boosting_classifier', 'x_gradient_boosting_classifier']
        
        if model_name in skip_models:
            print(f"⚠️  Skipping SHAP for {model_name} (KernelExplainer too slow)")
            return
        
        try:
            if model_name in tree_based:
                print(f"Using TreeExplainer for {model_name}")
                explainer = shap.TreeExplainer(ml_model.classifer)
            elif model_name in linear_based:
                print(f"Using LinearExplainer for {model_name}")
                explainer = shap.LinearExplainer(ml_model.classifer, X_train_array)
            else:
                print(f"⚠️  Unknown model type for {model_name}, trying default Explainer")
                explainer = shap.Explainer(ml_model.classifer, X_train_array)
            
            shap_values = explainer(X_train_array)
            
            # Save all 4 plot types
            for plot_type in ['waterfall', 'beeswarm', 'bar', 'heatmap']:
                print(f"Generating {plot_type} plot...")
                DataVisualizing.get_shap_plot(
                    shap_values=shap_values,
                    plot_type=plot_type,
                    sample_idx=0,
                    model_name=model_name,
                    save_path=save_path,
                    include_version=include_version
                )
                print(f"✓ Saved: shap_{plot_type}_{model_name}.png")
        
        except Exception as e:
            print(f"❌ SHAP failed for {model_name}: {e}")
            print("Skipping SHAP for this model.")



    def explain_text_with_lime(
        X_train_df: pd.DataFrame,
        text_col_name: str,
        embeddings_col_name: str,
        ml_model,
        model_name: str,
        save_path: str,
        num_samples: int = 1,
        num_features: int = 7
    ):
        """
        Generate LIME text explanations for a trained model.
        
        Parameters
        ----------
        X_train_df : pd.DataFrame
            Training features dataframe with original text
        text_col_name : str
            Name of column containing original text
        embeddings_col_name : str
            Name of embeddings column
        ml_model : SkLearnModelFactory
            Trained model instance
        model_name : str
            Name of the model (used in filenames)
        save_path : str
            Directory path to save LIME visualizations
        num_samples : int, default 3
            Number of training samples to explain
        num_features : int, default 10
            Number of top features (words) to show in explanation
        
        Notes
        -----
        LIME perturbs text at the word level to understand which words
        most influence the prediction. Unlike SHAP on embeddings, this
        shows which actual words in the sentence matter.
        """
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            print("⚠️  LIME not installed. Run: pip install lime")
            return
        
        print(f"\n{'='*50}")
        print(f"LIME TEXT EXPLAINABILITY: {model_name}")
        print(f"{'='*50}")
        

        # Select explainer based on model type
        # tree_based = ['random_forest_classifier', ]
        # linear_based = ['logistic_regression', 
        skip_models = ['perceptron', 'support_vector_machine_classifier', 'gradient_boosting_classifier', 'x_gradient_boosting_classifier'
                    'decision_tree_classifier', 'logistic_regression', 'ridge_classifier', 'sgd_classifier']

        
        # skip_models = ['perceptron', 'support_vector_machine_classifier']
        if model_name in skip_models:
            print(f"⚠️  Skipping LIME for {model_name} (too slow/unstable)")
            return
        
        def predict_proba_from_text(texts):
            """Text -> Embedding -> Prediction probability"""
            embeddings = []
            for text in texts:
                # Create temporary df with correct column name
                temp_df = pd.DataFrame({text_col_name: [text]})
                # Extract embedding using SpaCy
                spacy_fe = SpacyFeatureExtraction(temp_df, text_col_name)
                embedded_df = spacy_fe.sentence_embeddings_extraction(attach_to_df=True)
                embedding_col = f'{text_col_name} Embedding'
                embeddings.append(embedded_df.iloc[0][embedding_col])
            
            embeddings_array = np.array(embeddings)
            return ml_model.predict_proba(embeddings_array)
        
        # Initialize LIME explainer
        explainer = LimeTextExplainer(class_names=['Non-Prediction', 'Prediction'])
        
        # Explain multiple samples
        for idx in range(min(num_samples, len(X_train_df))):
            sentence = X_train_df.iloc[idx][text_col_name]
            
            print(f"\nExplaining sample {idx}: '{sentence[:80]}...'")
            
            try:
                # Generate explanation
                exp = explainer.explain_instance(
                    sentence,
                    predict_proba_from_text,
                    num_features=num_features,
                    num_samples=3  # Number of perturbations (reduced for speed)
                )
                
                # Save as HTML
                html_file = os.path.join(
                    save_path, 
                    f'lime_text_{model_name}_sample{idx}.html'
                )
                exp.save_to_file(html_file)
                print(f"✓ Saved HTML: lime_text_{model_name}_sample{idx}.html")
                
                # Print top features
                print(f"  Top {num_features} influential words:")
                for word, weight in exp.as_list():
                    direction = "→ Prediction" if weight > 0 else "→ Non-Prediction"
                    print(f"    '{word}': {weight:.3f} {direction}")
            
            except Exception as e:
                print(f"❌ LIME failed for sample {idx}: {e}")
                import traceback
                traceback.print_exc()  # Show full error for debugging
                continue
        
        print(f"✓ LIME text explanation complete for {model_name}")