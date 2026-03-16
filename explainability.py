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

from data_visualizing import DataVisualizing
from feature_extraction import SpacyFeatureExtraction

class Explainability:
    
    @staticmethod
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
        """
        print(f"\n{'='*50}")
        print(f"SHAP EXPLAINABILITY: {model_name}")
        print(f"{'='*50}")
        
        X_train_array = np.array(X_train_df[embeddings_col_name].to_list())
        
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

    @staticmethod
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
        Generate LIME text explanations for a single trained model.
        """
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            print("⚠️  LIME not installed. Run: pip install lime")
            return
        
        print(f"\n{'='*50}")
        print(f"LIME TEXT EXPLAINABILITY: {model_name}")
        print(f"{'='*50}")
        
        skip_models = ['perceptron', 'support_vector_machine_classifier', 'gradient_boosting_classifier', 'x_gradient_boosting_classifier',
                    'decision_tree_classifier', 'logistic_regression', 'ridge_classifier', 'sgd_classifier']
        
        if model_name in skip_models:
            print(f"⚠️  Skipping LIME for {model_name} (too slow/unstable)")
            return
        
        def predict_proba_from_text(texts):
            embeddings = []
            for text in texts:
                temp_df = pd.DataFrame({text_col_name: [text]})
                spacy_fe = SpacyFeatureExtraction(temp_df, text_col_name)
                embedded_df = spacy_fe.sentence_embeddings_extraction(attach_to_df=True)
                embedding_col = f'{text_col_name} Embedding'
                embeddings.append(embedded_df.iloc[0][embedding_col])
            
            embeddings_array = np.array(embeddings)
            return ml_model.predict_proba(embeddings_array)
        
        explainer = LimeTextExplainer(class_names=['Non-Prediction', 'Prediction'])
        
        for idx in range(min(num_samples, len(X_train_df))):
            sentence = X_train_df.iloc[idx][text_col_name]
            print(f"\nExplaining sample {idx}: '{sentence[:80]}...'")
            
            try:
                exp = explainer.explain_instance(
                    sentence,
                    predict_proba_from_text,
                    num_features=num_features,
                    num_samples=50  
                )
                
                html_file = os.path.join(save_path, f'lime_text_{model_name}_sample{idx}.html')
                exp.save_to_file(html_file)
                print(f"✓ Saved HTML: lime_text_{model_name}_sample{idx}.html")
                
                print(f"  Top {num_features} influential words:")
                for word, weight in exp.as_list():
                    direction = "→ Prediction" if weight > 0 else "→ Non-Prediction"
                    print(f"    '{word}': {weight:.3f} {direction}")
            
            except Exception as e:
                print(f"❌ LIME failed for sample {idx}: {e}")
                continue
        
        print(f"✓ LIME text explanation complete for {model_name}")

    @staticmethod
    def add_to_lime_comparison(
        X_train_df: pd.DataFrame,
        text_col_name: str,
        ml_model,
        model_name: str,
        save_path: str,
        sample_idx: int = 0,
        num_features: int = 8,
        num_samples: int = 50
    ):
        """
        Generate and append a LIME text explanation to a combined comparison HTML file 
        for a single model.
        """
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            print("⚠️  LIME not installed. Run: pip install lime")
            return

        print(f"\n{'='*60}")
        print(f"ADDING TO LIME MODEL COMPARISON: {model_name}")
        print(f"{'='*60}")

        target_sentence = X_train_df.iloc[sample_idx][text_col_name]
        os.makedirs(save_path, exist_ok=True)
        comparison_path = os.path.join(save_path, 'lime_comparison_all_models.html')

        classifier = ml_model.classifer if hasattr(ml_model, 'classifer') else ml_model
        
        if not hasattr(classifier, 'predict_proba'):
            print(f"\t⚠️ Skipping {model_name} (no predict_proba)")
            return

        try:
            print(f"\t→ Generating LIME text explanation...")
            
            def predict_proba_from_text(texts, current_classifier=classifier):
                embeddings = []
                for text in texts:
                    temp_df = pd.DataFrame({text_col_name: [text]})
                    spacy_fe = SpacyFeatureExtraction(temp_df, text_col_name)
                    embedded_df = spacy_fe.sentence_embeddings_extraction(attach_to_df=True)
                    embedding_col = f'{text_col_name} Embedding'
                    embeddings.append(embedded_df.iloc[0][embedding_col])
                return current_classifier.predict_proba(np.array(embeddings))
            
            lime_explainer = LimeTextExplainer(class_names=['Non-Prediction', 'Prediction'])
            lime_exp = lime_explainer.explain_instance(
                target_sentence,
                predict_proba_from_text,
                num_features=num_features,
                num_samples=num_samples
            )
            
            probs = lime_exp.predict_proba
            print(f"\t    Non-Prediction: {probs[0]:.3f}, Prediction: {probs[1]:.3f}")
            print(f"\t✓ LIME explanation collected\n")
            
        except Exception as e:
            print(f"\t❌ LIME explanation failed: {e}\n")
            return

        # Prepare the HTML snippet for this specific model
        model_html_snippet = f"""
        <div class="model-section">
            <div class="model-title">{model_name}</div>
            <div class="probability">
                <strong>Probabilities:</strong>
                Non-Prediction: {probs[0]:.3f} | Prediction: {probs[1]:.3f}
            </div>
            {lime_exp.as_html()}
        </div>
        """

        # Check if the file already exists to either create the header or append to it
        if not os.path.exists(comparison_path):
            final_html = f"""
            <html>
            <head>
                <title>LIME Model Comparison</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .model-section {{ border: 1px solid #ccc; margin: 20px 0; padding: 15px; }}
                    .model-title {{ font-size: 18px; font-weight: bold; color: #333; }}
                    .sentence {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; }}
                    .probability {{ margin: 10px 0; }}
                </style>
            </head>
            <body>
                <h1>LIME Explanations Comparison</h1>
                <div class="sentence">
                    <strong>Sentence:</strong> {target_sentence}
                </div>
                {model_html_snippet}
            </body>
            </html>
            """
        else:
            with open(comparison_path, 'r') as f:
                existing_html = f.read()
            
            # Strip the closing tags so we can inject the new model section right before them
            existing_html = existing_html.replace('</body>', '').replace('</html>', '').strip()
            
            final_html = existing_html + model_html_snippet + "\n</body>\n</html>"

        # Save the updated HTML
        with open(comparison_path, 'w') as f:
            f.write(final_html)
            
        print(f"✓ Comparison HTML updated: {comparison_path}")