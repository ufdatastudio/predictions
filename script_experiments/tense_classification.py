# Tense Classification
import os
import sys
import ast
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List

# Add project modules to path
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, '../'))
from data_processing import DataProcessing
from feature_extraction import SpacyFeatureExtraction

def load_dataset(base_data_path, dataset_name):
    """
    Load dataset from file path.
    """
    print("\n" + "="*50)
    print("LOAD DATASET")
    print("="*50)
    
    data_path = os.path.join(base_data_path, dataset_name)
    
    print(f"Dataset path: {dataset_name}")
    # Assuming DataProcessing handles file not found errors internally
    df = DataProcessing.load_from_file(data_path, 'csv', sep=',')
    print(f"Shape: {df.shape}")
    print(f"\nPreview:\n{df.head(10)}\n")
    
    return df

def get_features_of_interest(df: pd.DataFrame, features_of_interest: list):
    sentence_pos_df = df.loc[:, ['Sentence', 'POS Label', 'Detailed POS Label']]
    filt_aux = (sentence_pos_df['POS Label'] == features_of_interest) & (sentence_pos_df['Detailed POS Label'] == 'MD')
    results_df = sentence_pos_df[filt_aux]
    results_df['Future Tense'] = 1
    print(f"Shape: {results_df.shape}")
    print(f"\nPreview:\n{results_df.head(10)}\n")
    return results_df


if __name__ == "__main__":
    """Usage
    # Default models
    python3 tense_classification.py
    """
        
    print("\n" + "="*50)
    print("PREDICTION PROPERTY EXTRACTION")
    print("="*50)

    # ============================================================
    # 1. CONFIGURATION & ARGS
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = DataProcessing.load_base_data_path(script_dir)
    default_dataset = os.path.join(base_data_path, 'tense_extraction/batch_2-from_df_pos_features-v2.csv')
    
    parser = argparse.ArgumentParser(description='Extract prediction properties from sentences using LLMs')
    parser.add_argument('--dataset', default=default_dataset, 
                       help='Path to dataset relative to base data directory.')
    # parser.add_argument('--aux_feature', default=True, 
    #                    help='Path to dataset relative to base data directory.')
    args = parser.parse_args()
    
    # ============================================================
    # 2. LOAD DATASET
    # ============================================================
    df = load_dataset(base_data_path, args.dataset)
    
    aux_df = get_features_of_interest(df, 'AUX')
    aux_df['Dataset'] = args.dataset

    # ============================================================
    # 3. SAVE TENSE CLASSIFICATION RESULTS
    # ============================================================
    print("\n" + "="*50)
    print("SAVING TENSE CLASSIFICATION RESULTS")
    print("="*50)

    # Extract dataset name from features file path
    # e.g., 'batch_2-from_df_pos_features-v2.csv' -> 'batch_2'
    dataset_basename = os.path.basename(args.dataset)
    dataset_name = dataset_basename.split('_pos_features')[0].split('-from_df')[0]

    # Save path under rule_based method
    save_path = os.path.join(base_data_path, 'tense_classification', 'rule_based')

    # Save predictions
    DataProcessing.save_to_file(
        data=aux_df,
        path=save_path,
        prefix=f'{dataset_name}_predictions',
        save_file_type='csv',
        include_version=True
    )

    # Save metadata
    classification_metadata = {
        "dataset_name": dataset_name,
        "source_features_path": args.dataset,
        "method": "rule_based",
        "classification_rule": "AUX with MD tag (modals: will, shall, should, etc.)",
        "total_rows_input": len(df),
        "total_predictions": len(aux_df),
        "future_tense_count": int(aux_df['Future Tense'].sum()),
        "predictions_shape": aux_df.shape,
        "columns": list(aux_df.columns),
        "classification_timestamp": pd.Timestamp.now().isoformat()
    }

    DataProcessing.save_to_file(
        data=classification_metadata,
        path=save_path,
        prefix=f'{dataset_name}_classification_metadata',
        save_file_type='json',
        include_version=False
    )

    print("\n✓ Tense classification complete!")
    print(f"Files saved to: {save_path}")
    print(f"Future tense predictions: {len(aux_df)} out of {len(df)} sentences")