# Tense Extraction
import os
import sys
import ast
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List

from datetime import datetime

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
    print(f"\nPreview:\n{df.head()}\n")
    
    return df


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
    batch_idx = 1
    default_dataset = DataProcessing.load_single_synthetic_data(
        script_dir, batch_idx=batch_idx, sep=',', return_as='path'
    )
    
    parser = argparse.ArgumentParser(description='Extract prediction properties from sentences using LLMs')
    parser.add_argument('--dataset', default=default_dataset, 
                       help='Path to dataset relative to base data directory.')
    parser.add_argument('--dataset_name', type=str, default=f"synthetic_batch_{batch_idx}", 
                       help='Name of dataset. Used for saving.')
    parser.add_argument('--text_column', type=str, default='Base Sentence',
                       help='Column name containing the text to analyze')
    parser.add_argument('--spacy_disable_components', type=list, default=[],
                    help='The components to not use when extracting features')
    parser.add_argument('--visualize', type=bool, default=True,
                    help='Visualize the first three examples')
    args = parser.parse_args()
    
    # ============================================================
    # 2. LOAD DATASET & VALIDATE COLUMNS
    # ============================================================
    df = load_dataset(base_data_path, args.dataset)
    # df = df.loc[:7, : ]
    
    if args.text_column not in df.columns:
        print(f"\n❌ ERROR: Text column '{args.text_column}' not found in dataset")
        print(f"Available columns: {list(df.columns)}. Enter column of interests.")
        args.text_column = input("Enter column of interests: ")
        print(f"Text column '{args.text_column}'")
    
    # ============================================================
    # 3. EXTRACT FEATURES
    # ============================================================
    spe = SpacyFeatureExtraction(df, args.text_column)
    pos_df, ner_df = spe.extract_pos_ner_features(disable_components=args.spacy_disable_components, visualize=True)
    pos_df['Dataset Name'] = args.dataset_name
    ner_df['Dataset Name'] = args.dataset_name

    pos_df['N Sentences'] = len(df)
    ner_df['N Sentences'] = len(df)

    print(f"\nPOS Preview:\n{pos_df.head()}\n")
    print(f"\nNER Preview:\n{ner_df.head()}\n")

    # ============================================================
    # 4. SAVE EXTRACTED FEATURES
    # ============================================================
    print("\n" + "="*50)
    print("SAVING TENSE EXTRACTION FEATURES")
    print("="*50)

    save_path = os.path.join(base_data_path, 'tense_extraction', args.dataset_name)
    os.makedirs(save_path, exist_ok=True)
    # save_path = os.path.join(save_path, 'tense_extraction')

    # Save POS features
    DataProcessing.save_to_file(
        data=pos_df,
        path=save_path,
        prefix=f'pos_features',
        save_file_type='csv',
        include_version=True
    )

    # Save NER features
    DataProcessing.save_to_file(
        data=ner_df,
        path=save_path,
        prefix=f'ner_features',
        save_file_type='csv',
        include_version=True
    )

    # Save metadata
    extraction_metadata = {
        "dataset_name": f'{args.dataset_name}',
        "dataset_path": args.dataset,
        "save_path": save_path,
        "text_column": args.text_column,
        "total_rows": len(df),
        "pos_features_shape": pos_df.shape,
        "ner_features_shape": ner_df.shape,
        "spacy_disabled_components": args.spacy_disable_components,
        "extraction_timestamp": datetime.now().strftime('%Y-%m-%d')
    }

    DataProcessing.save_to_file(
        data=extraction_metadata,
        path=save_path,
        prefix=f'{args.dataset_name}_extraction_metadata',
        save_file_type='json',
        include_version=False
    )

    print("\n✓ Feature extraction complete!")
    print(f"Files saved to: {save_path}")