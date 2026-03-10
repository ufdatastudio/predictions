import os
import sys
import warnings
import argparse
import pandas as pd

from tqdm import tqdm
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document

# Get the current working directory of the notebook
script_dir = os.getcwd()
# Add the parent directory to the system path
sys.path.append(os.path.join(script_dir, '../'))

import log_files
from data_processing import DataProcessing
from vector_stores import ChromaVectorStore, VectorStoreDirector

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
    base_data_path = DataProcessing.load_base_data_path(script_dir)
    batch_idx = 7
    default_dataset = DataProcessing.load_single_synthetic_data(
        script_dir, batch_idx=batch_idx, sep=',', data_type='observation', return_as='path'
    )
    default_collection_name = "prediction_collection-synthetic_data-oberservations"
    default_persist_directory = os.path.join(base_data_path, "chroma/chroma_langchain_db-oberservations")
    defualt_embedding_model_name = "Hugging Face"
    
    parser = argparse.ArgumentParser(description='Store data in vector store.')
    parser.add_argument(
        '--dataset',
        default=default_dataset,
        help='Path to dataset relative to base data directory.'
        )
    parser.add_argument(
        '--collection_name',
        type=str,
        default=default_collection_name, 
        help='Name of dataset. Used for saving.'
        )
    parser.add_argument(
        '--persist_directory',
        type=str,
        default=default_persist_directory,
        help='Path to store vector database'
        )
    parser.add_argument(
        '--embedding_model_name',
        type=str,
        default=defualt_embedding_model_name,
        help='Embedding models (from Langchain)'
        )
    args = parser.parse_args()
    
    # ============================================================
    # 2. LOAD DATASET & VALIDATE COLUMNS
    # ============================================================
    df = load_dataset(base_data_path, args.dataset)    
    chroma_builder = ChromaVectorStore(args.collection_name, args.persist_directory, 'Base Sentence')
    chroma_director = VectorStoreDirector(builder=chroma_builder)
    chroma_director.construct(args.embedding_model_name, df)
