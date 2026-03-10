import os
import sys
import argparse

from pprint import pprint

# Get the current working directory of the notebook
script_dir = os.getcwd()
# Add the parent directory to the system path
sys.path.append(os.path.join(script_dir, '../'))

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
    default_embedding_model_name = "Hugging Face"
    default_k_results = 3
    
    parser = argparse.ArgumentParser(description='Store data in vector store.')

    # collection_name
    parser.add_argument(
        '--collection_name',
        type=str,
        default=default_collection_name, 
        help='Name of dataset. Used for loading.'
        )
    # persist_directory
    parser.add_argument(
        '--persist_directory',
        type=str,
        default=default_persist_directory,
        help='Path to store vector database.'
        )
    # embedding_model_name
    parser.add_argument(
        '--embedding_model_name',
        type=str,
        default=default_embedding_model_name,
        help='Embedding models (from Langchain).'
        )
    # k
    parser.add_argument(
        '--k',
        type=int,
        default=default_k_results,
        help='The number of retrieved results.'
        )
    # query
    parser.add_argument(
        '--query_string',
        type=str,
        required=True,
        help='Information you are looking for.'
        )
    args = parser.parse_args()
    
    # ============================================================
    # 2. LOAD DATASET & VALIDATE COLUMNS
    # ============================================================
    chroma_loader = ChromaVectorStore(args.collection_name, args.persist_directory)
    chroma_director = VectorStoreDirector(loader=chroma_loader)
    query_results = chroma_director.query(args.embedding_model_name, args.query_string, args.k)
    
    pprint(query_results, width=100, sort_dicts=False)

