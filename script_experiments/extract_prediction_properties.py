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

from prompts import Prompts
from data_processing import DataProcessing
from prediction_properties import PredictionProperties
from text_generation_models import TextGenerationModelFactory


def load_dataset(script_dir, base_data_path, dataset_name):
    """
    Load dataset from file path.
    
    Parameters
    ----------
    dataset_name : str
        Relative path to dataset file from base data directory
    
    Returns
    -------
    pd.DataFrame
        Loaded dataset
    
    Notes
    -----
    Prints dataset shape and preview to terminal for verification.
    """
    print("\n" + "="*50)
    print("LOAD DATASET")
    print("="*50)
    
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # base_data_path = DataProcessing.load_base_data_path(script_dir)
    data_path = os.path.join(base_data_path, dataset_name)
    
    print(f"Dataset path: {dataset_name}")
    df = DataProcessing.load_from_file(data_path, 'csv', sep=',')
    print(f"Shape: {df.shape}")
    print(f"\nPreview:\n{df.head()}\n")
    
    return df

def load_prompts_and_llms(model_names=None):
    """
    Initialize prompts and language models for prediction property extraction.
    
    Parameters
    ----------
    model_names : str or list of str, optional
        Single model name or list of model names to load.
        If None, loads default model (llama-3.1-8b-instant)
    
    Returns
    -------
    base_prompt : str
        System prompt with prediction properties context
    task : str
        Task description for the model
    format_output : str
        Expected output format instructions
    models : list
        List of initialized TextGenerationModel instances
    
    Notes
    -----
    Available models can be listed via TextGenerationModelFactory.get_all_model_names()
    """
    print("\n" + "="*50)
    print("LOAD PROMPTS & MODELS")
    print("="*50)
    
    # Build base prompt
    prediction_properties = PredictionProperties.get_prediction_properties()
    system_identity_prompt, task, format_output = Prompts.extract_projection_properties()
    base_prompt = f"""{system_identity_prompt} For each prediction, the format is based on: 
    {prediction_properties}
    """
    print("✓ Prompts loaded")
    
    # Initialize models
    tgmf = TextGenerationModelFactory()
    
    if model_names is None:
        # Default models
        model_names = ['llama-3.1-8b-instant']
    elif isinstance(model_names, str):
        # Single model passed as string
        model_names = [model_names]
    
    models = []
    for model_name in model_names:
        try:
            model = tgmf.create_instance(model_name=model_name)
            models.append(model)
            print(f"✓ Loaded: {model.__name__()}")
        except ValueError as e:
            print(f"✗ Failed to load {model_name}: {e}")
    
    if not models:
        raise ValueError("No models were successfully loaded")
    
    print(f"\n✓ Total models loaded: {len(models)}\n")
    
    return base_prompt, task, format_output, models

def llm_certifier(idx, sentence_to_classify, base_prompt, task, format_output, model):
    """
    Extract prediction properties from a sentence using an LLM.
    
    Parameters
    ----------
    idx : int
        Current row index (for debug printing)
    sentence_to_classify : str
        Input sentence to analyze
    base_prompt : str
        System prompt with context
    task : str
        Task instructions
    format_output : str
        Output format requirements
    model : TextGenerationModel
        Model instance to use for generation
    
    Returns
    -------
    str
        Raw LLM response containing extracted properties
    
    Notes
    -----
    Prints full prompt for first 2 examples for verification.
    """
    prompt = f"""{base_prompt}
    
    Sentence to extract the prediction properties: '{sentence_to_classify}'
    {task}
    
    {format_output}
    """
    
    # Debug: show prompt for first few examples
    if idx < 2:
        print(f"\n--- Sample Prompt (idx={idx}) ---")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    
    input_prompt = model.user(prompt)
    raw_text_llm_generation = model.chat_completion([input_prompt])
    
    return raw_text_llm_generation

def get_results(df, base_prompt, task, format_output, models):
    """
    Process all sentences through multiple models to extract prediction properties.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with 'Base Sentence' column
    base_prompt : str
        System prompt with context
    task : str
        Task instructions
    format_output : str
        Output format requirements
    models : list
        List of model instances to use
    
    Returns
    -------
    list of tuple
        Each tuple contains (sentence, model_response, model_name)
    
    Notes
    -----
    Shows detailed output for first 3 examples for verification.
    """
    print("\n" + "="*50)
    print("PROCESSING SENTENCES")
    print("="*50)
    print(f"Total sentences: {len(df)}")
    print(f"Models per sentence: {len(models)}")
    print(f"Total API calls: {len(df) * len(models)}\n")
    
    results = []
    df = df.loc[:3, :]
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sentences"):
        text = row['Base Sentence']
        
        for model in models:
            raw_response = llm_certifier(idx, text, base_prompt, task, format_output, model)
            result = (text, raw_response, model.__name__())
            results.append(result)
            
            # Show detailed output for first few examples
            if idx < 3:
                print(f"\n{'─'*50}")
                print(f"Example {idx} | Model: {model.__name__()}")
                print(f"{'─'*50}")
                print(f"Input: {text[:100]}...")
                print(f"Output: {raw_response[:200]}...")
    
    print(f"\n✓ Processing complete: {len(results)} total results\n")
    return results

def process_llm_results(results: List[tuple]) -> pd.DataFrame:
    """Process LLM results into structured DataFrame."""
    
    results_df = pd.DataFrame(results, columns=['Sentence', 'Raw Response', 'Model Name'])
    
    property_cols = ['No Property', 'Source', 'Target', 'Date', 'Outcome']
    for col in property_cols:
        results_df[col] = ''
    
    success_count = 0
    
    for idx, row in results_df.iterrows():
        try:
            parsed = ast.literal_eval(row['Raw Response'])
            results_df.at[idx, 'No Property'] = ', '.join(parsed.get(0, []) or parsed.get("0", []))
            results_df.at[idx, 'Source'] = ', '.join(parsed.get(1, []) or parsed.get("1", []))
            results_df.at[idx, 'Target'] = ', '.join(parsed.get(2, []) or parsed.get("2", []))
            results_df.at[idx, 'Date'] = ', '.join(parsed.get(3, []) or parsed.get("3", []))
            results_df.at[idx, 'Outcome'] = ', '.join(parsed.get(4, []) or parsed.get("4", []))
            success_count += 1
        except:
            pass  # Keep empty strings
    
    print(f"✓ Parsed {success_count}/{len(results_df)} responses")
    return results_df

if __name__ == "__main__":
    """Usage
    # Default models
    python3 extract_projection_properties.py

    # Single model
    python3 extract_projection_properties.py --models llama-3.1-70b-instruct

    # Multiple models
    python3 extract_projection_properties.py --models llama-3.1-8b-instant llama-3.3-70b-versatile gpt-oss-120b

    # Custom dataset + single model
    python3 extract_projection_properties.py --dataset_name financial_phrase_bank/data.csv --models mistral-7b-instruct

    # Custom dataset + single model + specify filename
    python3 extract_projection_properties.py --dataset_name financial_phrase_bank/annotators/maya_annotations-fpb-binary_labels-v2.csv --models mistral-7b-instruct --save_filename fpb
    
    """
    print("\n" + "="*50)
    print("PREDICTION PROPERTY EXTRACTION")
    print("="*50)
    
    # Parse arguments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = DataProcessing.load_base_data_path(script_dir)
    default_dataset = DataProcessing.load_single_synthetic_data(
        script_dir, batch_idx=1, sep=',', return_as='path'
    )
    
    parser = argparse.ArgumentParser(description='Extract prediction properties from sentences using LLMs')
    parser.add_argument('--dataset_name', default=default_dataset, 
                       help='Path to dataset relative to base data directory. Default: synthetic dataset')
    parser.add_argument('--models', nargs='+', default='llama-3.1-8b-instant',
                       help='Model name(s) to use. Single model or space-separated list. Default: llama-3.1-8b-instant')
    parser.add_argument('--save_filename', type=str, default='extracted_prediction_properties',
                       help='Save the data with extracted properties. Location: data/extract_prediction_properties')
    args = parser.parse_args()
    
    # Execute pipeline
    df = load_dataset(script_dir, base_data_path, args.dataset_name)
    base_prompt, task, format_output, models = load_prompts_and_llms(args.models)
    results = get_results(df, base_prompt, task, format_output, models)
    
    # Process results
    results_df = process_llm_results(results)
    # Build filename with model names
    model_names_str = '-'.join([m.__name__() for m in models])
    filename = f"{args.save_filename}-{model_names_str}"

    extract_prediction_properties_path = "extract_prediction_properties/"
    extract_prediction_properties_full_path = os.path.join(base_data_path, extract_prediction_properties_path)
    DataProcessing.save_to_file(results_df, extract_prediction_properties_full_path, filename, 'csv')
    
    # Summary as JSON
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    summary = {
        "total_processed": len(results_df),
        "shape": results_df.shape,
        "columns": list(results_df.columns),
        "models_used": list(results_df['Model Name'].unique()),
        "sample_results": results_df[['Sentence', 'Source', 'Target', 'Date', 'Model Name']].head(3).to_dict('records')
    }
    
    print(json.dumps(summary, indent=2))
    print("\n✓ Pipeline complete!")
    print(f"\nFull results available in results_df with shape {results_df.shape}")