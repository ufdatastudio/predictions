import pandas as pd
import re
import os
import sys
import argparse

# Add project modules to path
script_dir = os.getcwd()
sys.path.append(os.path.join(script_dir, '../'))
from data_processing import DataProcessing

def main(input_folder):
    data = []
    
    # Get all .tml files
    all_files = [f for f in os.listdir(input_folder) if f.endswith('.tml')]
    total_files = len(all_files)
    
    print(f"Found {total_files} .tml files")
    print("Processing...")
    
    for idx, filename in enumerate(all_files, 1):
        filepath = os.path.join(input_folder, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove XML declaration and TimeML tags
        content = re.sub(r'<\?xml.*?\?>', '', content)
        content = re.sub(r'<TimeML[^>]*>', '', content)
        content = re.sub(r'</TimeML>', '', content)
        
        # Remove all XML tags (EVENT, TIMEX3, MAKEINSTANCE, TLINK, SLINK, etc.)
        clean_text = re.sub(r'<[^>]+>', '', content)
        
        # Clean up whitespace and newlines
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if clean_text:
            data.append({
                'Base Sentence': clean_text,
                'File Name': filename
            })
        
        # Display progress every 20 files or at the end
        if idx % 1 == 0 or idx == total_files:
            print(f"  Progress: {idx}/{total_files} files processed")
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    print("\n" + "="*50)
    print("TIMEBANK SENTENCES EXTRACTION")
    print("="*50)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_path = DataProcessing.load_base_data_path(script_dir)

    parser = argparse.ArgumentParser(description='Extract prediction properties from sentences using LLMs')
    parser.add_argument('--save_filename', type=str, default='extracted_timebank_sentences',
                       help='Save the data with extracted properties. Location: data/timebank_1_2/data/extracted_timebank_sentences')
    args = parser.parse_args()
    
    input_folder = os.path.join(base_data_path, 'timebank_1_2/data/timeml')
    print(f"Input folder: {input_folder}\n")
    
    df = main(input_folder)
    filename = f"{args.save_filename}"

    extracted_timebank_sentences_path = "timebank_1_2/data/extracted_timebank_sentences"
    extract_timebank_sentence_full_path = os.path.join(base_data_path, extracted_timebank_sentences_path)
    DataProcessing.save_to_file(df, extract_timebank_sentence_full_path, filename, 'csv')
    
    print(f"\nExtracted {len(df)} sentences")
    print(df.head())