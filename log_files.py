import logging, os

import pandas as pd

from pathlib import Path
from datetime import datetime


# class DataFrameFormatter(logging.Formatter):
#     def __init__(self, fmt: str, n_rows: int = 4) -> None:
#         self.n_rows = n_rows
#         super().__init__(fmt)
        
#     def format(self, record: logging.LogRecord) -> str:
#         if isinstance(record.msg, pd.DataFrame):
#             s = ''
#             if hasattr(record, 'n_rows'):
#                 self.n_rows = record.n_rows
#             lines = record.msg.head(self.n_rows).to_string().splitlines()
#             if hasattr(record, 'header'):
#                 record.msg = record.header.strip()
#                 s += super().format(record) + '\n'
#             for line in lines:
#                 record.msg = line
#                 s += super().format(record) + '\n'
#             return s.strip()
#         return super().format(record)
    
#     # if file exists, then create new version
    
#     def log_df(self, df, sentence_label):

#         log_dir = 'log_text_generation'
#         os.makedirs(log_dir, exist_ok=True)

#         # Get the current date and time
#         now = datetime.now()
#         date_time_str = now.strftime("%d%b%Y-%H%M%S")

#         log_file = os.path.join('../log_text_generation/', f'{date_time_str}-{sentence_label}.log')
#         logging.basicConfig(filename=log_file, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)


#         logger = logging.getLogger()
#         logger.setLevel(logging.DEBUG)
#         ch = logging.StreamHandler()
#         ch.setFormatter(self)
#         logger.addHandler(ch)

#         logger.info(f"Start logging df for {sentence_label}")
#         logger.info(df)
#         logger.info(f"End logging df for {sentence_label}")
    
#     def open_log_df(self, log_file_path):
#         # Load the log file
#         try:
#             with open(log_file_path, 'r') as file:
#                 log_lines = file.readlines()
#         except FileNotFoundError:
#             print(f"Error: The file at {log_file_path} does not exist.")
#             return pd.DataFrame()  # Return an empty DataFrame
        
#         # Initialize an empty list to store the data
#         data = []
        
#         # Process each line in the log file
#         for line in log_lines:
#             print(f"Reading line: {line.strip()}")  # Debug print
#             if line.startswith('root - INFO -'):
#                 # Clean and split the line into fields
#                 clean_line = line.strip().replace('root - INFO - ', '')
#                 fields = clean_line.split('\t')  # Split by tab since the data is tab-separated
#                 print(f"Fields: {fields}")  # Debug print
                
#                 # Check if the line contains actual data
#                 if len(fields) == 5:
#                     data.append(fields)
#                 else:
#                     print("Warning: Line format is incorrect or does not contain exactly five fields.")
        
#         if not data:
#             print("No data lines were parsed correctly.")
        
#         # Create DataFrame with the essential columns
#         df = pd.DataFrame(data, columns=["Base Sentence", "Sentence Label", "Model Name", "Domain", "Batch Index"])
        
#         return df


# class DataFrameLogger:
#     def __init__(self, log_dir='../data/logs/', log_filename=None):
#         if log_filename is None:
#             now = datetime.now()
#             date_time_str = now.strftime("%Y%m%d-%H%M%S")
#             log_filename = f'{date_time_str}-log.csv'
#         self.log_filepath = os.path.join(log_dir, log_filename)
#         os.makedirs(log_dir, exist_ok=True)

#         # Check if log file already exists, initialize if not
#         if not os.path.exists(self.log_filepath):
#             self.init_log_file()

#     def init_log_file(self):
#         # Initialize log file with column headers
#         col_names = ["Base Sentence", "Sentence Label", "Model Name", "Domain", "Batch Index", "Timestamp"]
#         empty_df = pd.DataFrame(columns=col_names)
#         empty_df.to_csv(self.log_filepath, index=False)

#     def log_df(self, df, additional_info):
#         # Make sure additional_info keys match initialized columns except 'Timestamp'
#         if 'Timestamp' not in df.columns:
#             df['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         df.to_csv(self.log_filepath, mode='a', header=False, index=False)

#     def load_log(self):
#         return pd.read_csv(self.log_filepath)


import os
import pandas as pd
from datetime import datetime

class DataFrameLogger:
    def __init__(self, log_dir='../data/logs/', log_filename=None):
        if log_filename is None:
            now = datetime.now()
            date_time_str = now.strftime("%Y%m%d-%H%M%S")
            log_filename = f'{date_time_str}-log.log'
        self.log_filepath = os.path.join(log_dir, log_filename)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize if the log file doesn't exist
        if not os.path.isfile(self.log_filepath):
            with open(self.log_filepath, 'w') as file:
                file.write(self.create_log_header())

    def create_log_header(self):
        col_names = ["Base Sentence", "Sentence Label", "Model Name", "Domain", "Batch Index", "Timestamp"]
        return '\t'.join(col_names) + '\n'

    def log_df(self, df):
        # Ensure Timestamp is a column in df
        if 'Timestamp' not in df.columns:
            df['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Append to log file
        with open(self.log_filepath, 'a') as file:
            for index, row in df.iterrows():
                log_line = '\t'.join(str(x) for x in row)
                file.write(log_line + '\n')

    def load_log(self):
        # Load log file as DataFrame
        return pd.read_csv(self.log_filepath, sep='\t')

# Example Usage
