import logging, os

import pandas as pd

from pathlib import Path
from datetime import datetime


class DataFrameFormatter(logging.Formatter):
    def __init__(self, fmt: str, n_rows: int = 4) -> None:
        self.n_rows = n_rows
        super().__init__(fmt)
        
    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, pd.DataFrame):
            s = ''
            if hasattr(record, 'n_rows'):
                self.n_rows = record.n_rows
            lines = record.msg.head(self.n_rows).to_string().splitlines()
            if hasattr(record, 'header'):
                record.msg = record.header.strip()
                s += super().format(record) + '\n'
            for line in lines:
                record.msg = line
                s += super().format(record) + '\n'
            return s.strip()
        return super().format(record)
    
    # if file exists, then create new version
    
    def log_df(self, df, sentence_label):

        log_dir = 'log_text_generation'
        os.makedirs(log_dir, exist_ok=True)

        # Get the current date and time
        now = datetime.now()
        date_time_str = now.strftime("%d%b%Y-%H%M%S")

        log_file = os.path.join('../log_text_generation/', f'{date_time_str}-{sentence_label}.log')
        logging.basicConfig(filename=log_file, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)


        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(self)
        logger.addHandler(ch)

        logger.info(f"Start logging df for {sentence_label}")
        logger.info(df)
        logger.info(f"End logging df for {sentence_label}")
    
    def open_log_df(self, log_file_path):
      # Define the path to the log file
      log_file_path = Path(log_file_path)

      # Read the contents of the log file
      log_contents = log_file_path.read_text()

      # Split the log contents into lines
      log_lines = log_contents.split('\n')

      # Extract the header and data lines
      header = log_lines[0]
      data_lines = log_lines[1:]

      # Create a DataFrame from the data lines
      data = [line.split(' - ') for line in data_lines if line]
      df = pd.DataFrame(data, columns=header.split(' - '))

      return df