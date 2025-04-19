import logging, os, csv, pathlib

import pandas as pd

from datetime import datetime

class LogData:
    def __init__(self, base_path, log_file_path, batch_path, batch_name):
        self.base_path = base_path
        self.log_file_path = log_file_path
        self.batch_path = batch_path
        save_info_log_file_name = batch_name

        self.log_directory = os.path.join(self.base_path, self.log_file_path)
        # print(f"self.log_directory: {self.log_directory}")
        os.makedirs(self.log_directory, exist_ok=True)

        self.save_info_file = os.path.join(self.batch_path, save_info_log_file_name)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=self.save_info_file)
        logging.info(f"Logging configured to save to: {self.log_directory}")
  
    def dataframe_to_csv(self, df, file_name):
        """Writes a Pandas DataFrame to a CSV file."""

        save_csv = os.path.join(self.batch_path, file_name)
        print(f"Save CSV: {save_csv}")

        try:
            df.to_csv(save_csv, index=False)  # index=False to avoid writing DataFrame index
            logging.info(f"DataFrame successfully written to CSV: {save_csv}")
            return True
        except Exception as e:
            logging.error(f"Error writing DataFrame to CSV: {e}")
            return False

    def csv_to_log(self, csv_file_name, log_file_name):
        """Reads a CSV file and writes its content to a log file."""
        print("\nCSV to Log")
        saved_csv = os.path.join(self.batch_path, csv_file_name)
        save_log = os.path.join(self.batch_path, log_file_name)

        try:
            with open(saved_csv, 'r') as csvfile, open(save_log, 'a') as logfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    log_message = f"CSV Row: {', '.join(map(str, row))}"
                    logfile.write(log_message + '\n')
            logging.info(f"CSV content successfully written to log file: {save_log}")
            return True
        except FileNotFoundError:
            logging.error(f"CSV file not found: {saved_csv}")
            return False
        except Exception as e:
            logging.error(f"Error writing CSV to log file: {e}")
            return False

    def log_to_csv(self, log_file_name, csv_file_name, lines_to_ignore=None, delimiter=','):
        """
        Reads a log file, extracts relevant lines, and writes them to a CSV file.

        Args:
            log_file_path (str): Path to the log file.
            csv_file_path (str): Path to save the CSV file.
            lines_to_ignore (list, optional): List of strings or patterns to identify lines to skip. Defaults to None.
            delimiter (str, optional): Delimiter for the CSV file. Defaults to ','.
        """
        print("=====READ LOG=====\nLog to CSV")
        # log_input_path = os.path.join(self.log_directory, file_name)
        saved_log = os.path.join(self.batch_path, log_file_name)
        print(f"saved_log: {saved_log}")
        saved_csv = os.path.join(self.batch_path, csv_file_name)
        print(f"saved_csv: {saved_csv}")

        if lines_to_ignore is None:
            lines_to_ignore = []

        try:
            extracted_data = []
            with open(saved_log, 'r') as logfile:
                print("Open log")
                for line in logfile:
                    skip_line = False
                    for ignore_pattern in lines_to_ignore:
                        if ignore_pattern in line:
                            skip_line = True
                            break
                    if not skip_line:
                        # Assuming the relevant data in the log file is comma-separated
                        # You might need more sophisticated parsing based on your log format
                        parts = line.strip().split(delimiter)
                        extracted_data.append(parts)

            with open(saved_csv, 'w', newline='') as csvfile:
                print("Open csv")
                writer = csv.writer(csvfile, delimiter=delimiter)
                writer.writerows(extracted_data)

            logging.info(f"Successfully extracted data from log file to CSV: {saved_csv}, ignoring lines containing: {lines_to_ignore}")
            return True
        except FileNotFoundError:
            logging.error(f"Log file not found: {saved_log}")
            return False
        except Exception as e:
            logging.error(f"Error processing log file to CSV: {e}")
            return False
        
    def csv_to_dataframe(self, csv_file_name):
        """Reads a CSV file into a Pandas DataFrame."""
        print("CSV to DF")
        saved_csv = os.path.join(self.batch_path, csv_file_name)
        print(f"Load saved csv: {saved_csv}")
        
        try:
            df = pd.read_csv(saved_csv, delimiter=',')
            logging.info(f"CSV file successfully read into DataFrame: {saved_csv}")
            return df
        except FileNotFoundError:
            logging.error(f"CSV file not found: {saved_csv}")
            return None
        except pd.errors.EmptyDataError:
            logging.warning(f"CSV file is empty: {saved_csv}")
            return pd.DataFrame() # Return an empty DataFrame
        except Exception as e:
            logging.error(f"Error reading CSV file into DataFrame: {e}")
            return None

