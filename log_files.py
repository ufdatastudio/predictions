import logging, os, csv

import pandas as pd

from datetime import datetime

class LogData:

    def __init__(self, log_directory, file_name):
        # Configure logging (if not already configured)
        self.log_directory = log_directory
        os.makedirs(log_directory, exist_ok=True)
        self.log_file_path = os.path.join(log_directory, file_name)

        if not logging.root.handlers:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s',
                                filename=self.log_file_path)
            logging.info(f"Logging configured to save to: {self.log_file_path}")
        else:
            logging.info("Logging already configured.")

    def dataframe_to_csv(self, df, csv_file_path):
        """Writes a Pandas DataFrame to a CSV file."""
        try:
            df.to_csv(csv_file_path, index=False)  # index=False to avoid writing DataFrame index
            logging.info(f"DataFrame successfully written to CSV: {csv_file_path}")
            return True
        except Exception as e:
            logging.error(f"Error writing DataFrame to CSV: {e}")
            return False

    def csv_to_log(self, csv_file_path, file_name):
        """Reads a CSV file and writes its content to a log file."""
        log_output_path = os.path.join(self.log_directory, file_name)

        try:
            with open(csv_file_path, 'r') as csvfile, open(log_output_path, 'a') as logfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    log_message = f"CSV Row: {', '.join(map(str, row))}"
                    logfile.write(log_message + '\n')
            logging.info(f"CSV content successfully written to log file: {log_output_path}")
            return True
        except FileNotFoundError:
            logging.error(f"CSV file not found: {csv_file_path}")
            return False
        except Exception as e:
            logging.error(f"Error writing CSV to log file: {e}")
            return False

    def log_to_csv(self, file_name, csv_file_path, lines_to_ignore=None, delimiter=','):
        """
        Reads a log file, extracts relevant lines, and writes them to a CSV file.

        Args:
            log_file_path (str): Path to the log file.
            csv_file_path (str): Path to save the CSV file.
            lines_to_ignore (list, optional): List of strings or patterns to identify lines to skip. Defaults to None.
            delimiter (str, optional): Delimiter for the CSV file. Defaults to ','.
        """
        log_input_path = os.path.join(self.log_directory, file_name)
        
        if lines_to_ignore is None:
            lines_to_ignore = []

        try:
            extracted_data = []
            with open(log_input_path, 'r') as logfile:
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

            with open(csv_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=delimiter)
                writer.writerows(extracted_data)

            logging.info(f"Successfully extracted data from log file to CSV: {csv_file_path}, ignoring lines containing: {lines_to_ignore}")
            return True
        except FileNotFoundError:
            logging.error(f"Log file not found: {log_input_path}")
            return False
        except Exception as e:
            logging.error(f"Error processing log file to CSV: {e}")
            return False
        

    def csv_to_dataframe(self, csv_file_path):
        """Reads a CSV file into a Pandas DataFrame."""
        try:
            df = pd.read_csv(csv_file_path)
            logging.info(f"CSV file successfully read into DataFrame: {csv_file_path}")
            return df
        except FileNotFoundError:
            logging.error(f"CSV file not found: {csv_file_path}")
            return None
        except pd.errors.EmptyDataError:
            logging.warning(f"CSV file is empty: {csv_file_path}")
            return pd.DataFrame() # Return an empty DataFrame
        except Exception as e:
            logging.error(f"Error reading CSV file into DataFrame: {e}")
            return None

