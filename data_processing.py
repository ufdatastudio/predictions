import re
import os
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from spacy import displacy
from sklearn.model_selection import train_test_split

# from feature_extraction import SpacyFeatureExtraction
from prediction_properties import PredictionProperties

class DataProcessing:
    """A class to preprocess data"""

    def concat_dfs(dfs: list[pd.DataFrame], axis: int = 0, ignore_index: bool = True) -> pd.DataFrame:
        """Concatenate multiple DataFrames"""
        df = pd.concat(dfs, axis=axis, ignore_index=ignore_index)
        return df
    
    def shuffle_df(df: pd.DataFrame):
        """Shuffle the data"""
        df = df.sample(frac=1).reset_index(drop=True)
        return df

    def split_data(
        vectorized_features,
        cols_with_labels: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = False,
        stratify_by=None  # column name (str) or positional index (int); default None
    ):
        """
        Split features and one or more label columns into train/test sets.

        Parameters
        ----------
        vectorized_features : pd.DataFrame or np.ndarray or sparse matrix
            Row-aligned features.

        cols_with_labels : pd.DataFrame
            One or more label columns (e.g., ['Sentence Label'], or ['Sentence Label', 'Author Type']).

        test_size : float, default=0.2
            Fraction of the dataset to include in the test split.

        random_state : int, default=42
            RNG seed for reproducibility.

        stratify : bool, default=False
            If True, preserve the class proportions of the specified `stratify_by` label
            in both train and test sets.

        stratify_by : str or int, default=None
            Column name (preferred) or positional index in `cols_with_labels` to use for stratification.
            If None and `stratify=True`, defaults to the first label column (index 0).
            If `stratify=False`, this is ignored.
            If there exists multi-variate wrt columns,
                - **Me:** Can still choose the one that's imbalanced more and the not as much imbalanced columns will split based on the one that's the most imbalance.
                - **Copilot:** Stratify on the column with the greatest imbalance or the one most critical for model performance. This ensures that rare classes in that column are preserved across train and test sets. Other label columns will split based on the same indices, which may slightly alter their original ratios.

        Returns
        -------
        tuple
            If 1 label column:
                (X_train, X_test, y1_train, y1_test)
            If 2+ label columns:
                (X_train, X_test, y1_train, y1_test, y2_train, y2_test, ...)
        """
        # Validate label columns
        n_label_cols = cols_with_labels.shape[1]
        if n_label_cols == 0:
            raise ValueError("cols_with_labels must contain at least one label column.")

        label_series_list = []
        for i in range(n_label_cols):
            label_series_list.append(cols_with_labels.iloc[:, i])

        # Determine stratification target if requested
        stratify_target = None
        if stratify:
            # is None, then automatically select first
            if stratify_by is None:
                stratify_target = label_series_list[0]
            # is int, index labels to get which one
            elif isinstance(stratify_by, int):
                if not (0 <= stratify_by < n_label_cols):
                    raise ValueError(f"stratify_by index must be in [0, {n_label_cols-1}]")
                stratify_target = label_series_list[stratify_by]
            # is string, index labels to get which one
            elif isinstance(stratify_by, str):
                if stratify_by not in cols_with_labels.columns:
                    raise ValueError(f"'{stratify_by}' not found in cols_with_labels columns: "
                                    f"{list(cols_with_labels.columns)}")
                stratify_target = cols_with_labels[stratify_by]
            else:
                raise TypeError("stratify_by must be None, an int (column index), or a str (column name).")

        # Perform the split, with fallback if stratification is infeasible
        try:
            splits = train_test_split(
                vectorized_features,
                *label_series_list,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_target
            )
        except ValueError as e:
            # Common causes: a class too small to appear in both splits; test_size too large for minor class
            print(
                f"[WARN] Stratified split failed: {e}\n"
                f"Falling back to non-stratified split. Consider reducing test_size, "
                f"ensuring every class has sufficient samples, or disabling stratification."
            )
            splits = train_test_split(
                vectorized_features,
                *label_series_list,
                test_size=test_size,
                random_state=random_state,
                stratify=None
            )

        return splits

    def join_predictions_with_labels(df: pd.DataFrame, true_labels: pd.Series, y_predictions: pd.Series, model) -> pd.DataFrame:
        """Join the predictions with the true labels DF
        
        Parameters:
        -----------
        df: `pd.DataFrame`
            A DataFrame containing the data

        true_labels: `pd.DataFrame`
            A DataFrame containing the true labels

        y_predictions: `pd.DataFrame`
            A DataFrame containing the predictions from the model

        model: `object`
            An instance of the model
        
        Returns:
        --------
        `pd.DataFrame`
            A DataFrame containing the sentence, labels, and predictions
        """
        assert len(true_labels) == len(y_predictions), "The length of the true labels and predictions must be the same"

        N = len(df) - len(y_predictions)
        test_df = df.loc[N:].copy()
        model_name = model.__name__()
        model_col_name = f"{model_name} Prediction"
        test_df.loc[:, model_col_name] = y_predictions.values

        # Move "Prediction Label" column to second to last position
        cols = list(test_df.columns)
        cols.remove("Prediction Label")
        cols.insert(-1, "Prediction Label")
        test_df = test_df[cols]
        
        return test_df
    
    def join_predictions_with_sentences(df: pd.DataFrame, y_predictions: pd.Series, model) -> pd.DataFrame:
        """Join the model predictions with the sentences. In this case, no true labels are provided
        
        Parameters:
        -----------
        df: `pd.DataFrame`
            A DataFrame containing the data

        y_predictions: `pd.DataFrame`
            A DataFrame containing the predictions from the model

        model: `object`
            An instance of the model
        
        Returns:
        --------
        `pd.DataFrame`
            A DataFrame containing the sentence, labels, and predictions
        """
        assert len(df) == len(y_predictions), "The length of the true labels and predictions must be the same"

        joint_df = df.copy()
        model_name = model.__name__()
        model_col_name = f"{model_name} Prediction"
        joint_df.loc[:, model_col_name] = y_predictions.values
        
        return joint_df
    
    def reformat_df_with_template_number(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Reformat the DataFrame with the template number
        
        Parameters:
        -----------
        df: `pd.DataFrame`
            A DataFrame containing the data
        
        col_name: `str`
            The column name to use for the template number
        
        Returns:
        --------
        `pd.DataFrame`
            A DataFrame with the template number in the specified column
        """
        
        template_numbers = []
        reformat_predictions = []
        indices_to_keep = []

        for idx, prediction in enumerate(df[col_name].values):
            first_word = prediction.split()[0]
            if first_word == "T0:":
                template_numbers.append(0)
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T1:":
                template_numbers.append(1)
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T2:":
                template_numbers.append(2)
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T3:":
                template_numbers.append(3)
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T4:":
                template_numbers.append(4)
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T5:":
                template_numbers.append(5)
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T6:":
                template_numbers.append(6)
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            else:
                continue
        
        new_df = df.iloc[indices_to_keep].copy()
        new_df[col_name] = reformat_predictions
        new_df['Template Number'] = template_numbers
        return new_df
    
    def df_to_list(df: pd.DataFrame, col: str = None, type_of_df: str = "Standard") -> list:
        """Convert a DataFrame to a list
        
        Parameters:
        -----------
        df: `pd.DataFrame`
            A DataFrame containing the data
        
        col: `str`
            The column to convert to a list

        type_of_df: `str`
            The type of DataFrame either Standard or Pivot Table

        Returns:
        --------
        list
            A list containing the data from the DataFrame
        """
        if type_of_df == "Standard":
            return df[col].values.tolist()
        elif type_of_df == "Pivot Table":
            return df.index.values.tolist()
        else:
            return "Options: Standard or Pivot Table"

    def drop_df_columns(df: pd.DataFrame, columns: list):
        """Drop columns
        
        Parameters
        ----------
        df: pd.DataFrame
            The DataFrame to drop columns from
        
        columns: list 
            The list of columns to drop from the DataFrame
        
        Returns
        -------
        df: pd.DataFrame
            The DataFrame with the columns dropped
        """
        df = df.drop(columns=columns)
        return df

    def encode_tags_entities_df(df: pd.DataFrame, sentence_and_label_df: pd.DataFrame) -> pd.DataFrame:
        """Encode the tags or entities in the DataFrame. Use 1 for the presence of the tag or entity and 0 if NaN
        
        Parameters
        ----------
        df: pd.DataFrame
            The DataFrame to encode

        sentence_and_label_df: pd.DataFrame
            The DataFrame containing the sentence and prediction labels

        Returns
        -------
        encoded_df: pd.DataFrame
            The DataFrame with the tags or entities encoded
        """
        bool_df = df.notnull() # Convert the DataFrame to boolean where if presence, place True and NaN place False
        encoded_df = bool_df.astype('int') # Convert the boolean DataFrame to integer where True is 1 and False is 0
        updated_encoded_df = DataProcessing.include_sentence_and_label(encoded_df, sentence_and_label_df)

        return updated_encoded_df

    def array_to_df(data: np.array) -> pd.DataFrame:
        """Convert a numpy array to a DataFrame
        
        Parameters:
        -----------
        data: `np.array`
            An array containing the data to convert to a DataFrame

        Returns:
        --------
        `pd.Series`
            A Series containing the data
        """
        return pd.Series(data)
    
    def convert_tags_entities_to_dataframe(keys_of_mappings: set, mappings: list[list]) -> pd.DataFrame:
        """
        Convert extracted features (Part-of-Speech (POS) tags and Named Entities Recognition (NER)) into a pandas DataFrame.
        
        Parameters:
        -----------
        entities : list
            A list of entities extracted from documents.
        
        all_ner_entities : list
            A list of all unique NER (Named Entity Recognition) tags.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing entities organized by NER tags.
        """
        df_ner = pd.DataFrame(columns=list(keys_of_mappings))
        for i, document_mapping in tqdm(enumerate(mappings)):
            for text, label in document_mapping:
                df_ner.at[i, label] = text
        return df_ner
    
    def include_sentence_and_label(df_to_update: pd.DataFrame, sentence_and_label_df: pd.DataFrame) -> pd.DataFrame:
        """Include the sentence and prediction labels in the DataFrame"""

        if len(df_to_update) == len(sentence_and_label_df):
            df_to_update.insert(0, 'Base Sentence', sentence_and_label_df['Base Sentence'].values)
            df_to_update.insert(1, 'Sentence Label', sentence_and_label_df['Sentence Label'].values)
            return df_to_update
        else:
            print(f"Error: The lengths of df_to_update ({len(df_to_update)}) and sentence_and_label_df ({len(sentence_and_label_df)}) do not match.")
            return df_to_update
        
    def convert_to_df(data, mapping=None):
        """Convert data to a DataFrame or Series.

        Parameters:
        -----------
        data: `np.array`, `list`, `dict`, or `set`
            An array, list, dictionary, or set containing the data to convert to a DataFrame or Series.
        
        mapping: `any`, `optional`
            convert_tags_entities_to_dataframe() --- A dictionary containing mappings for tags/entities


        Returns:
        --------
        `pd.DataFrame` or `pd.Series`
            A DataFrame or Series containing the data.
        """
        if isinstance(data, np.ndarray):
            return DataProcessing.array_to_df(data)
        elif isinstance(data, set) and isinstance(mapping, list):
            return DataProcessing.convert_tags_entities_to_dataframe(data, mapping)
        elif isinstance(data, list) and mapping == 'Open Measures':
            sources = []
            for hit in data:
                sources.append(hit['_source'])
            return pd.DataFrame(sources)
        # elif isinstance(data, tuple[int, json]):
        #     return DataProcessing.json_to_pd(data)
        else:
            raise ValueError("Invalid input: data must be a numpy array, dictionary, list, or set with a mapping.")
    
    def load_prediction_properties():
        return PredictionProperties.get_prediction_properties()
        
    def patterns(nlp):
        # ruler = nlp.add_pipe("entity_ruler", before="ner")
        # patterns = [
        #     {"label": "DATE", "pattern": [{"TEXT": {"REGEX": "20\\d{2}/\\d{2}/\\d{2}"}}]},
        #     {"label": "DATE", "pattern": [{"TEXT": {"REGEX": "20\\d{2}"}}, {"TEXT": {"REGEX": "Q[1-4]"}}]},
        #     {"label": "GPE", "pattern": [{"TEXT": {"REGEX": "LOC"}}, {"TEXT": {"REGEX": "LOC_\d"}}]},
        #     {"label": "DATE", "pattern": [{"TEXT": {"REGEX": "\d{1,2} [a-zA-Z]+ \d{4}"}}]} # NOT WORKING for ex 15 November 2021
        # ]
        # ruler.add_patterns(patterns)
        pass

    def update_ner(label, text):
        """Updates the NER label based on provided rules."""
        if label == "LOC" or label.startswith("LOC_"):
            return "GPE"
        elif label == "FAC":
            return "PERSON"
        elif label == "WORK_OF_ART":
            if re.match(r"Q\d of \d{4}|Quarter \d|\d Q\d", text):
                return "DATE"
            elif text in ["Weather Underground", "The Weather Channel]"]:
                return "ORG"
            else:
                return label  # Keep original label if no match
        elif label == "CARDINAL":
            if re.match(r"\d{1,2} [a-zA-Z]+ \d{4}", text):
                return "DATE"
            elif re.match(r"-\d+°[CF]|-\d+°[CF] to -\d+°[CF]|\d+°[CF] to \d+°[CF]|\d+°[CF]", text):
                return "TEMPERATURE"
            else: return label
        else:
            return label
    
    def select_pattern(template_number: int) -> str:
        """Select the pattern to use based on the template number
        
        Parameters:
        -----------
        template_number: `int`
           The template number that corresponds to the pattern to use

        Returns:
        --------
        `str`
            A string containing the pattern to use
        """

        if template_number == 1:
            pattern = r"On \[(.*?)\], \[(.*?)\] (.*?) that the \[(.*?)\] at \[(.*?)\] \[(.*?)\] \[(.*?)\] by \[(.*?)\] in \[(.*?)\]."
            return pattern
        elif template_number == 2:
            pattern = r"In \[(.*?)\], \[(.*?)\] from \[(.*?)\], (.*?) that the \[(.*?)\] \[(.*?)\] \[(.*?)\] from \[(.*?)\] to \[(.*?)\] in \[(.*?)\]."
            return pattern
        elif template_number == 3:
            pattern = r"\[(.*?)\] (.*?) on \[(.*?)\] that the \[(.*?)\] in \[(.*?)\] \[(.*?)\] \[(.*?)\] by \[(.*?)\] in \[(.*?)\]."
            return pattern
        elif template_number == 4:
            pattern = r"According to a \[(.*?)\] from \[(.*?)\], on \[(.*?)\], the \[(.*?)\] \[(.*?)\] \[(.*?)\] beyond \[(.*?)\] in the timeframe of \[(.*?)\]."
            return pattern
        elif template_number == 5:
            pattern = r"In \[(.*?)\], the \[(.*?)\] in \[(.*?)\] \[(.*?)\] \[(.*?)\] by \[(.*?)\], as (.*?) by \[(.*?)\] on \[(.*?)\]."
            return pattern
        else:
            raise ValueError("The template number is not recognized.")
    
    def sentence_to_df(df: pd.DataFrame) -> pd.DataFrame:
        """Convert an example sentence to a DataFrame

        NOTE: This function is specific to the template used in the example sentence
        
        Parameters:
        -----------
        sentence: `str`
            A sentence containing the variables to extract

        Returns:
        --------
        `pd.DataFrame`
            A DataFrame containing the extracted variables
        """
        template_numbers = df['Template Number'].values
        base_sentences = df['Base Sentence'].values
        
        extracted_data = []

        for template_number, sentence in zip(template_numbers, base_sentences):
            print(sentence)
            pattern = DataProcessing.select_pattern(template_number)
            print(pattern)
            
            match = re.match(pattern, sentence)
            
            if match:
                p_t, p_p, p_a, p_o, p_v, p_s, p_m, p_f = match.groups()
                data = {
                    'p_p': p_p,
                    'p_o': p_o,
                    'p_t': p_t,
                    'p_f': p_f,
                    'p_a': p_a,
                    'p_s': p_s,
                    'p_m': p_m,
                    'p_v': p_v,
                    'p_l': None  # Assuming y_l is not used in this template
                }
                extracted_data.append(data)
            else:
                print(sentence)
                raise ValueError("The sentence does not match the expected template.")
        
        return pd.DataFrame(extracted_data)
    
    def visualize_spacy_doc(doc, save_dir: str = "../paper/exSentence.png"):

        from PIL import Image
        import io
        import cairosvg

        options = {"compact": True, 
                   "bg": "#09a3d5", 
                   "font": "Source Sans Pro"}
        svg = displacy.render(doc, style="dep", jupyter=False, options=options)

            # Convert SVG to PNG
        png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
        img = Image.open(io.BytesIO(png_data))
        img.save(save_dir)

    def get_next_file_number(directory: str, prefix: str, extensions: str = ('.json', '.log', '.csv')):
        """
        Scans the directory for files starting with the given prefix and ending with one of the specified extensions.
        Extracts the numeric suffix and returns the next available number.
        
        Determine the next available file number based on existing files in a directory.

        Parameters
        ----------
        directory : str
            Path to the directory where files are stored.
        prefix : str
            The prefix used in filenames (e.g., 'siteA' for files like 'siteA-1.json').
        extensions : tuple of str, optional
            File extensions to consider when scanning for existing files. Default is ('.json', '.log', '.csv').

        Returns
        -------
        int
            The next available file number (e.g., returns 4 if files with numbers 1, 2, and 3 exist).

        """
        numbers = []
        for name in os.listdir(directory):
            # print(f"Found file: {name}")
            
            if name.startswith(prefix) and name.endswith(extensions):
                try:
                    # Assumes format: prefix-vN.ext (e.g., siteA-v3.json)
                    # Extract the part between prefix and extension
                    after_prefix = name[len(prefix):]  # e.g., "-v3.json"
                    
                    # Remove the extension
                    without_ext = after_prefix.rsplit('.', 1)[0]  # e.g., "-v3"
                    
                    # Extract just the number (handle both "-v3" and "-3" formats)
                    if without_ext.startswith('-v'):
                        number_part = without_ext[2:]  # Remove "-v"
                    elif without_ext.startswith('-'):
                        number_part = without_ext[1:]  # Remove "-"
                    else:
                        continue
                    
                    # print(f"Extracted number part: {number_part}")
                    number = int(number_part)
                    numbers.append(number)
                    # print(f"Added number: {number}")
                    
                except (ValueError, IndexError) as e:
                    print(f"Skipping {name}: {e}")
                    continue
        
        next_num = max(numbers, default=0) + 1
        # print(f"Next file number will be: {next_num}")
        return next_num

    def save_to_file(data, path: str, prefix: str, save_file_type: str):
        """ 
        Save data to any file with an incremented filename based on existing files.

        Parameters
        ----------
        data : dict or list
            The data to be saved in file type format.
        path : str
            Directory path where the file type file will be saved.
        prefix : str
            Prefix for the filename (e.g., 'siteA' results in 'siteA-1.json', 'siteA-2.json', etc.).
        save_file_type : str
            File types such as json, csv, etc

        Returns
        -------
        None
            Saves the file to disk and prints the file path.

        """
        os.makedirs(path, exist_ok=True)
        next_number = DataProcessing.get_next_file_number(path, prefix)
        print(f"Using file number: {next_number}")
        
        if save_file_type == 'json':
            file_name = f"{prefix}-v{next_number}.json"
            file_path = os.path.join(path, file_name)
            print(f"Saving JSON file to: {file_path}")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        elif save_file_type == 'csv' or save_file_type == '.csv':
            file_name = f"{prefix}-v{next_number}.csv"
            file_path = os.path.join(path, file_name)
            print(f"Saving CSV file to: {file_path}")
            data.to_csv(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file type: {save_file_type}")
        # print(f"Saved to: \n\t{file_path}")

    def load_from_file(path: str, 
                       file_type: str = 'csv', 
                       sep = "\t", 
                       encoding = 'utf-8'
                       ):
        """Load data from directory
        
        Parameters
        ----------
        path : str
            Directory path where the file will be loaded from.
        save_file_type : str
            File types such as json, csv, etc

        Returns
        -------
        None
            Saves the file to disk and prints the file path.

        """
        
        if file_type == 'csv': 
            df = pd.read_csv(path, sep=sep, encoding=encoding)
            return df
        else:
            return 'Did not properly load'

    def remove_duplicates(df):
        filt_no_duplicates = (df.duplicated() == False)
        no_duplicates_df = df[filt_no_duplicates]

        return no_duplicates_df
    
    def json_to_df(data) -> pd.DataFrame:
        """
        Notes
        -----
        Modular to where we can use directly or call convert_to_df().
        Need to update calling in convert_to_df().
        """
        row, value = data
        print(row)
        df = pd.DataFrame(value, index=row)
        return df
    
    def load_base_data_path(notebook_dir: str) -> str:
        """Path to data/"""
        return os.path.join(notebook_dir, "../data")
        
    def _build_batch_path(notebook_dir: str, data_type: str, batch_idx: int) -> str:
        """
        Build the file path for a specific batch.
        
        Parameters
        ----------
        notebook_dir : str
            Base notebook directory
        data_type : str
            Either 'prediction' or 'observation'
        batch_idx : int
            Batch index number
        
        Returns
        -------
        str
            Full path to the batch CSV file
        """
        if data_type not in ['prediction', 'observation']:
            raise ValueError("data_type must be either 'prediction' or 'observation'")
        
        base_data_path = DataProcessing.load_base_data_path(notebook_dir)
        batch_folder = f"batch_{batch_idx}-{data_type}"
        data_folder = f"{data_type}_logs"
        file_name = f"batch_{batch_idx}-from_df.csv"
        
        return os.path.join(base_data_path, data_folder, batch_folder, file_name)

    def load_single_synthetic_data(notebook_dir: str, batch_idx: int, sep: str, data_type: str = 'prediction') -> pd.DataFrame:
        """
        Load a single batch of synthetic data.
        
        Parameters
        ----------
        notebook_dir : str
            Base notebook directory
        data_type : str
            Either 'prediction' or 'observation'. Default is 'prediction'.
        batch_idx : int
            Batch index number to load. Default is 7.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing the batch data
        """
        file_path = DataProcessing._build_batch_path(notebook_dir, data_type, batch_idx)
        print(f"Loading: {file_path}")
        
        df = DataProcessing.load_from_file(file_path, 'csv', sep)
        return df

    def load_multiple_batches(notebook_dir: str, sep: str, data_type: str = 'prediction', 
                            batch_indices: list = None, start_idx: int = 1, 
                            end_idx: int = None) -> pd.DataFrame:
        """
        Load multiple batches of synthetic data and concatenate them.
        
        Parameters
        ----------
        notebook_dir : str
            Base notebook directory
        data_type : str
            Either 'prediction' or 'observation'. Default is 'prediction'.
        batch_indices : list, optional
            Specific list of batch indices to load. If provided, start_idx and end_idx are ignored.
        start_idx : int, optional
            Starting batch index (inclusive). Default is 1.
        end_idx : int, optional
            Ending batch index (inclusive). If None, automatically detects the last available batch.
        
        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame containing all batch data
        """
        if data_type not in ['prediction', 'observation']:
            raise ValueError("data_type must be either 'prediction' or 'observation'")
        
        # Determine which batches to load
        if batch_indices is not None:
            indices = batch_indices
        else:
            # Auto-detect the number of available batches if end_idx is None
            if end_idx is None:
                base_data_path = DataProcessing.load_base_data_path(notebook_dir)
                log_directory = os.path.join(base_data_path, f"{data_type}_logs")
                
                if not os.path.exists(log_directory):
                    raise FileNotFoundError(f"Directory not found: {log_directory}")
                
                # Count directories that match the pattern batch_N-{data_type}
                batch_dirs = [d for d in os.listdir(log_directory) 
                            if os.path.isdir(os.path.join(log_directory, d)) 
                            and d.startswith('batch_') 
                            and d.endswith(f'-{data_type}')]
                
                if not batch_dirs:
                    raise ValueError(f"No batch directories found in {log_directory}")
                
                # Extract batch numbers and find the maximum
                batch_numbers = []
                for d in batch_dirs:
                    try:
                        num = int(d.split('_')[1].split('-')[0])
                        batch_numbers.append(num)
                    except (IndexError, ValueError):
                        continue
                
                end_idx = max(batch_numbers) if batch_numbers else start_idx
            
            indices = range(start_idx, end_idx + 1)
        
        # Load all batches
        dfs = []
        for idx in indices:
            try:
                df = DataProcessing.load_single_synthetic_data(
                    notebook_dir=notebook_dir, 
                    batch_idx = idx,
                    sep=sep,
                    data_type=data_type, 
                    )
                dfs.append(df)
                print(f"✓ Loaded batch {idx}")
            except FileNotFoundError:
                print(f"⚠ Warning: Batch {idx} not found, skipping...")
                continue
            except Exception as e:
                print(f"⚠ Error loading batch {idx}: {e}")
                continue
        
        if not dfs:
            raise ValueError("No batches were successfully loaded")
        
        # Concatenate all dataframes
        combined_df = DataProcessing.concat_dfs(dfs)
        print(f"\nSuccessfully loaded and combined {len(dfs)} batches")
        print(f"Total rows: {len(combined_df)}")
        
        return combined_df

    def match_text_label_to_int(df: pd.DataFrame, text_label_col_name: str, 
                            target_label: str = 'PREDICTION',
                            int_label_col_name: str = 'Binary Label') -> pd.DataFrame:
        """
        Convert text labels to integer labels (binary classification).
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the text labels
        text_label_col_name : str
            Name of the column containing text labels (e.g., 'maya_label')
        target_label : str
            The label to encode as 1. Default is 'PREDICTION'.
            Common values: 'PREDICTION', 'NON-PREDICTION', 'OBSERVATION', 'NON-OBSERVATION'
        int_label_col_name : str
            Name for the new integer label column. Default is 'Binary Label'.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with added integer label column where:
            - 1 = matches the target_label
            - 0 = does not match the target_label
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Create filter for matching labels
        filt_match = (result_df[text_label_col_name] == target_label)
        
        # Create two dataframes: matches and non-matches
        match_df = result_df[filt_match].copy()
        match_df[int_label_col_name] = 1
        
        non_match_df = result_df[~filt_match].copy()
        non_match_df[int_label_col_name] = 0
        
        # Concatenate and return
        numerical_labels_df = DataProcessing.concat_dfs([match_df, non_match_df])
        
        return numerical_labels_df