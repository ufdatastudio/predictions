import re
import os
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
from spacy import displacy

from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from prediction_properties import PredictionProperties

class DataProcessing:
    """A class to preprocess data"""

    def concat_dfs(dfs: list[pd.DataFrame], axis: int = 0, ignore_index: bool = True) -> pd.DataFrame:
        """Concatenate multiple DataFrames"""
        df = pd.concat(dfs, axis=axis, ignore_index=ignore_index)
        return df
    
    def shuffle_df(df: pd.DataFrame, random_state):
        """Shuffle the data"""
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        return df
    
    def split_data(
            features_df, 
            labels_df,
            random_state,
            val_size=None, 
            test_size=None,
            stratify_by=None, 
            stratify_kfold: int = None,
            **kwargs):
        """
        Split features and labels into train/test or train/val/test sets.
        
        Parameters
        ----------
        features_df : pd.DataFrame or np.ndarray
            Features to split
        labels_df : pd.DataFrame
            Label columns (e.g., ['Sentence Label', 'Author Type'])
        test_size : float, default=0.2
            Fraction for test set
        val_size : float, default=None
            Fraction for validation set (if None, only train/test split)
        random_state : int, default=42
            Random seed for reproducibility
        stratify_by : str, default=None
            Column name to stratify on (e.g., 'Sentence Label')
            If None, no stratification is used
        
        Returns
        -------
        tuple
            If val_size is None: (X_train, X_test, y_train, y_test)
            If val_size is set: (X_train, X_val, X_test, y_train, y_val, y_test)
        
        Examples
        --------
        # 2-way split with stratification
        X_train, X_test, y_train, y_test = split_data(
            features, labels, stratify_by='Sentence Label'
        )
        
        # 3-way split (60% train, 20% val, 20% test)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            features, labels, val_size=0.2, stratify_by='Sentence Label'
        )
        """
        stratify = labels_df[stratify_by] if stratify_by else None

        # 1) CHECK K-FOLD FIRST
        if stratify_kfold:
            skf = StratifiedKFold(n_splits=stratify_kfold, shuffle=True, random_state=random_state)

            all_folds = []

            for train_idx, val_idx in skf.split(features_df, labels_df):
                X_train = features_df.iloc[train_idx]
                X_val = features_df.iloc[val_idx]
                y_train = labels_df.iloc[train_idx]
                y_val = labels_df.iloc[val_idx]
                
                all_folds.append((X_train, X_val, y_train, y_val))
                
            return all_folds
        
        # No validation set - simple 2-way split
        if val_size is None:
            return train_test_split(
                features_df, 
                labels_df, 
                test_size=test_size, 
                random_state=random_state,
                stratify=stratify
            )
        
        # With validation set - two splits
        # Split 1: Remove test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            features_df, 
            labels_df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Split 2: Split remaining into train and val
        val_size_adjusted = val_size / (1 - test_size)
        stratify_temp = y_temp[stratify_by] if stratify_by else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, 
            y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

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
    
    def reformat_df_with_template_number(df: pd.DataFrame, prediction_templates: list, col_name: str) -> pd.DataFrame:
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
        template_texts = []
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
                template_texts.append(prediction_templates[0])
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T2:":
                template_numbers.append(2)
                template_texts.append(prediction_templates[1])
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T3:":
                template_numbers.append(3)
                template_texts.append(prediction_templates[2])
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T4:":
                template_numbers.append(4)
                template_texts.append(prediction_templates[3])
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T5:":
                template_numbers.append(5)
                template_texts.append(prediction_templates[4])
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            elif first_word == "T6:":
                template_numbers.append(6)
                template_texts.append(prediction_templates[5])
                reformat_predictions.append(prediction[4:])
                indices_to_keep.append(idx)
            else:
                continue
        
        new_df = df.iloc[indices_to_keep].copy()
        new_df[col_name] = reformat_predictions
        new_df['Template Number'] = template_numbers
        new_df['Template Text'] = template_texts
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

    def get_next_file_number(directory: str, prefix: str, extensions: str = ('.json', '.log', '.csv', '.png')):
        """
        Determine the next available file number based on existing files in a directory.
        
        Parameters
        ----------
        directory : str
            Path to the directory where files are stored.
        prefix : str
            The prefix used in filenames (e.g., 'siteA' for files like 'siteA-v3.json').
        extensions : tuple of str, optional
            File extensions to consider when scanning for existing files. Default is ('.json', '.log', '.csv', '.png').
        
        Returns
        -------
        int
            The next available file number (e.g., returns 4 if files with numbers 1, 2, and 3 exist).
        """
        if not os.path.exists(directory):
            return 1
        
        numbers = []
        for name in os.listdir(directory):
            full_path = os.path.join(directory, name)
            
            # Only check files
            if not os.path.isfile(full_path):
                continue
            
            if name.startswith(prefix) and name.endswith(extensions):
                try:
                    # Assumes format: prefix-vN.ext (e.g., siteA-v3.json)
                    after_prefix = name[len(prefix):]  # e.g., "-v3.json"
                    without_ext = after_prefix.rsplit('.', 1)[0]  # e.g., "-v3"
                    
                    # Extract just the number (handle both "-v3" and "-3" formats)
                    if without_ext.startswith('-v'):
                        number_part = without_ext[2:]  # Remove "-v"
                    elif without_ext.startswith('-'):
                        number_part = without_ext[1:]  # Remove "-"
                    else:
                        continue
                    
                    number = int(number_part)
                    numbers.append(number)
                    
                except (ValueError, IndexError):
                    continue
        
        next_num = max(numbers, default=0) + 1
        return next_num

    def get_next_directory_number(directory: str, prefix: str, date_filter: str = None):
        """
        Determine the next available directory number based on existing directories.
        
        Parameters
        ----------
        directory : str
            Path to the parent directory where subdirectories are stored.
        prefix : str
            The prefix used in directory names (e.g., 'undersampled_96d-v4_').
        date_filter : str, optional
            Filter directories by date (e.g., '2026-02-17'). Only counts directories with this date.
        
        Returns
        -------
        int
            The next available directory number.
        
        Notes
        -----
        Expected directory format: {prefix}v{N}_{date}
        Example: undersampled_96d-v4_v2_2026-02-17
        """
        if not os.path.exists(directory):
            return 1
        
        numbers = []
        print(f"\n[DEBUG] Looking for directories with prefix: '{prefix}' and date: '{date_filter}'")
        
        for name in os.listdir(directory):
            full_path = os.path.join(directory, name)
            
            # Only check directories
            if not os.path.isdir(full_path):
                continue
            
            print(f"[DEBUG] Checking directory: '{name}'")
            
            # Check if starts with prefix
            if name.startswith(prefix):
                # If date filter specified, check if directory ends with that date
                if date_filter and not name.endswith(date_filter):
                    print(f"[DEBUG] ✗ Doesn't match date filter '{date_filter}'")
                    continue
                
                print(f"[DEBUG] ✓ Matches prefix and date")
                try:
                    # Extract part after prefix: "v1_2026-02-17" or "v2_2026-02-17"
                    after_prefix = name[len(prefix):]
                    print(f"[DEBUG] After prefix: '{after_prefix}'")
                    
                    # Split by underscore: ["v1", "2026-02-17"]
                    parts = after_prefix.split('_')
                    print(f"[DEBUG] Parts: {parts}")
                    
                    # First part should be version: "v1", "v2", etc.
                    if parts and parts[0].startswith('v') and len(parts[0]) > 1:
                        number_part = parts[0][1:]  # Remove 'v'
                        number = int(number_part)
                        numbers.append(number)
                        print(f"[DEBUG] Found version number: {number}")
                            
                except (ValueError, IndexError) as e:
                    print(f"[DEBUG] ✗ Error extracting number: {e}")
                    continue
            else:
                print(f"[DEBUG] ✗ Doesn't match prefix")
        
        next_num = max(numbers, default=0) + 1
        print(f"[DEBUG] Numbers found: {numbers}")
        print(f"[DEBUG] Next number: {next_num}\n")
        return next_num
    
    def save_to_file(data, 
                     path: str, 
                     prefix: str, 
                     save_file_type: str, 
                     include_version: bool = True, 
                     **kwargs: dict) -> None:
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
            File types such as json, csv, png, etc
        include_version : bool, optional
            If True, uses versioning system (adds -v1, -v2, etc.)
            If False, saves directly without version suffix (will overwrite)
            Default is False.
        **kwargs : dict
            Additional arguments for specific file types:
            - For PNG: dpi (default=300), bbox_inches (default='tight')
        Returns
        -------
        None
            Saves the file to disk and prints the file path.
        """
        os.makedirs(path, exist_ok=True)
        
        # Determine filename based on versioning
        if include_version:
            next_number = DataProcessing.get_next_file_number(path, prefix)
            print(f"Using file number: {next_number}")
        
        if save_file_type == 'json':
            if include_version:
                file_name = f"{prefix}-v{next_number}.json"
            else:
                file_name = f"{prefix}.json"
            file_path = os.path.join(path, file_name)
            print(f"Saving JSON file to: {file_path}")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        
        elif save_file_type in ['csv', '.csv']:
            if include_version:
                file_name = f"{prefix}-v{next_number}.csv"
            else:
                file_name = f"{prefix}.csv"
            file_path = os.path.join(path, file_name)
            print(f"Saving CSV file to: {file_path}")
            data.to_csv(file_path, index=False)
        
        elif save_file_type in ['png', '.png', 'PNG']:
            import matplotlib.pyplot as plt
            
            if include_version:
                file_name = f"{prefix}-v{next_number}.png"
            else:
                file_name = f"{prefix}.png"
            file_path = os.path.join(path, file_name)
            print(f"Saving PNG file to: {file_path}")
            
            # Get optional parameters with defaults
            dpi = kwargs.get('dpi', 300)
            bbox_inches = kwargs.get('bbox_inches', 'tight')
            
            # Save the current figure
            plt.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
            plt.close()
        
        else:
            raise ValueError(f"Unsupported file type: {save_file_type}. Choose from [json, csv, png]")

    def load_from_file(path: str, 
                    file_type: str = 'csv', 
                    sep = ",", 
                    encoding = 'utf-8',
                    **kwargs
                    ):
        """Load data from directory
        
        Parameters
        ----------
        path : str
            Directory path where the file will be loaded from.
        file_type : str
            File types such as json, csv, etc
        sep : str
            Delimiter for CSV files
        encoding : str
            File encoding
        **kwargs
            Additional keyword arguments passed to pd.read_csv()
            (e.g., header=None, names=['col1', 'col2'], dtype=...)
        
        Returns
        -------
        pd.DataFrame
            Loaded dataframe
        """
        
        if file_type == 'csv': 
            df = pd.read_csv(path, sep=sep, encoding=encoding, **kwargs)
            return df
        else:
            return 'Did not properly load'

    def get_latest_file(directory: str, prefix: str, file_type: str = 'csv') -> str:
        """
        Parameters
        ----------
        directory : str
            Path to the directory to search in.
        prefix : str
            Filename prefix to match (e.g., 'classifications').
        file_type : str, default 'csv'
            File extension to match (without dot).

        Notes
        -----
        Relies on the same versioning format as get_next_file_number()
        (e.g., classifications-v3.csv). Returns the filename with the
        highest version number.

        Returns
        -------
        str
            Filename of the latest versioned file.
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        matched = {}
        for name in os.listdir(directory):
            full_path = os.path.join(directory, name)
            if not os.path.isfile(full_path):
                continue
            if name.startswith(prefix) and name.endswith(f'.{file_type}'):
                try:
                    after_prefix = name[len(prefix):]        # e.g., "-v3.csv"
                    without_ext = after_prefix.rsplit('.', 1)[0]  # e.g., "-v3"
                    if without_ext.startswith('-v'):
                        number = int(without_ext[2:])
                    elif without_ext.startswith('-'):
                        number = int(without_ext[1:])
                    else:
                        continue
                    matched[number] = name
                except (ValueError, IndexError):
                    continue

        if not matched:
            raise FileNotFoundError(
                f"No versioned '{prefix}' files found in: {directory}"
            )

        # Return the filename with the highest version number
        latest_filename = matched[max(matched)]
        return latest_filename

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

    def load_single_synthetic_data(notebook_dir: str, 
                                   batch_idx: int, 
                                   sep: str, 
                                   data_type: str = 'prediction',
                                   return_as: str = 'dataframe') -> pd.DataFrame:
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
        sep : str
            Separator for CSV file
        return_as : str
            Either 'dataframe' or 'path'. Default is 'dataframe'.
        
        Returns
        -------
        pd.DataFrame or str
            DataFrame containing the batch data or path string
        """
        file_path = DataProcessing._build_batch_path(notebook_dir, data_type, batch_idx)
        
        if return_as.lower() in ['path', 'string']:
            return file_path
        else:  # default to dataframe
            print(f"Loading: {file_path}")
            df = DataProcessing.load_from_file(file_path, 'csv', sep)
            return df

    def load_multiple_batches(notebook_dir: str, sep: str = ',', data_type: str = 'prediction', 
                            batch_indices: list = None, start_idx: int = 1, 
                            end_idx: int = None, return_as: str = 'dataframe') -> pd.DataFrame:
        """
        Load multiple batches of synthetic data and concatenate them.
        
        Parameters
        ----------
        notebook_dir : str
            Base notebook directory
        sep : str
            Separator for CSV file. Default is ','.
        data_type : str
            Either 'prediction' or 'observation'. Default is 'prediction'.
        batch_indices : list, optional
            Specific list of batch indices to load. If provided, start_idx and end_idx are ignored.
        start_idx : int, optional
            Starting batch index (inclusive). Default is 1.
        end_idx : int, optional
            Ending batch index (inclusive). If None, automatically detects the last available batch.
        return_as : str
            Either 'dataframe' or 'path'. Default is 'dataframe'.
            If 'path', returns list of file paths instead of loading data.
        
        Returns
        -------
        pd.DataFrame or list
            Concatenated DataFrame containing all batch data, or list of file paths
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
        
        # If return_as is 'path', return list of paths
        if return_as.lower() in ['path', 'string']:
            paths = []
            for idx in indices:
                try:
                    path = DataProcessing.load_single_synthetic_data(
                        notebook_dir=notebook_dir, 
                        batch_idx=idx,
                        sep=sep,
                        data_type=data_type,
                        return_as='path'
                    )
                    paths.append(path)
                    print(f"✓ Found batch {idx}: {path}")
                except FileNotFoundError:
                    print(f"⚠ Warning: Batch {idx} not found, skipping...")
                    continue
                except Exception as e:
                    print(f"⚠ Error locating batch {idx}: {e}")
                    continue
            
            if not paths:
                raise ValueError("No batch paths were found")
            
            print(f"\nFound {len(paths)} batch file(s)")
            return paths
        
        # Default behavior: load as dataframes
        dfs = []
        for idx in indices:
            try:
                df = DataProcessing.load_single_synthetic_data(
                    notebook_dir=notebook_dir, 
                    batch_idx=idx,
                    sep=sep,
                    data_type=data_type,
                    return_as='dataframe'
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
    
    def apply_resampling_full_dimensions(
            df: pd.DataFrame,
            embedding_col: str,
            label_col: str,
            random_state,
            method: str,
            sampling_strategy: str = 'auto',
            save_path: str = None,
            save_prefix: str = None,
            save_file_type: str = None) -> pd.DataFrame:
        """Apply resampling to full 96-dimensional embeddings."""
        X_full = np.stack(df[embedding_col].values)
        y = df[label_col].values

        print("\n" + "="*40)
        print(f"RESAMPLE: {embedding_col} | {label_col}")
        print("="*40)
        print(f"X_full Shape: {X_full.shape}")
        print(f"\nX_full Preview:\n{X_full}\n")
        print(f"y Shape: {y.shape}")
        print(f"\ny Preview:\n{y}\n")


        if method == 'oversample':
            resampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        elif method == 'undersample':
            resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'oversample' or 'undersample'.")
        
        X_resampled, y_resampled = resampler.fit_resample(X_full, y)
        
        # Get indices from resampling
        indices = resampler.sample_indices_ if hasattr(resampler, 'sample_indices_') else None
        
        # Create resampled dataframe by duplicating original rows
        if indices is not None:
            resampled_df = df.iloc[indices].copy().reset_index(drop=True)
            # Update the embedding and label columns with resampled values
            resampled_df[embedding_col] = list(X_resampled)
            resampled_df[label_col] = y_resampled
        else:
            # Fallback: just include embedding and label columns
            resampled_df = pd.DataFrame({
                embedding_col: list(X_resampled),
                label_col: y_resampled
            })
        
        if save_path:
            DataProcessing.save_to_file(
                data=resampled_df,
                path=save_path,
                prefix=save_prefix,
                save_file_type=save_file_type
            )
        
        return resampled_df
    
    def extract_features_for_visualization(df: pd.DataFrame, 
                                       embedding_col: str, 
                                       label_col: str) -> pd.DataFrame:
        """Extract first 2 dimensions from embeddings for visualization."""
        print("\n" + "="*40)
        print(f"EXTRACT FEATURES FOR VISUALIZING: {embedding_col} | {label_col}")
        print("="*40)
        print(f"df Shape: {df.shape}")
        print(f"\ndf Preview:\n{df.head(7)}\n")

        embeddings_array = np.stack(df[embedding_col].values)
        features_df = pd.DataFrame({
            'Feature_1': embeddings_array[:, 0],
            'Feature_2': embeddings_array[:, 1],
            'Label': df[label_col]
        })
        return features_df