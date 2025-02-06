import re, spacy
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

class DataProcessing:
    """A class to preprocess data"""

    def concat_dfs(dfs: list[pd.DataFrame]):
        """Concatenate multiple DataFrames"""
        df = pd.concat(dfs, ignore_index=True)
        return df
    
    def shuffle_df(df: pd.DataFrame):
        """Shuffle the data"""
        df = df.sample(frac=1).reset_index(drop=True)
        return df
    
    def split_data(vectorized_features, prediction_labels: pd.DataFrame):
        """Split the data into training and testing sets
        
        Parameters:
        -----------
        vectorized_features: Matrix
            A DataFrame containing the vectorized features
        
        prediction_labels: `pd.DataFrame`
            A DataFrame containing the prediction labels

        Returns:
        --------
        tuple
            A tuple containing the training and testing sets for the vectorized predictions and prediction labels
        """

        X_train, X_test, y_train, y_test = train_test_split(vectorized_features, prediction_labels, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    
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

        for prediction in df[col_name].values:
            first_word = prediction.split()[0]
            if first_word == "T1:":
                template_numbers.append(1)
                reformat_predictions.append(prediction[4:])
            elif first_word == "T2:":
                template_numbers.append(2)
                reformat_predictions.append(prediction[4:])
            elif first_word == "T3:":
                template_numbers.append(3)
                reformat_predictions.append(prediction[4:])
            elif first_word == "T4:":
                template_numbers.append(4)
                reformat_predictions.append(prediction[4:])
            elif first_word == "T5:":
                template_numbers.append(5)
                reformat_predictions.append(prediction[4:])
            else:
                raise ValueError("The template number is not recognized.")
        
        df[col_name] = reformat_predictions
        df['Template Number'] = template_numbers
        return df
    
    def df_to_list(df: pd.DataFrame, col: str) -> list:
        """Convert a DataFrame to a list
        
        Parameters:
        -----------
        df: `pd.DataFrame`
            A DataFrame containing the data
        
        col: `str`
            The column to convert to a list

        Returns:
        --------
        list
            A list containing the data
        """
        return df[col].values.tolist()
    
    @staticmethod
    def setup_spacy():
        """
        Setup the NLP pipeline with SpaCy and add custom entity rulers if needed.
        
        Returns:
        --------
        nlp : spacy.Language
            A SpaCy language processing object with an added entity ruler for custom patterns.
        """
        nlp = spacy.load("en_core_web_sm")
        # ruler = nlp.add_pipe("entity_ruler", before="ner")
        # patterns = [
        #     {"label": "DATE", "pattern": [{"TEXT": {"REGEX": "20\\d{2}/\\d{2}/\\d{2}"}}]},
        #     {"label": "DATE", "pattern": [{"TEXT": {"REGEX": "20\\d{2}"}}, {"TEXT": {"REGEX": "Q[1-4]"}}]},
        #     {"label": "GPE", "pattern": [{"TEXT": {"REGEX": "LOC"}}, {"TEXT": {"REGEX": "LOC_\d"}}]},
        #     {"label": "DATE", "pattern": [{"TEXT": {"REGEX": "\d{1,2} [a-zA-Z]+ \d{4}"}}]} # NOT WORKING for ex 15 November 2021
        # ]
        # ruler.add_patterns(patterns)
        return nlp

    @staticmethod
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


    @staticmethod
    def extract_entities(data: pd.Series, nlp: spacy.Language, disable_components: list, batch_size: int = 50):
        """
        Extract entities using the provided SpaCy NLP model.

        Parameters:
        -----------
        data : `pd.Series`
            A Series containing textual data for entity extraction.
        
        nlp : `spacy.Language`
            A SpaCy NLP model.
        
        batch_size : `int`
            The batch size for processing the data.

        Returns:
        --------
        tuple
            A tuple containing a list of entities and a set of unique NER tags.
        """
        tags = []
        all_pos_tags = set()

        entities = []
        all_ner_tags = set()

        label_counts = {}

        for doc in nlp.pipe(data, disable=disable_components, batch_size=batch_size):
            doc_tags = []
            for token in doc:
                doc_tags.append((token.text, token.pos_))
                all_pos_tags.add(token.pos_)
            tags.append(doc_tags)

            doc_entities = []
            for ent in doc.ents:
                label = ent.label_
                text = ent.text
                # updated_label = DataProcessing.update_ner(label, text)  # update the label
                
                count_key = f"{label}_{doc}"
                if count_key in label_counts:
                    label_counts[count_key] += 1
                else:
                    label_counts[count_key] = 1
                unique_label = f"{label}_{label_counts[count_key]}"

                doc_entities.append((text, unique_label))  # changed label to updated_label
                all_ner_tags.add(unique_label)

            entities.append(doc_entities)

        return tags, all_pos_tags, entities, all_ner_tags

    @staticmethod
    def entities_to_dataframe(entities, all_ner_tags):
        """
        Convert extracted entities into a pandas DataFrame.
        
        Parameters:
        -----------
        entities : list
            A list of entities extracted from documents.
        all_ner_tags : list
            A list of all unique NER tags.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing entities organized by NER tags.
        """
        df_ner = pd.DataFrame(columns=list(all_ner_tags))
        for i, document_entities in enumerate(entities):
            for text, label in document_entities:
                df_ner.at[i, label] = text
        return df_ner
    

 # Functions to disregard   
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
    

    

    