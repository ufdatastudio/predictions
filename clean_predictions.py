"""
Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)

Factory Method Design Pattern (https://refactoring.guru/design-patterns/factory-method/python/example#lang-features)
"""

import re
import pandas as pd
from abc import ABC, abstractmethod

class DataCleaningFactory(ABC):
    """An abstract base class for data cleaning steps"""

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the data cleaning factory with a dataframe.

        Parameters:
        -----------
        dataframe : `pd.DataFrame`
            The DataFrame containing prediction data.
        """
        self.df = dataframe

    def get_reviews(self, col_name):
        """Get the reviews from the DataFrame"""
        return self.df[col_name].values
    
    @abstractmethod
    def lower_case(self):
        """Convert all text to lower case"""
        pass
    
    @abstractmethod
    def remove_html_and_urls(self):
        """Remove HTML tags and URLs from the text"""
        pass
    
    def locate_and_replace_contractions(self, review):
        """Find the contractions to replace from a specific review

        Parameters
        ----------
        review: `str`
            A specific review

        Return
        ------
        non_contraction_review: `str`
            The updated specific review with contractions expanded
        
        """
        store_contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he's": "he is",
            "isn't": "is not",
            "it's": "it is",
            "let's": "let us",
            "mustn't": "must not",
            "shan't": "shall not",
            "she's": "she is",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they're": "they are",
            "wasn't": "was not",
            "we're": "we are",
            "weren't": "were not",
            "won't": "will not",
            "wouldn't": "would not",
            "you're": "you are",
            "you'll": "you will",
            "you'd": "you would",
            "we'll": "we will",
            "we've": "we have",
            "we'd": "we would",
            "I'm": "I am",
            "i've": "I have",
            "I've": "I have",
            "I'd": "I would",
            "it'll": "it will",
            "they'll": "they will",
            "they've": "they have",
            "they'd": "they would",
            "he'll": "he will",
            "he'd": "he would",
            "she'll": "she will",
            "we'd": "we would",
            "we'll": "we will",
            "you've": "you have",
            "you'd": "you would",
            "you'll": "you will",
            "I'll": "I will",
            "I'd": "I would",
            "it's": "it is",
            "it'd": "it would",
            "i'm": "I am",
            "he's": "he is",
            "he'll": "he will",
            "she's": "she is",
            "she'll": "she will",
            "we're": "we are",
            "we've": "we have",
            "we'll": "we will",
            "you're": "you are",
            "you've": "you have",
            "you'll": "you will",
            "they're": "they are",
            "they've": "they have",
            "they'll": "they will",
            "that's": "that is",
            "that'll": "that will",
            "that'd": "that would",
            "who's": "who is",
            "who'll": "who will",
            "who'd": "who would",
            "what's": "what is",
            "what'll": "what will",
            "what'd": "what would",
            "when's": "when is",
            "when'll": "when will",
            "when'd": "when would",
            "where's": "where is",
            "where'll": "where will",
            "where'd": "where would",
            "why's": "why is",
            "why'll": "why will",
            "why'd": "why would",
            "how's": "how is",
            "how'll": "how will",
            "how'd": "how would"
        }
        if isinstance(review, str):
            get_words = review.split()

            store_non_contraction_words = []

            for word in get_words:
                if word in store_contractions:
                    non_contraction_form = store_contractions[word]
                    store_non_contraction_words.append(non_contraction_form)

                else:
                    store_non_contraction_words.append(word)

            non_contraction_review = ' '.join(store_non_contraction_words)
            return non_contraction_review
        else:
            return review
        
    @abstractmethod
    def remove_contractions(self):
        """Remove contractions in the text"""
        pass
    
    @abstractmethod
    def remove_non_alphabetical_characters(self):
        """Remove non-alphabetical characters from the text"""
        pass
    
    @abstractmethod
    def remove_extra_spaces(self):
        """Remove extra spaces from the text"""
        pass

class PredictionDataCleaner(DataCleaningFactory):
    """A concrete class for cleaning prediction data"""

    def lower_case(self, col_name: str):
        """Convert all text to lower case

        Parameters:
        -----------
        df: `pd.DataFrame`
            The input data to lower case
        
        col_name: `str`
            Column to lower case

        Return:
        -------
        df: `pd.DataFrame`
            An updated DataFrame with the lower cased reviews
        """
        
        lower_case_reviews = []
        text_reviews = self.get_reviews(col_name)
        
        for text_reviews_idx in range(len(text_reviews)):
            text_review = text_reviews[text_reviews_idx]

            # NOT all reviews are strings, thus all can't be converted to lower cased
            if type(text_review) != str:
                lower_case_reviews.append(text_review)
            else:
                update_text_review = text_review.lower()
                lower_case_reviews.append(update_text_review)

        self.df[col_name] = lower_case_reviews
        return self.df
    
    def remove_html_and_urls(self, col_name: str):
        """Remove HTML and URLs from all reviews

        Parameters:
        -----------
        col_name: `str`
            Column with reviews

        Return:
        -------
        df: `pd.DataFrame`
            An updated DataFrame with the html_and_urls removed
        """
    
        cleaned_reviews = []
        text_reviews = self.get_reviews(col_name)

        for text_reviews_idx in range(len(text_reviews)):
            text_review = text_reviews[text_reviews_idx]

            if isinstance(text_review, str):
                # Check and remove HTML tags
                has_html = bool(re.search('<.*?>', text_review))
                if has_html == True:
                    pass

                no_html_review = re.sub('<.*?>', ' ', text_review)
            
                has_url = bool(re.search(r'http\S+', no_html_review))
                if has_url == True:
                    pass

                no_html_url_review = re.sub(r'http\S+', '', no_html_review)
                cleaned_reviews.append(no_html_url_review)
            else:
                # print(text_reviews_idx, text_review)
                cleaned_reviews.append(text_review)
        self.df[col_name] = cleaned_reviews
        return self.df
    
    def remove_contractions(self, col_name: str):
        """Remove contractions from all reviews

        Parameters:
        -----------
        col_name: `str`
            Column with reviews

        Return:
        -------
        df: `pd.DataFrame`
            An updated DataFrame with the extra spaces removed
        """
        
        without_contractions_reviews = []
        text_reviews = self.get_reviews(col_name)

        for text_reviews_idx in range(len(text_reviews)):
            text_review = text_reviews[text_reviews_idx]

            # print("Review", text_reviews_idx, "with possible contraction(s) -- ", text_review)

            without_contraction = self.locate_and_replace_contractions(text_review)

            # print("Review", text_reviews_idx, "without contraction -- ", without_contraction)
            # print()

            without_contractions_reviews.append(without_contraction)

        self.df[col_name] = without_contractions_reviews
        return self.df
    
    def remove_non_alphabetical_characters(self, col_name: str):
        """Remove Non-alphabetical characters from all reviews

        Parameters:
        -----------
        col_name: `str`
            Column with reviews

        Return:
        -------
        df: `pd.DataFrame`
            An updated DataFrame with the non-alphabetical characters removed
        """

        alphabetical_char_reviews = []
        text_reviews = self.get_reviews(col_name)
        for text_reviews_idx in range(len(text_reviews)):
            text_review = text_reviews[text_reviews_idx]
            
            if isinstance(text_review, str):

                # Check for non-alphabetical characters
                has_non_alphabetical_char = bool(re.search(r'[^a-zA-Z]', text_review))
                if has_non_alphabetical_char == True:
                    # print("Review", text_reviews_idx, "has HTML -- ", text_review)
                    pass
                
                # Remove non-alphabetical characters
                with_alphabetical_char = re.sub(r'[^a-zA-Z\s]', ' ', text_review)
                # print("Review", text_reviews_idx, "has HTML -- ", with_alphabetical_char)
                alphabetical_char_reviews.append(with_alphabetical_char)
            else:
                alphabetical_char_reviews.append(text_review)

        self.df[col_name] = alphabetical_char_reviews
        return self.df
    
    def remove_symbols(self, col_name: str):
        """Remove symbols (except for % and $) from all text

        Parameters:
        -----------
        col_name: `str`
            Column with reviews

        Return:
        -------
        df: `pd.DataFrame`
            An updated DataFrame with the non-alphabetical characters removed
        """

        alphabetical_char_reviews = []
        text_reviews = self.get_reviews(col_name)
        for text_reviews_idx in range(len(text_reviews)):
            text_review = text_reviews[text_reviews_idx]
            
            if isinstance(text_review, str):

                # Check for non-alphabetical characters
                has_non_alphabetical_char = bool(re.search(r'[^a-zA-Z]', text_review))
                if has_non_alphabetical_char == True:
                    # print("Review", text_reviews_idx, "has HTML -- ", text_review)
                    pass
                
                # Remove non-alphabetical characters
                with_alphabetical_char = re.sub(r'[^a-zA-Z0-9\$\%\s]', ' ', text_review)
                # print("Review", text_reviews_idx, "has HTML -- ", with_alphabetical_char)
                alphabetical_char_reviews.append(with_alphabetical_char)
            else:
                alphabetical_char_reviews.append(text_review)

        self.df[col_name] = alphabetical_char_reviews
        return self.df
    
    def remove_extra_spaces(self, col_name: str):
        """Remove extra spaces from all reviews

        Parameters:
        -----------
        col_name: `str`
            Column with reviews

        Return:
        -------
        df: `pd.DataFrame`
            An updated DataFrame with the extra spaces removed
        """
        
        single_spaced_reviews = []
        text_reviews = self.get_reviews(col_name)

        for text_reviews_idx in range(len(text_reviews)):
            text_review = text_reviews[text_reviews_idx]

            if isinstance(text_review, str):
            # Check if there are any extra spaces
                has_extra_space = bool(re.search(r' +', text_review))
                if has_extra_space == True:
                    # print("Review", text_reviews_idx, "has extra space -- ", text_review)
                    pass
                
                # Remove extra spaces
                single_spaced_review = re.sub(r' +', ' ', text_review)
                # print("Review", text_reviews_idx, "without extra space -- ", single_spaced_review)
                # print()
                
                single_spaced_reviews.append(single_spaced_review)
            else:
                single_spaced_reviews.append(text_review)

        self.df[col_name] = single_spaced_reviews
        return self.df
    