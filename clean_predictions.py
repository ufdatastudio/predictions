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
    
    @abstractmethod
    def lower_case(self):
        """Convert all text to lower case"""
        pass
    
    @abstractmethod
    def remove_html_and_urls(self):
        """Remove HTML tags and URLs from the text"""
        pass
    
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
        """Convert all reviews to lower case

        Parameters
        ----------
        df: `pd.DataFrame`
            The data
        
        col_name: `str`
            Column with reviews

        Return
        ------
        df: `pd.DataFrame`
            An updated DataFrame with the lower cased reviews
        """
        
        lower_case_reviews = []
        updated_df = self.df.copy()
        text_reviews = self.df[col_name].values
        
        for text_reviews_idx in range(len(text_reviews)):
            text_review = text_reviews[text_reviews_idx]
            # print(text_reviews_idx, type(text_review), text_review)

            # NOT all reviews are strings, thus all can't be converted to lower cased
            if type(text_review) != str:
                converted_str = str(text_review)
                # update_text_review = converted_str.lower()
                lower_case_reviews.append(text_review)
                # print(text_reviews_idx, update_text_review)
                # print()
            else:
                update_text_review = text_review.lower()
                lower_case_reviews.append(update_text_review)
                # print(text_reviews_idx, update_text_review)
                # print()

        updated_df['lower_cased'] = lower_case_reviews
        return updated_df
    
    def remove_html_and_urls(self):
        # Placeholder for the method implementation
        pass
    
    def remove_contractions(self):
        # Placeholder for the method implementation
        pass
    
    def remove_non_alphabetical_characters(self):
        # Placeholder for the method implementation
        pass
    
    def remove_extra_spaces(self):
        # Placeholder for the method implementation
        pass