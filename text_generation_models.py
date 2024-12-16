"""
Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)

Factory Method Design Pattern (https://refactoring.guru/design-patterns/factory-method/python/example#lang-features)
"""

import os

import pandas as pd

from typing import Dict, List
from abc import ABC, abstractmethod
from groq import Groq

class TextGenerationModelFactory(ABC):
    """An abstract base class to load any pre-trained generation model"""
    
    @abstractmethod
    def __init__(self, model_name: str, prompt_template: str, temperature: float, top_p: float):
        """Initialize the model with necessary parameters"""
        pass
    
    @abstractmethod
    def assistant(self, content: str) -> Dict:
        """Generate an assistant message"""
        pass
    
    @abstractmethod
    def user(self, content: str) -> Dict:
        """Create a user message"""
        pass
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict]) -> str:
        """Generate a chat completion response"""
        pass
    
    @abstractmethod
    def completion(self) -> pd.DataFrame:
        """Generate a completion response and return as a DataFrame"""
        pass

class LlamaTextGenerationModel(TextGenerationModelFactory):
    """
    A class to interact with the LLaMA model.
    
    Attributes:
    -----------
    api_key: `str`
        The API key to authenticate the user.
    model_name: `str`
        The name of the LLaMA model, e.g., "llama-3.1-70b-versatile".
    prompt_template: `str`
        The prompt template to generate a prediction prompt or a non-prediction prompt.
    temperature: `float`
        The temperature parameter for the model.
    top_p: `float`
        The top_p parameter for the model.
    """
    
    def __init__(self, model_name: str, prompt_template: str, temperature: float, top_p: float):
        """
        Parameters:
        -----------
        model_name: `str`
            The name of the LLaMA model, e.g., "llama-3.1-70b-versatile".
        prompt_template: `str`
            The prompt template to generate a prediction prompt or a non-prediction prompt.
        temperature: `float`
            The temperature parameter for the model.
        top_p: `float`
            The top_p parameter for the model.
        """
        # Groq client
        api_key = os.getenv('API_KEY')
        if api_key is None:
            raise ValueError("API_KEY environment variable not set")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.top_p = top_p
    
    def assistant(self, content: str) -> Dict:
        """
        Create an assistant message.
        
        Parameters:
        -----------
        content: `str`
            The content of the assistant message.
        
        Returns:
        --------
        Dict
            A dictionary representing the assistant message.
        """
        return {"role": "assistant", "content": content}

    def user(self, content: str) -> Dict:
        """
        Create a user message.
        
        Parameters:
        -----------
        content : `str`
            The content of the user message.
        
        Returns:
        --------
        Dict
            A dictionary representing the user message.
        """
        return {"role": "user", "content": content}

    def chat_completion(self, messages: List[Dict], temperature: float = 0.3, top_p: float = 0.9) -> str:
        """
        Generate a chat completion response.
        
        Parameters:
        -----------
        messages: `List[Dict]`
            A list of dictionaries representing the chat history.
        model: `str`
            The name of the model to use.
        temperature: `float`
            Sampling temperature.
        top_p: `float`
            Nucleus sampling parameter.
        
        Returns:
        -------
        `str`
            The generated chat completion response.
        """
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.choices[0].message.content
    
    def completion(self, cols_name) -> pd.DataFrame:
        """
        Generate a completion response and return as a DataFrame.

        Parameters:
        -----------
        cols_name: `List[str]`
            The column names for the DataFrame.
        
        Returns:
        --------
        `pd.DataFrame`
            The generated completion response formatted as a DataFrame.
        """
        # Generate the raw prediction text
        raw_text = self.chat_completion([self.user(self.prompt_template)])
        
        # Parse the raw text into structured data (assuming a consistent format)
        predictions = []
        for line in raw_text.split("\n"):
            if line.strip():  # Skip empty lines
                predictions.append(line.strip())
        
        # Convert to DataFrame
        df = pd.DataFrame(predictions, columns=cols_name)
        
        return df