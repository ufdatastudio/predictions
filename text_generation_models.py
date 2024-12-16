import os
from typing import Dict, List
from abc import ABC, abstractmethod
from groq import Groq





class TextGenerationModelFactory(ABC):
    """An abstract base class to load any pre-trained generation model"""
    
    @abstractmethod
    def __init__(self, api_key: str, model_name: str, prompt_template: str, temperature: float, top_p: float):
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
    def chat_completion(self, messages: List[Dict], model: str, temperature: float, top_p: float) -> str:
        """Generate a chat completion response"""
        pass
    
    @abstractmethod
    def completion(self) -> str:
        """Generate a completion response"""
        pass
    

class LlamaTextGenerationModel(TextGenerationModelFactory):
    """
    A class to interact with the LLaMA model.
    
    Attributes:
    -----------
    api_key : `str`
        The API key to authenticate the user.
    model_name : `str`
        The name of the LLaMA model, e.g., "llama-3.1-70b-versatile".
    prompt_template : `str`
        The prompt template to generate a prediction prompt or a non-prediction prompt.
    temperature : `float`
        The temperature parameter for the model.
    top_p : `float`
        The top_p parameter for the model.
    """
    
    def __init__(self, api_key: str, model_name: str, prompt_template: str, temperature: float, top_p: float):
        """
        Parameters:
        -----------
        api_key : `str`
            The API key to authenticate the user.
        model_name : `str`
            The name of the LLaMA model, e.g., "llama-3.1-70b-versatile".
        prompt_template : `str`
            The prompt template to generate a prediction prompt or a non-prediction prompt.
        temperature : `float`
            The temperature parameter for the model.
        top_p : `float`
            The top_p parameter for the model.
        """
        # Groq client
        self.client = Groq(api_key=api_key)
        self.api_key = api_key
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.top_p = top_p
    
    def assistant(self, content: str) -> Dict:
        """
        Create an assistant message.
        
        Parameters:
        -----------
        content : `str`
            The content of the assistant message.
        
        Returns:
        -------
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
        -------
        Dict
            A dictionary representing the user message.
        """
        return {"role": "user", "content": content}

    def chat_completion(self, messages: List[Dict]) -> str:
        """
        Generate a chat completion response.
        
        Parameters:
        -----------
        messages : `List[Dict]`
            A list of dictionaries representing the chat history.
        model : `str`
            The name of the model to use.
        temperature : `float`
            Sampling temperature.
        top_p : `float`
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
    
    def completion(self) -> str:
        """
        Generate a completion response.
        
        Returns:
        -------
        `str`
            The generated completion response.
        """
        return self.chat_completion([self.user(self.prompt_template)])