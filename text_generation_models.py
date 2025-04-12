"""
Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)

Factory Method Design Pattern (https://refactoring.guru/design-patterns/factory-method/python/example#lang-features)
"""

import os, openai

import pandas as pd

from groq import Groq
from tqdm import tqdm
from typing import Dict, List
from dotenv import load_dotenv
from abc import ABC, abstractmethod

load_dotenv()  # Load environment variables from .env file

class TextGenerationModelFactory(ABC):
    """An abstract base class to load any pre-trained generation model"""
    
    def __init__(self):
        """Initialize the model with necessary parameters"""
        self.temperature = 0.3
        self.top_p = 0.9
        self.model_name = None
   

    def map_platform_to_api(self, platform_name: str):
        """
        Parameter:
        ----------
        platform_name : `str`
            Platform to use for generations.
        
        Returns:
        --------
        api_key : `str`
            The api key of specified platform.

        """
        platform_to_api_mappings = {
            "GROQ_CLOUD" : os.getenv('GROQ_CLOUD_API_KEY'),
            "NAVI_GATOR" : os.getenv('NAVI_GATOR_API_KEY')
        }

        api_key = platform_to_api_mappings.get(platform_name)
        
        if api_key is None:
            raise ValueError("API_KEY environment variable not set")
        
        return api_key
    
    @classmethod        
    def create_instance(self, model_name):

        if model_name == 'llama-3.3-70b-versatile':
            return LlamaVersatileTextGenerationModel()
        elif model_name == 'llama-3.1-8b-instant':
            return LlamaInstantTextGenerationModel()
        elif model_name == 'llama3-70b-8192':
            return Llama70B8192TextGenerationModel()
        elif model_name == 'llama3-8b-8192':
            return Llama8B8192TextGenerationModel()
        elif model_name == 'gpt-3.5-turbo':
            return OpenAiTextGenerationModel()
        else:
            raise ValueError(f"Unknown class name: {model_name}")


    def assistant(self, content: str) -> Dict:
        """Create an assistant message.
        
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
        """Create a user message.
        
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
    
    def chat_completion(self, messages: List[Dict]) -> str:
        """Generate a chat completion response.
        
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
        --------
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
    
    def generate_predictions(self, prompt_template: str, label: str, domain: str) -> pd.DataFrame:
        """Generate a completion response and return as a DataFrame.

        Parameters:
        -----------
        prompt_template: `str`
            The prompt template to generate a prediction prompt or a non-prediction prompt.
        
        label: `str`
            The prediction label for the prediction. Either 0 (non-prediction) or 1 (prediction).
        
        domain: `str`
            The domain of the prediction. As of now, the domains are finance, weather, health, and policy.

        template_number: `int`
            The template number to use for the prediction. For non-prediction prompts, the template number is 0 and for prediction prompts, the template number is 1 to 5.
        
        Returns:
        --------
        `pd.DataFrame`
            The generated completion response formatted as a DataFrame.
        """
        # Generate the raw prediction text
        raw_text = self.chat_completion([self.user(prompt_template)])
        
        # Parse the raw text into structured data (assuming a consistent format)
        predictions = []
        for line in raw_text.split("\n"):
            if line.strip():  # Skip empty lines
                predictions.append(line.strip())
        
        # Convert to DataFrame
        df = pd.DataFrame(predictions, columns=['Base Sentence'])
        df['Sentence Label'] = label
        df['Model Name'] = self.model_name
        df['Domain'] = domain
        return df

    def batch_generate_predictions(self, N_batches, text_generation_models, domains, prompt_outputs, sentence_label):
        all_batches_df = []
   
        for batch_idx in tqdm(range(N_batches)):
            batch_dfs = [] # Reset when starting a new batch

            # print(f"Batch ID: {batch_idx}")
            # print(f"Batch ID: {batch_idx}")
            for domain in domains:
                # print(f"    Domain: {domain}")            
                for text_generation_model in text_generation_models:
 
                    # print(f"Batch ID: {batch_idx} --- {text_generation_model}")

                    # print(f"Batch ID: {batch_idx} --- Domain: {domain} --- Model: {text_generation_model}")
                    # print(f"Batch ID: {batch_idx} --- {domain} --- {text_generation_model}")
                    print(f"{domain} --- {text_generation_model}")

                    prompt_output = prompt_outputs[domain]
                    model_df = text_generation_model.generate_predictions(prompt_output, label=sentence_label, domain=domain)
                    model_df["Batch Index"] = batch_idx

                    batch_dfs.append(model_df)
                print()
                
            print(f"====================================================================================")
      
            # Extend the main DataFrame list with the batch DataFrames
            all_batches_df.extend(batch_dfs)

        return all_batches_df    

    def __name__(self):
        pass

class LlamaVersatileTextGenerationModel(TextGenerationModelFactory):    
    def __init__(self):
        super().__init__()
        self.api_key = self.map_platform_to_api(platform_name="GROQ_CLOUD")
        self.client = Groq(api_key=self.api_key)
        self.model_name = "llama-3.3-70b-versatile"
    
    def __name__(self):
        return "llama-3.3-70b-versatile"

class LlamaInstantTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_key = self.map_platform_to_api(platform_name="GROQ_CLOUD")
        self.client = Groq(api_key=self.api_key)
        self.model_name = "llama-3.1-8b-instant"

    def __name__(self):
        return "llama-3.1-8b-instant"

class Llama70B8192TextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_key = self.map_platform_to_api(platform_name="GROQ_CLOUD")
        self.client = Groq(api_key=self.api_key)
        self.model_name = "llama3-70b-8192"

    def __name__(self):
        return "llama3-70b-8192"

class Llama8B8192TextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_key = self.map_platform_to_api(platform_name="GROQ_CLOUD")
        self.client = Groq(api_key=self.api_key)
        self.model_name = "llama3-8b-8192"
    
    def __name__(self):
        return "llama3-8b-8192"

class OpenAiTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_key = self.map_platform_to_api(platform_name="NAVI_GATOR")
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = "gpt-3.5-turbo"
    
    def __name__(self):
        return "gpt-3.5-turbo"




