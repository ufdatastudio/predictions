"""
Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)

Factory Method Design Pattern (https://refactoring.guru/design-patterns/factory-method/python/example#lang-features)
"""

import os
import re
import json
import openai
import pathlib
import torch
import ipdb

import pandas as pd

from groq import Groq
from tqdm import tqdm
from typing import Dict, List

from dotenv import load_dotenv
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer


from log_files import LogData
from data_processing import DataProcessing
load_dotenv()  # Load environment variables from .env file

class TextGenerationModelFactory(ABC):
    """An abstract base class to load any pre-trained generation model"""
    
    def __init__(self):
        """In the init method (also called constructor), initialize our class with variables or attributes."""
        # Create instance variables or attributes
        # Standardized model parameters
        self.temperature = 0.6
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
            "GROQ_CLOUD" : os.getenv('GROQ_CLOUD_API_KEY'), # https://console.groq.com/docs/models
            "NAVI_GATOR" : os.getenv('NAVI_GATOR_API_KEY'), # https://it.ufl.edu/ai/navigator-toolkit/
            "HUGGING_FACE": os.getenv('HUGGING_FACE_API_KEY') # https://huggingface.co/models?pipeline_tag=text-generation&sort=trending
        }

        api_key = platform_to_api_mappings.get(platform_name)
        
        if api_key is None:
            raise ValueError("API_KEY environment variable not set")
        
        return api_key
    
    @classmethod        
    def create_instance(self, model_name):

        # Groq Cloud models
        # if model_name == 'distil-whisper-large-v3-en':
        #     return DistilWhisperLarge3TextGenerationModel()
        # if model_name == 'gemma2-9b-it':
        #     return Gemma29bTextGenerationModel()
        if model_name == 'llama-3.1-8b-instant':
            return LlamaInstantTextGenerationModel()
        elif model_name == 'llama-3.3-70b-versatile':
            return LlamaVersatileTextGenerationModel()
        elif model_name == 'openai/gpt-oss-120b':
            return GptOss120bTextGenerationModel()
        elif model_name == 'openai/gpt-oss-20b':
            return GptOss20bTextGenerationModel()
   
        # elif model_name == 'gpt-4o':
        #     return Gpt4oTextGenerationModel()
        elif model_name == 'llama-3.1-70b-instruct':
            return Llama3170BInstructTextGenerationModel()
        elif model_name == 'llama-3.1-8b-instruct':
            return Llama318BInstructTextGenerationModel()
        elif model_name == 'llama-3.1-nemotron-nano-8B-v1':
            return Llama31NemotronNano8BTextGenerationModel()
        elif model_name == 'llama-3.3-70b-instruct':
            return Llama3370BInstructTextGenerationModel()
        elif model_name == 'mistral-7b-instruct':
            return Mistral7BInstructTextGenerationModel()     
        elif model_name == 'mistral-small-3.1':
            return MistralSmall31TextGenerationModel()
        elif model_name == 'codestral-22b':
            return Codestral22BTextGenerationModel()
        elif model_name == 'gemma-3-37b-it':
            return Gemma337bItTextGenerationModel()
        elif model_name == 'gpt-oss-20b':
            return GptOss20TextGenerationModel()
        elif model_name == 'gpt-oss-120b':
            return GptOss120TextGenerationModel()
        elif model_name == 'granite-3.3-8b-instruct':
            return Granite338BInstructTextGenerationModel()
        elif model_name == 'sfr-embedding-mistral':
            return SfrEmbeddingMistralTextGenerationModel()
        elif model_name == 'nomic-embed-text-v1.5':
            return NomicEmbedTextV15TextGenerationModel()
        elif model_name == 'flux.1-dev':
            return Flux1DevTextGenerationModel()
        elif model_name == 'flux.1-schnell':
            return Flux1SchnellTextGenerationModel()
        elif model_name == 'whisper-large-v3':
            return WhisperLargeV3TextGenerationModel()
        elif model_name == 'kokoro':
            return KokoroTextGenerationModel()

        # Hugging Face models
        # elif model_name == 'DeepSeek-Prover-V2-7B':
        #     return DeepSeekProverV2TextGenerationModel()
        else:
            raise ValueError(f"Unknown class name: {model_name}")
    @classmethod
    def create_instances(self, model_names=None):
        """
        Create multiple model instances.
        
        Args:
            model_names: List of model names to create, or None for all models
            
        Returns:
            List of model instances
        """
        if model_names is None:
            # Return all available models
            model_names = self.get_all_model_names()
        
        models = []
        for model_name in model_names:
            try:
                models.append(self.create_instance(model_name))
            except ValueError as e:
                print(f"Warning: {e}")
        
        return models

    @classmethod
    def get_all_model_names(self):
        """Return list of all available model names"""
        return [
            # Groq Cloud models
            'distil-whisper-large-v3-en',
            'gemma2-9b-it',
            'llama-3.1-8b-instant',
            'llama-3.3-70b-versatile',
            'meta-llama/llama-guard-4-12b',
            'whisper-large-v3',
            'whisper-large-v3-turbo',
            # NaviGator models
            'gpt-oss-120b',
            'gpt-4o',
            'llama-3.1-70b-instruct',
            'llama-3.3-70b-instruct',
            'mixtral-8x7b-instruct',
            'llama-3.1-8b-instruct',
            'mistral-7b-instruct',
            'mistral-small-3.1'
        ]
    
    @classmethod
    def get_groq_model_names(self):
        """Return list of Groq Cloud model names"""
        return [
            'openai/gpt-oss-120b',
            'openai/gpt-oss-20b',
            'whisper-large-v3',
            'whisper-large-v3-turbo'
        ]
    
    @classmethod
    def get_navigator_model_names(self):
        """Return list of NaviGator model names"""
        return [
            'llama-3.1-70b-instruct',
            'llama-3.1-8b-instruct',
            # 'llama-3.1-nemotron-nano-8B-v1', BadRequestError: Error code: 400 - {'error': {'message': "{'error': '/chat/completions: Invalid model name passed in model=llama-3.1-nemotron-nano-8B-v1. Call `/v1/models` to view available models for your key.'}", 'type': 'None', 'param': 'None', 'code': '400'}}
            'llama-3.3-70b-instruct',
            'mistral-7b-instruct',
            'mistral-small-3.1',
            'codestral-22b',
            'gemma-3-27b-it',
            'gpt-oss-20b',
            'gpt-oss-120b',
            'granite-3.3-8b-instruct',
            # 'sfr-embedding-mistral', # NotFoundError: Error code: 404 - {'error': {'message': "litellm.NotFoundError: NotFoundError: OpenAIException - Error code: 404 - {'detail': 'Not Found'}. Received Model Group=sfr-embedding-mistral\nAvailable Model Group Fallbacks=None", 'type': None, 'param': None, 'code': '404'}}
            # 'nomic-embed-text-v1.5', # NotFoundError: Error code: 404 - {'error': {'message': "litellm.NotFoundError: NotFoundError: OpenAIException - Error code: 404 - {'detail': 'Not Found'}. Received Model Group=nomic-embed-text-v1.5\nAvailable Model Group Fallbacks=None", 'type': None, 'param': None, 'code': '404'}}
            # 'flux.1-dev', # NotFoundError: Error code: 404 - {'error': {'message': "litellm.NotFoundError: NotFoundError: OpenAIException - Error code: 404 - {'detail': 'Not Found'}. Received Model Group=flux.1-dev\nAvailable Model Group Fallbacks=None", 'type': None, 'param': None, 'code': '404'}}
            # 'flux.1-schnell', # NotFoundError: Error code: 404 - {'error': {'message': "litellm.NotFoundError: NotFoundError: OpenAIException - Error code: 404 - {'detail': 'Not Found'}. Received Model Group=flux.1-schnell\nAvailable Model Group Fallbacks=None", 'type': None, 'param': None, 'code': '404'}}
            # 'whisper-large-v3', # NotFoundError: Error code: 404 - {'error': {'message': "litellm.NotFoundError: NotFoundError: OpenAIException - Error code: 404 - {'detail': 'Not Found'}. Received Model Group=whisper-large-v3\nAvailable Model Group Fallbacks=None", 'type': None, 'param': None, 'code': '404'}}
            # 'kokoro' # NotFoundError: Error code: 404 - {'error': {'message': "litellm.NotFoundError: NotFoundError: OpenAIException - Error code: 404 - {'detail': 'Not Found'}. Received Model Group=kokoro\nAvailable Model Group Fallbacks=None", 'type': None, 'param': None, 'code': '404'}}

        ]

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
    
    def generate_predictions(self, prompt_template: str, label: str, domain: str, batch_id: int) -> pd.DataFrame:
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
        # print(f"\n  prompt_template: \n{prompt_template}\n\n")
        raw_text = self.chat_completion([self.user(prompt_template)])
        # print(f"    {self.model_name} + {domain} generates: {raw_text}")
        print(f"generates:\n{raw_text}")
        
        
        # Parse the raw text into structured data (assuming a consistent format)
        predictions = []
        for line in raw_text.split("\n"):
            if line.strip():  # Skip empty lines
                predictions.append(line.strip())
        
        # Convert to DataFrame
        df = pd.DataFrame(predictions, columns=['Base Sentence'])
        df['Sentence Label'] = label
        df['Domain'] = domain
        df['Model Name'] = self.model_name
        df['API Name'] = self.api_name
        df['Batch ID'] = batch_id
        # print()
        # print(df)
        # ipdb.set_trace()
        # print()
        return df

    def log_batch_df(self, reformat_batch_predictions_df, sentence_label, save_path: str):
        print("Start logging batch")

        base_path = pathlib.Path(__file__).parent.resolve()
        
        prediction_files = None
        if sentence_label == 0:
            prediction_files = "observation"
        elif sentence_label == 1:
            prediction_files = "prediction"
        else:
            print("Invalid sentence label. It should be 0 (non-prediction) or 1 (prediction).")
            quit()


        log_file_path = f"{save_path}/{prediction_files}_logs"
        log_directory = os.path.join(base_path, log_file_path)
        print(f"log_directory: {log_directory}")
        
        n = 1
        save_batch_directory = os.path.join(log_directory, f"batch_{n}-{prediction_files}")
    
        while os.path.exists(save_batch_directory):
            n += 1
            save_batch_directory = os.path.join(log_directory, f"batch_{n}-{prediction_files}")

        os.makedirs(save_batch_directory)
        save_batch_name = f"batch_{n}-info.log"
        save_from_df_name = f"batch_{n}-from_df.csv"
        save_from_csv_name = f"batch_{n}-from_csv.log"
    
        logger = LogData(base_path, log_file_path, save_batch_directory, save_batch_name)
        logger.dataframe_to_csv(reformat_batch_predictions_df, save_from_df_name)
        logger.csv_to_log(save_from_df_name, save_from_csv_name)

    def batch_generate_data(self, N_batches, text_generation_models, domains, prompt_outputs, sentence_label, save_path: str):
        """Generate a completion response and return as a DataFrame.

        Parameters:
        -----------
        N_batches: `int`
            The number of batches
        
        text_generation_models: `list`
            The models to use to generate data
        
        domains: `str`
            The domain of the prediction. As of now, the domains are finance, weather, health, and policy.

        prompt_outputs: `dict`
            Langchain dictionary to map the domain to which prompt to use for generation.

        sentence_label: `int`
            The prediction label for the prediction. Either 0 (non-prediction) or 1 (prediction).

        Returns:
        --------
        `pd.DataFrame`
            The generated completion response formatted as a DataFrame.
    """

        all_batches_df = []   
        for batch_idx in tqdm(range(N_batches)):
            print(f"===================================== Batch {batch_idx} ===============================================")

            batch_dfs = [] # Reset when starting a new batch

            for domain in domains:
                # print(f"    Domain: {domain}")            
                for text_generation_model in text_generation_models:
 
                    print(f"{domain} --- {text_generation_model.__name__()} --- {text_generation_model.api_name}")

                    prompt_output = prompt_outputs[domain]
                    model_df = text_generation_model.generate_predictions(prompt_output, label=sentence_label, domain=domain, batch_id=batch_idx)

                    batch_dfs.append(model_df)
                    batch_predictions_df = DataProcessing.concat_dfs(batch_dfs)
                    reformat_batch_predictions_df = DataProcessing.reformat_df_with_template_number(batch_predictions_df, col_name="Base Sentence")
                print()

                # print(f"NEW DOMAIN: {domain}")
            # ipdb.set_trace()

            self.log_batch_df(reformat_batch_predictions_df, sentence_label, save_path)
            # print(reformat_batch_predictions_df)

            # Extend the main DataFrame list with the batch DataFrames
            all_batches_df.append(reformat_batch_predictions_df)
        # print(all_batches_df)
        updated_all_batches_df = DataProcessing.concat_dfs(all_batches_df)
        return updated_all_batches_df    

    def __name__(self):
        pass


# class DistilWhisperLarge3TextGenerationModel(TextGenerationModelFactory):
#     def __init__(self):
#         super().__init__()
#         self.api_name = "GROQ_CLOUD"
#         self.api_key = self.map_platform_to_api(platform_name=self.api_name)
#         self.client = Groq(api_key=self.api_key)
#         self.model_name = self.__name__()

#     def __name__(self):
#         return "distil-whisper-large-v3-en"
    
# class Gemma29bTextGenerationModel(TextGenerationModelFactory):
#     def __init__(self):
#         super().__init__()
#         self.api_name = "GROQ_CLOUD"
#         self.api_key = self.map_platform_to_api(platform_name=self.api_name)
#         self.client = Groq(api_key=self.api_key)
#         self.model_name = self.__name__()

#     def __name__(self):
#         return "gemma2-9b-it"
    
class LlamaInstantTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "GROQ_CLOUD"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = Groq(api_key=self.api_key)
        self.model_name = self.__name__()

    def __name__(self):
        return "llama-3.1-8b-instant"
    
class LlamaVersatileTextGenerationModel(TextGenerationModelFactory):    
    def __init__(self):
        super().__init__()
        self.api_name = "GROQ_CLOUD"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = Groq(api_key=self.api_key)
        self.model_name = self.__name__()
    
    def __name__(self):
        return "llama-3.3-70b-versatile"

class GptOss120bTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "GROQ_CLOUD"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = Groq(api_key=self.api_key)
        self.model_name = self.__name__()

    def __name__(self):
        return "openai/gpt-oss-120b"
    
class GptOss20bTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "GROQ_CLOUD"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = Groq(api_key=self.api_key)
        self.model_name = self.__name__()

    def __name__(self):
        return "openai/gpt-oss-20b"

class WhisperLarge3TextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "GROQ_CLOUD"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = Groq(api_key=self.api_key)
        self.model_name = self.__name__()
    
    def __name__(self):
        return "whisper-large-v3"

class WhisperLarge3TurboTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "GROQ_CLOUD"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = Groq(api_key=self.api_key)
        self.model_name = self.__name__()
    
    def __name__(self):
        return "whisper-large-v3-turbo"

class GptOss120TextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = self.__name__()
    
    def __name__(self):
        return "gpt-oss-120b"

class Gpt4oTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = self.__name__()
    
    def __name__(self):
        return "gpt-4-turbo"

class Llama3170BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "llama-3.1-70b-instruct"

class Llama3370BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "llama-3.3-70b-instruct"

class Mixtral87BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = self.__name__()
    
    def __name__(self):
        return "mixtral-8x7b-instruct"    

class Llama318BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "llama-3.1-8b-instruct"

class Mistral7BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "mistral-7b-instruct"
    
class MistralSmall31TextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "mistral-small-3.1"

class Codestral22BTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" 
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "codestral-22b"

class NomicEmbedTextV15TextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" 
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "nomic-embed-text-v1.5"

class SfrEmbeddingMistralTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu" 
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "sfr-embedding-mistral"
    
class GtelargeEnV15TextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu"
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "gte-large-en-v1.5"

class WhisperLargeV3TextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu"
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "whisper-large-v3"
    
class Flux1SchnellTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu"
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "flux.1-schnell"
    

class Flux1DevTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu"
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "flux.1-dev"
    

class Granite338BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu"
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "granite-3.3-8b-instruct"

class Llama31NemotronNano8BTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu"
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "llama-3.1-nemotron-nano-8B-v1"
    

class Gemma337bItTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu"
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "gemma-3-27b-it"
    


class GptOss20TextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu"
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "gpt-oss-20b"
    
class KokoroTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key= self.api_key,
            base_url="https://api.ai.it.ufl.edu"
            )
        self.model_name = self.__name__()

    def __name__(self):
        return "kokoro"
    
def parse_json_response(response):
    """Parse JSON response from LLM to extract label and reasoning"""
    
    try:
        # Extract JSON if there's extra text
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get('label'), data.get('reasoning')
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None, None

def llm_classify_text(data: str, base_prompt: str, model):
    errors = {}
    prompt = f""" Given this: {base_prompt}. Also given the sentence '{data}', your task is to analyze the sentence and determine if it is a prediction. If prediction, generate label as 1 and if non-prediction generate label as 0.
    Respond ONLY with valid JSON in this exact format:
    {{"label": 0, "reasoning": "your explanation here"}}
    Examples:
    - "It will rain tomorrow." → {{"label": 1, "reasoning": "Contains the future tense words 'will' and 'tomorrow'"}}
    - "The stock market is expected to rise next quarter." → {{"label": 1, "reasoning": "Contains future tense words 'is expected'"}}
    - "I am going to the store." → {{"label": 0, "reasoning": "Does not contain a future tense word"}}
    - "Lakers will win the championship." → {{"label": 1, "reasoning": "Contains the future tense word 'will'"}}
    """

    idx = 1
    if idx == 1:
        #   print(f"\tPrompt: {prompt}")
            idx = idx + 1
    input_prompt = model.user(prompt)
    raw_text_llm_generation = model.chat_completion([input_prompt])
    
    try: 
        # Parse the JSON response
        label, reasoning = parse_json_response(raw_text_llm_generation)
        return raw_text_llm_generation, label, reasoning
    except Exception as e:
        print(f"Error: {e}")
        errors[data] = e

        