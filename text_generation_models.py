"""
Detravious Jamari Brinkley, Kingdom Man (https://brinkley97.github.io/expertise_and_portfolio/research/researchIndex.html)
UF Data Studio (https://ufdatastudio.com/) with advisor Christan E. Grant, Ph.D. (https://ceg.me/)

Factory Method Design Pattern (https://refactoring.guru/design-patterns/factory-method/python/example#lang-features)
"""

import os
import re
import json
import time
import openai
import pathlib


import pandas as pd
from datetime import date

from groq import Groq
from tqdm import tqdm
from typing import Dict, List

from dotenv import load_dotenv
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

from log_files import LogData
from data_processing import DataProcessing
load_dotenv()  # Load environment variables from .env file

class TextGenerationModelFactory(ABC):
    """An abstract base class to load any pre-trained generation model"""
    
    def __init__(self, temperature: float = 0.8, top_p: float = 0.9):
        """In the init method (also called constructor), initialize our class with variables or attributes.
        
        Notes:
            Temperature is like a coach deciding how risky the play call is. In NLP, after the model scores every possible next word, 
                temperature scales those scores before sampling. 
                    A low temperature (e.g., 0.1) makes the gap between "will" and "may" 
                enormous, so the model almost always picks "will". 
                    A high temperature (e.g., 1.5) flattens those scores so "will", "may", 
                "could", and even "shall" all look equally attractive, making the output more surprising.


            Top-P is like the coach deciding who is even dressed out for the game. In NLP, instead of scaling scores, Top-P cuts the 
                vocabulary. 
                    With top_p=0.1, the model only considers the smallest set of words whose combined probability adds up to 10% — 
                likely just 2-3 dominant words like "will" and "may". 
                    With top_p=0.95, the eligible pool expands to hundreds of words, 
                including rare but valid ones like "anticipates" or "envisions."

        **Low temp + High top-p:** Large roster dressed out, but coach almost always runs the ball with the star player anyway. Many words are eligible, but the model still heavily favors the most probable one.
        **High temp + Low top-p:** Tiny roster of 2-3 players, but coach randomizes who touches the ball every play. Few words are eligible, but the model picks unpredictably among them.

        temperature=0.1, top_p=0.1 — a very conservative, deterministic setting that keeps predictions grounded and realistic.
        temperature=0.1, top_p=0.6 — use words in sentence, while reasoning about structure and phrasing (ie: Rep. Jasmine Crockett, a congresswoman from Texas)
        temperature=0.8, top_p=0.9 — more flexible, which helps the model reason across varied sentence structures and phrasing.


        """
        self.temperature = temperature
        self.top_p = top_p
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
            "GROQ_CLOUD": os.getenv('GROQ_CLOUD_API_KEY'),   # https://console.groq.com/docs/models
            "NAVI_GATOR": os.getenv('NAVI_GATOR_API_KEY'),   # https://it.ufl.edu/ai/navigator-toolkit/
            "HUGGING_FACE": os.getenv('HUGGING_FACE_API_KEY') # https://huggingface.co/models?pipeline_tag=text-generation&sort=trending
        }
        api_key = platform_to_api_mappings.get(platform_name)
        
        if api_key is None:
            raise ValueError("API_KEY environment variable not set")
        
        return api_key

    @classmethod
    def create_instance(cls, model_name):
        # -------------------------------------------------------
        # Groq Cloud models
        # -------------------------------------------------------
        if model_name == 'llama-3.1-8b-instant':
            return LlamaInstantTextGenerationModel()
        elif model_name == 'llama-3.3-70b-versatile':
            return LlamaVersatileTextGenerationModel()
        elif model_name == 'openai/gpt-oss-120b':
            return OpenAIGptOss120bTextGenerationModel()
        elif model_name == 'openai/gpt-oss-20b':
            return OpenAIGptOss20bTextGenerationModel()
        elif model_name == 'whisper-large-v3':
            return WhisperLarge3TextGenerationModel()
        elif model_name == 'whisper-large-v3-turbo':
            return WhisperLarge3TurboTextGenerationModel()
        
        # -------------------------------------------------------
        # NaviGator models
        # -------------------------------------------------------
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
        elif model_name == 'gemma-3-27b-it':
            return Gemma337bItTextGenerationModel()
        elif model_name == 'gpt-oss-20b':
            return GptOss20bTextGenerationModel()
        elif model_name == 'gpt-oss-120b':
            return GptOss120bTextGenerationModel()
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
        else:
            raise ValueError(f"Unknown class name: {model_name}")

    @classmethod
    def create_instances(cls, model_names=None):
        """
        Create multiple model instances.
        
        Args:
            model_names: List of model names to create, or None for all models
            
        Returns:
            Dict of {model_name: model_instance}
        """
        if model_names is None:
            model_names = cls.get_all_model_names()
        
        models = {}
        for model_name in model_names:
            try:
                models[model_name] = cls.create_instance(model_name)
            except ValueError as e:
                print(f"Warning: {e}")
        return models

    @classmethod
    def get_all_model_names(cls):
        return cls.get_groq_model_names() + cls.get_navigator_model_names()

    @classmethod
    def get_groq_model_names(cls):
        return [
            'llama-3.1-8b-instant',
            'llama-3.3-70b-versatile',
            'openai/gpt-oss-120b',
            'openai/gpt-oss-20b',
            'whisper-large-v3',
            'whisper-large-v3-turbo',
        ]

    @classmethod
    def get_navigator_model_names(cls):
        return [
            'llama-3.1-70b-instruct',
            'llama-3.1-8b-instruct',
            'llama-3.3-70b-instruct',
            'mistral-7b-instruct',
            'mistral-small-3.1',
            'codestral-22b',
            'gemma-3-27b-it',
            'gpt-oss-20b',
            'gpt-oss-120b',
            'granite-3.3-8b-instruct',
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
    
    def chat_completion(self, messages: List[Dict], max_tokens: int = None) -> str:
        """Generate a chat completion response.
        
        Parameters:
        -----------
        messages: `List[Dict]`
            A list of dictionaries representing a single, standalone conversation per sentence.
            Each sentence is processed independently with no memory of previous sentences.

            Process per sentence, no follow-up or reprompting:

                [{"role": "user", "content": "Extract properties from: 'Apple stock will rise in Q3 2025'"}]
                
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
    
    def safe_chat_completion(self, messages: List[Dict], idx: int = 0, wait_time: int = 200, max_attempts: int = 3) -> str | None:
        """
        Wrap chat_completion with retry logic and rate limit handling.
        Returns None if all attempts fail, letting the caller decide how to handle it.

        Parameters
        ----------
        messages : List[Dict]
            A list of dictionaries representing a single, standalone conversation per sentence.
            Each sentence is processed independently with no memory of previous sentences.

            Process per sentence, no follow-up or reprompting (your pipeline)::

                [{"role": "user", "content": "Extract properties from: 'Apple stock will rise in Q3 2025'"}]

            Process per sentence, with follow-up or reprompting (not used in this pipeline)::

                [{"role": "user", "content": "Extract properties from: 'Apple stock will rise in Q3 2025'"},
                {"role": "assistant", "content": "{'0': [], '1': ['Apple'], '2': [], '3': ['Q3 2025'], '4': ['rise']}"},
                {"role": "user", "content": "Are you sure about the source?"}]

        idx : int, optional
            The index of the current sentence being processed, by default 0.
            Used for logging purposes.
        wait_time : int, optional
            Fallback seconds to wait if Groq's suggested wait time cannot be parsed,
            by default 200. Long enough for token bucket to refill for ~10 sentences.
        max_attempts : int, optional
            Maximum number of retry attempts per sentence before giving up, by default 3.

        Returns
        -------
        str or None
            The generated response string if successful, or None if all attempts fail.

        Notes
        -----
        For each sentence, the pipeline will:
            1. Try to call the model and get a response.
            2. If it fails, parse Groq's suggested wait time from the error message.
            Handles both TPM (e.g., 8.5s) and TPD (e.g., 7m21.936s) rate limits.
            3. If it fails a second time, wait again and try one last time.
            4. If all 3 attempts fail, return None and move on to the next sentence.

        The caller is responsible for handling None, e.g., recording ERROR_MAX_RETRIES
        in the results CSV so the resume logic skips it on the next run.

        Examples
        --------
        >>> messages = [model.user("Extract properties from: 'Apple stock will rise in Q3 2025'")]
        >>> response = model.safe_chat_completion(messages, idx=0)
        >>> if response is None:
        ...     print("Failed to get response, recording error and moving on.")
        """
        attempt = 0

        while attempt < max_attempts:
            try:
                return self.chat_completion(messages)

            except Exception as e:
                error_msg = str(e).lower()
                attempt += 1
                print(f"Attempt {attempt}/{max_attempts} failed for index {idx}. Error: {e}")

                if "rate limit" in error_msg or "429" in error_msg:

                    # Try to parse "7m21.936s" format first (TPD limit)
                    match_minutes = re.search(r'try again in (\d+)m(\d+\.?\d*)s', str(e))

                    # Then try "8.5875s" format (TPM limit)
                    match_seconds = re.search(r'try again in (\d+\.?\d*)s', str(e))

                    if match_minutes:
                        minutes = float(match_minutes.group(1))
                        seconds = float(match_minutes.group(2))
                        actual_wait = (minutes * 60) + seconds + 5  # 5s buffer
                    elif match_seconds:
                        actual_wait = float(match_seconds.group(1)) + 5  # 5s buffer
                    else:
                        actual_wait = wait_time

                    print(f"Rate limit hit. Waiting {actual_wait:.1f}s...")
                    time.sleep(actual_wait)

                elif "badrequesterror" in error_msg:
                    print(f"Bad Request. Stopping retry for index {idx}.")
                    break

                else:
                    time.sleep(5)

        return None
    
    def generate_predictions(self, prompt_template: str, label: str, domain: str, batch_id: int, prediction_date: datetime) -> pd.DataFrame:
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
        
        prediction_date: `datetime`
            The date of which the prediction was created. 
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
        df['Temperature'] = self.temperature
        df['Top P'] = self.top_p
        df['Prompt Used'] = prompt_template
        df['Source'] = 1
        df['Target'] = 1
        df['Prediction Date'] = 1
        df['Generation Date'] = prediction_date
        df['Outcome'] = 1
        df['Raw Text'] = raw_text
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

        df_to_save = reformat_batch_predictions_df.copy()
        if "Prompt Used" in df_to_save.columns:
            df_to_save["Prompt Used"] = (
                df_to_save["Prompt Used"]
                .astype(str)
                .str.replace("\r\n", " ")
                .str.replace("\n", " ")
                .str.replace("\r", " ")
            )
        logger = LogData(base_path, log_file_path, save_batch_directory, save_batch_name)
        logger.dataframe_to_csv(df_to_save, save_from_df_name)
        logger.csv_to_log(save_from_df_name, save_from_csv_name)

    def batch_generate_data(self, N_batches, text_generation_models, domains, prompt_outputs, sentence_label, save_path: str, batch_prediction_date: datetime, prediction_templates: list):
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
        
        batch_prediction_date: `datetime`
            The date of which the prediction was created. 

        Returns:
        --------
        `pd.DataFrame`
            The generated completion response formatted as a DataFrame.
    """
        # Apply factory's temperature and top_p to all models for this run
        for m in text_generation_models:
            if hasattr(m, 'temperature'):
                m.temperature = self.temperature
            if hasattr(m, 'top_p'):
                m.top_p = self.top_p
        all_batches_df = []   
        for batch_idx in tqdm(range(N_batches)):
            print(f"===================================== Batch {batch_idx} ===============================================")

            batch_dfs = [] # Reset when starting a new batch

            for domain in domains:
                # print(f"    Domain: {domain}")            
                for text_generation_model in text_generation_models:
 
                    print(f"{domain} --- {text_generation_model.__name__()} --- {text_generation_model.api_name}")

                    prompt_output = prompt_outputs[domain]
                    model_df = text_generation_model.generate_predictions(prompt_output, label=sentence_label, domain=domain, batch_id=batch_idx, prediction_date=batch_prediction_date)

                    batch_dfs.append(model_df)
                    batch_predictions_df = DataProcessing.concat_dfs(batch_dfs)
                    reformat_batch_predictions_df = DataProcessing.reformat_df_with_template_number(batch_predictions_df, prediction_templates, col_name="Base Sentence")
                print()

                # print(f"NEW DOMAIN: {domain}")
            # ipdb.set_trace()

            self.log_batch_df(reformat_batch_predictions_df, sentence_label, save_path)
            #print(reformat_batch_predictions_df)

            # Extend the main DataFrame list with the batch DataFrames
            all_batches_df.append(reformat_batch_predictions_df)
        # print(all_batches_df)
        updated_all_batches_df = DataProcessing.concat_dfs(all_batches_df)
        return updated_all_batches_df    

    def __name__(self):
        pass


# ================================================================
# GROQ CLOUD MODELS
# ================================================================
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

class OpenAIGptOss120bTextGenerationModel(TextGenerationModelFactory):
    """Groq-hosted version of gpt-oss-120b (accessed via openai/ prefix)."""
    def __init__(self):
        super().__init__()
        self.api_name = "GROQ_CLOUD"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = Groq(api_key=self.api_key)
        self.model_name = self.__name__()

    def __name__(self):
        return "openai/gpt-oss-120b"

class OpenAIGptOss20bTextGenerationModel(TextGenerationModelFactory):
    """Groq-hosted version of gpt-oss-20b (accessed via openai/ prefix)."""
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


# ================================================================
# NAVIGATOR MODELS
# ================================================================
class Llama3170BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"  # LiteLLM Proxy is OpenAI compatible
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "llama-3.1-70b-instruct"


class Llama318BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "llama-3.1-8b-instruct"


class Llama31NemotronNano8BTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "llama-3.1-nemotron-nano-8B-v1"


class Llama3370BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "llama-3.3-70b-instruct"


class Mistral7BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
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
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
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
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "codestral-22b"


class Gemma337bItTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "gemma-3-27b-it"  # <--- FIXED: was gemma-3-37b-it


class GptOss20bTextGenerationModel(TextGenerationModelFactory):
    """NaviGator-hosted version of gpt-oss-20b."""
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "gpt-oss-20b"


class GptOss120bTextGenerationModel(TextGenerationModelFactory):
    """NaviGator-hosted version of gpt-oss-120b."""
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"  # LiteLLM Proxy is OpenAI compatible
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "gpt-oss-120b"


class Granite338BInstructTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "granite-3.3-8b-instruct"


class SfrEmbeddingMistralTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "sfr-embedding-mistral"


class NomicEmbedTextV15TextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "nomic-embed-text-v1.5"


class Flux1DevTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "flux.1-dev"


class Flux1SchnellTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "flux.1-schnell"


class WhisperLargeV3TextGenerationModel(TextGenerationModelFactory):
    """NaviGator-hosted version of whisper-large-v3."""
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "whisper-large-v3"


class GtelargeEnV15TextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "gte-large-en-v1.5"


class KokoroTextGenerationModel(TextGenerationModelFactory):
    def __init__(self):
        super().__init__()
        self.api_name = "NAVI_GATOR"
        self.api_key = self.map_platform_to_api(platform_name=self.api_name)
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.ai.it.ufl.edu"
        )
        self.model_name = self.__name__()

    def __name__(self):
        return "kokoro"