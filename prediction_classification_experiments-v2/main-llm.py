# %% [markdown]
# # LLM Classifiers
# 
# **Goal:** Given a sentence as input, classify it as either a prediction or non-prediction.

# %%
import os
import sys

import pandas as pd

from tqdm import tqdm

# Get the current working directory of the notebook
notebook_dir = os.getcwd()
# Add the parent directory to the system path
sys.path.append(os.path.join(notebook_dir, '../'))

# import log_files
from data_processing import DataProcessing
from classification_models import EvaluationMetric
from text_generation_models import TextGenerationModelFactory

# %%
pd.set_option('max_colwidth', 800)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# %% [markdown]
# ## Load Data

# %%
print("======= LOAD DATA =======")

# %%
base_data_path = os.path.join(notebook_dir, '../data/')
combine_data_path = os.path.join(base_data_path, 'combined_generated_fin_phrase_bank')
X_test_set_path = os.path.join(combine_data_path, 'x_test_set-v1.csv')
y_test_set_path = os.path.join(combine_data_path, 'y_test_set-v1.csv')

# %%
X_test_df = DataProcessing.load_from_file(X_test_set_path, 'csv')
X_test_df.drop(columns=['Unnamed: 0'], inplace=True)
X_test_df.head(7)

# %%
y_test_df = DataProcessing.load_from_file(y_test_set_path, 'csv')
y_test_df.drop(columns=['Unnamed: 0'], inplace=True)
print(f"\t{y_test_df.head(7)}")

# %% [markdown]
# ## Load Prompt

# %%
# prediction_properties = PredictionProperties.get_prediction_properties()
# prediction_requirements = PredictionProperties.get_requirements()
# system_identity_prompt = "You are an expert at identifying specific types of sentences by knowing the sentence format."
# prediction_examples_prompt = """Some examples of predictions in the PhraseBank dataset are
#     1. According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .
#     2. According to the company 's updated strategy for the years 2009-2012 , Basware targets a long-term net sales growth in the range of 20 % -40 % with an operating profit margin of 10 % -20 % of net sales .
#     3. Its board of directors will propose a dividend of EUR0 .12 per share for 2010 , up from the EUR0 .08 per share paid in 2009 .
# """
# non_prediction_examples_prompt = """Some examples of non-predictions in the PPhraseBank dataset are
#     1. Net sales increased to EUR193 .3 m from EUR179 .9 m and pretax profit rose by 34.2 % to EUR43 .1 m. ( EUR1 = USD1 .4 )
#     2. Net sales surged by 18.5 % to EUR167 .8 m. Teleste said that EUR20 .4 m , or 12.2 % , of the sales came from the acquisitions made in 2009 .
#     3. STORA ENSO , NORSKE SKOG , M-REAL , UPM-KYMMENE Credit Suisse First Boston ( CFSB ) raised the fair value for shares in four of the largest Nordic forestry groups .
# """
# # goal_prompt = "Given the above, identify the prediction."

# base_prompt = f"""{system_identity_prompt} The sentence format is based on: 
    
#     {prediction_properties}
#     Enforce: {prediction_requirements}
#     Know: {prediction_examples_prompt}
#     Know: {non_prediction_examples_prompt}

# """
# base_prompt

# %%
prompt_1 = """ 

Role: 
You are a linguist expert. You are acting as a prediction detector. Your task is to identify if a given sentence is a prediction about the future.

Background:
A prediction is a statement about what someone thinks will happen in the future.
Examples of predictions:
- "It will rain tomorrow."
- "The stock market is expected to rise next quarter."
- "I am going to the store."
- “Lakers will win the championship.”

A prediction may contain: source, target, date, outcome.
"""

# %% [markdown]
# ## Models

# %%
tgmf = TextGenerationModelFactory()


# Groq Cloud (https://console.groq.com/docs/overview)
llama_318b_instant_generation_model = tgmf.create_instance('llama-3.1-8b-instant') 
llama_3370b_versatile_generation_model = tgmf.create_instance('llama-3.3-70b-versatile')  

# models  = [llama_318b_instant_generation_model]
models  = [llama_318b_instant_generation_model, llama_3370b_versatile_generation_model]


# NaviGator
# llama_31_70b_instruct = tgmf.create_instance('llama-3.1-70b-instruct') 
# mistral_small_31 = tgmf.create_instance('mistral-small-3.1') 
# models = [llama_31_70b_instruct, mistral_small_31]

# %%
import json
import re

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

# %%
def llm_certifier(data: str, base_prompt: str, model):
    
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
        
        # Parse the JSON response
        label, reasoning = parse_json_response(raw_text_llm_generation)
        
        return raw_text_llm_generation, label, reasoning

# %%
print("======= PROMPT + MODEL -> LABEL and REASONING =======")

# %%
# content : meta :: text : meta_data
results = []
for idx, row in X_test_df.iterrows():
    text = row['Base Sentence']
    print('\t', idx, text)
    for model in models:
        raw_response, llm_label, llm_reasoning = llm_certifier(text, prompt_1, model)
        print('\t\t', model.__name__(), '--- Label:', llm_label, '--- Reasoning:', llm_reasoning)
        result = (text, raw_response, llm_label, llm_reasoning, model.__name__())
        results.append(result)


# %%
results

# %%
results_with_llm_label_df = pd.DataFrame(results, columns=['text', 'raw_response', 'llm_label', 'llm_reasoning', 'llm_name'])
results_with_llm_label_df

# %%
def get_llm_labels(df, model_name):
    filt_llama = (df['llm_name'] == model_name)
    filt_df = df[filt_llama]
    return filt_df['llm_label']

llama_instant_labels = get_llm_labels(results_with_llm_label_df, 'llama-3.1-8b-instant')
llama_versatile_labels = get_llm_labels(results_with_llm_label_df, 'llama-3.3-70b-versatile')
print(f"\tllama-3.1-8b-instant: {llama_instant_labels}")
print(f"\tllama-3.3-70b-versatile: {llama_versatile_labels}")

# %%
model_predictions_df = pd.concat([X_test_df['Base Sentence'], y_test_df], axis=1)
model_predictions_df.columns = ['Sentence', 'Actual Label']
model_predictions_df['Instant'] = llama_instant_labels.to_numpy().ravel()
model_predictions_df['Versatile Label'] = llama_versatile_labels.to_numpy().ravel()
model_predictions_df

# print(f"{model_predictions_df}")

# %% [markdown]
# ## Evaluation

# %%
print("======= EVALUATION/RESULTS =======")

# %%
get_metrics = EvaluationMetric()
get_metrics

# %%
metrics = get_metrics.eval_classification_report(y_test_df, llama_instant_labels)
metrics

# %%
metrics = get_metrics.eval_classification_report(y_test_df, llama_versatile_labels)
metrics

# %%



