# %% [markdown]
# # Extract Embeddings: Combine Generated Synthetic Data with Financial PhraseBank Data

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
from feature_extraction import SpacyFeatureExtraction
from classification_models import SkLearnPerceptronModel, SkLearnSGDClassifier, EvaluationMetric

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
combine_data_path = os.path.join(base_data_path, 'combined_generated_fin_phrase_bank/combined_generated_fin_phrase_bank-v1.csv')

# %%
df = DataProcessing.load_from_file(combine_data_path, 'csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
print(f"\t{df.head(7)}")

# %% [markdown]
# ## Shuffle Data

# %%
print("======= SHUFFLE DATA =======")

# %%
shuffled_df = DataProcessing.shuffle_df(df)
shuffled_df

# %% [markdown]
# ## Extract Sentence Embeddings

# %%
print("======= EMBED SENTENCES: Spacy =======")

# %%
spacy_fe = SpacyFeatureExtraction(shuffled_df, 'Base Sentence')
spacy_fe

# %%
spacy_sentence_embeddings_df = spacy_fe.sentence_feature_extraction(attach_to_df=True)
# print(f"{spacy_sentence_embeddings_df.head(3)}")

# %% [markdown]
# ## Normalize Embeddings
# 
# - Why: Getting the below warnings
#     1. sklearn/utils/extmath.py:203: RuntimeWarning: divide by zero encountered in matmul ret = a @ b
#     2. sklearn/utils/extmath.py:203: RuntimeWarning: overflow encountered in matmul ret = a @ b
#     3. sklearn/utils/extmath.py:203: RuntimeWarning: invalid value encountered in matmul ret = a @ b
# 
# - Normalize will place data within "boundaries" to be all on one scale

# %%
print("======= NORMALIZE EMBEDDINGS =======")

# %%
from sklearn.preprocessing import StandardScaler

# Convert embeddings to matrix if not already
embeddings_matrix = pd.DataFrame(spacy_sentence_embeddings_df["Embedding"].tolist())

# Scale the embeddings
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings_matrix)

spacy_sentence_embeddings_df['Normalized Embeddings'] = list(scaled_embeddings)

# %%
# print(f"{spacy_sentence_embeddings_df.head(3)}")

# %%
embeddings_col_name = 'Normalized Embeddings'

# %% [markdown]
# ## Split Data

# %%
print("======= SPLIT DATA =======")

# %%
# spacy_embeds = spacy_sentence_embeddings_df['Embedding'].to_list()
labels_col = spacy_sentence_embeddings_df['Sentence Label']
X_train_df, X_test_df, y_train_df, y_test_df = DataProcessing.split_data(spacy_sentence_embeddings_df, labels_col)
# print(f"{X_train_df.head(3)}")

# %%
len(y_train_df)

# %%
X_train_df[embeddings_col_name].to_list()

# %% [markdown]
# ## Models

# %%
print("======= TRAIN x TEST MODELS =======")

# %% [markdown]
# > Track loss: try BCE (Binary Cross Entropy)

# %%
perception_model = SkLearnPerceptronModel()
perception_model.train_model(X_train_df[embeddings_col_name].to_list(), y_train_df)
perceptron_predictions = perception_model.predict(X_test_df[embeddings_col_name].to_list())
perceptron_predictions.to_numpy().ravel()

sgd_model = SkLearnSGDClassifier()
sgd_model.train_model(X_train_df[embeddings_col_name].to_list(), y_train_df)
sgd_predictions = perception_model.predict(X_test_df[embeddings_col_name].to_list())
sgd_predictions.to_numpy().ravel()

# %%
model_predictions_df = pd.concat([X_test_df['Base Sentence'], y_test_df], axis=1)
model_predictions_df.columns = ['Sentence', 'Actual Label']
model_predictions_df['Perceptron Predicted Label'] = perceptron_predictions.to_numpy().ravel()
model_predictions_df['SGD Predicted Label'] = sgd_predictions.to_numpy().ravel()
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
metrics = get_metrics.eval_classification_report(y_test_df, perceptron_predictions)
metrics

# %%
metrics = get_metrics.eval_classification_report(y_test_df, sgd_predictions)
metrics

# %%



