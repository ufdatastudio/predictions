# %% [markdown]
# # ML Classifiers
# 
# **Goal:** Given a sentence as input, classify it as either a prediction or non-prediction.

# %%
import os
import sys
import warnings

import pandas as pd

from tqdm import tqdm

# Get the current working directory of the notebook
notebook_dir = os.getcwd()
# Add the parent directory to the system path
sys.path.append(os.path.join(notebook_dir, '../'))

# import log_files
from data_processing import DataProcessing
from feature_extraction import SpacyFeatureExtraction
# from classification_models import SkLearnPerceptronModel, SkLearnSGDClassifier, EvaluationMetric
from classification_models import SkLearnModelFactory, EvaluationMetric

# %%
pd.set_option('max_colwidth', 800)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', None)

warnings.filterwarnings('ignore')

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
print(f"\tShape: {df.shape}, \nSubset of Data:{df.head(7)}")
df.shape, df.head(3)

# %% [markdown]
# ## Shuffle Data

# %%
print("======= SHUFFLE DATA =======")

# %%
shuffled_df = DataProcessing.shuffle_df(df)
print(f"\tShape: {shuffled_df.shape}, \nSubset of Data:{shuffled_df.head(7)}")

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
# spacy_sentence_embeddings_df.columns.

# %%


# %%
# print(f"{spacy_sentence_embeddings_df.head(3)}")
# spacy_sentence_embeddings_df
# print(f"{spacy_sentence_embeddings_df.to_dict()}")

for idx, row in spacy_sentence_embeddings_df.iterrows():
    text = row['Base Sentence']
    label = row['Sentence Label']
    embedding = row['Embedding']
    norm_embedding = row['Normalized Embeddings']
    if idx < 7:
        print(f"{idx}\n Sentence: {text}\n Label: {label}\n Embeddings Shape: {embedding.shape}\n\t Embeddings Subset [:6]: {embedding[:6]} \n Norm Embeddings: {norm_embedding.shape}, \n\tNorm Embeddings Subset [:6]: {norm_embedding[:6]}")

# %%


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
save_df = True

if save_df == True:
    print("Save test set so we can pass these into LLMs")
    save_path = os.path.join(base_data_path, 'combined_generated_fin_phrase_bank')
    DataProcessing.save_to_file(X_test_df, save_path, 'x_test_set', 'csv')
    DataProcessing.save_to_file(y_test_df, save_path, 'y_test_set', 'csv')

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
sklmf = SkLearnModelFactory
perception_model = sklmf.select_model('perceptron')
sgd_classifier_model = sklmf.select_model('sgd_classifier')
logistic_regression_model = sklmf.select_model('logistic_regression')
ridge_classifier_model = sklmf.select_model('ridge_classifier')
# linear_regression_model = sklmf.select_model('linear_regression')
# elastic_net_model = sklmf.select_model('elastic_net')

ml_models = [perception_model, sgd_classifier_model, logistic_regression_model, ridge_classifier_model]

# %%
models_with_predictions = {}
for ml_model in ml_models:
    print(f"Train -> Predict for {ml_model.get_model_name()}")
    ml_model.train_model(X_train_df[embeddings_col_name].to_list(), y_train_df)
    ml_model_predictions = ml_model.predict(X_test_df[embeddings_col_name].to_list())
    models_with_predictions[ml_model.get_model_name()] = ml_model_predictions

models_with_predictions

# %%
# models_predictions_df = pd.DataFrame(models_to_predictions)
# models_predictions_df

# %%
y_test_df.rename(index='Actual Label', inplace=True)

# %%
test_and_models_df = pd.concat([X_test_df.loc[:, :], y_test_df], axis=1)
# test_and_models_df = pd.concat([test_df, models_predictions_df])

for key, value in models_with_predictions.items():
    test_and_models_df[key] = value.to_numpy().ravel()

test_and_models_df.head(3)

# %% [markdown]
# ## Evaluation

# %%
print("======= EVALUATION/RESULTS =======")

# %%
get_metrics = EvaluationMetric()
get_metrics

# %% [markdown]
# > - Results may differ (from previous runs and even terminal runs) because we shuffle the data.

# %%
actual_label = test_and_models_df['Actual Label'].values
for ml_model in ml_models:
    ml_model_name = ml_model.get_model_name()
    print(f"Actual Label:\t\t{actual_label}")
    ml_model_predictions = test_and_models_df[ml_model_name].values
    print(f"{ml_model_name}:\t\t{ml_model_predictions}")
    print()
    get_metrics.eval_classification_report(y_test_df, ml_model_predictions)

# %%



