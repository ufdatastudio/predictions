# %% [markdown]
# # ML Classifiers
# 
# **Goal:** Given a sentence as input, classify it as either a prediction or non-prediction.

# %% [markdown]
# > `prediction_classification_experiments-v2/create_dataset.ipynb` to combine model training and testing data.

# %%
import os
import sys
import warnings
import joblib


import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

# Get the current working directory of the notebook
notebook_dir = os.getcwd()
# Add the parent directory to the system path
sys.path.append(os.path.join(notebook_dir, '../'))

# import log_files
from data_processing import DataProcessing
from feature_extraction import SpacyFeatureExtraction
from classification_models import SkLearnModelFactory
from metrics import EvaluationMetric

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
base_data_path = os.path.join(notebook_dir, '../data')
combine_data_path = os.path.join(base_data_path, 'combined_datasets')
data_path = os.path.join(combine_data_path, 'combined-synthetic-fin_phrase_bank-v5.csv')

# %%
df = DataProcessing.load_from_file(data_path, 'csv', sep=',')
df

# %%
print(len(df))
# df.drop(columns=['Unnamed: 0'], inplace=True)
print(f"\tShape: {df.shape}, \nSubset of Data:{df.head(7)}")
df.shape, df.tail(3)

# %% [markdown]
# ## Shuffle Data

# %%
df.head(3)

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
spacy_sentence_embeddings_df = spacy_fe.sentence_embeddings_extraction(attach_to_df=True)
spacy_sentence_embeddings_df

# %% [markdown]
# ### Normalize Embeddings
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
# Convert embeddings to matrix if not already
embeddings_matrix = pd.DataFrame(spacy_sentence_embeddings_df["Base Sentence Embedding"].tolist())

# Scale the embeddings
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings_matrix)

spacy_sentence_embeddings_df['Normalized Embedding'] = list(scaled_embeddings)

# %%
# print(f"{spacy_sentence_embeddings_df.head(3)}")
# spacy_sentence_embeddings_df
# print(f"{spacy_sentence_embeddings_df.to_dict()}")

for idx, row in spacy_sentence_embeddings_df.iterrows():
    text = row['Base Sentence']
    label = row['Sentence Label']
    embedding = row['Base Sentence Embedding']
    norm_embedding = row['Normalized Embedding']
    if idx < 7:
        print(f"{idx}\n Sentence: {text}\n Label: {label}\n Embeddings Shape: {embedding.shape}\n\t Embeddings Subset [:6]: {embedding[:6]} \n Norm Embeddings: {norm_embedding.shape}, \n\tNorm Embeddings Subset [:6]: {norm_embedding[:6]}")

# %%
embeddings_col_name = 'Base Sentence Embedding'

# %% [markdown]
# ## Split Data
# 
# 
# > **Stratification preserves the original dataset ratio when splitting into train-test splits.**  
# > **Example:** If we have 1,000 samples: 920 non-predictions, 80 predictions  
# > - **Train size = 0.8 → 800 samples in train**  
# >     - **With stratify:** train ≈ 736 non-predictions, 64 predictions.
# >     - **Without stratify:** train could randomly have 797 non-predictions and 3 predictions (or worse).
# >     - **Without stratify:** train could randomly have 100 non-predictions and 700 predictions (or worse).
# > - **Test size = 0.2 → 200 samples in test**  
# >     - **With stratify:** test ≈ 184 non-predictions, 16 predictions.
# >     - **Without stratify:** test could randomly have 198 non-predictions and 2 predictions (or worse).
# > - If there exists multi-variate wrt columns, 
# >   - **Me:** Can still choose the one that's imbalanced more and the not as much imbalanced columns will split based on the one that's the most imbalance.
# >   - **Copilot:** Stratify on the column with the greatest imbalance or the one most critical for model performance. This ensures that rare classes in that column are preserved across train and test sets. Other label columns will split based on the same indices, which may slightly alter their original ratios.
# 

# %%
print("======= SPLIT DATA =======")

# %%
spacy_sentence_embeddings_df.head(3)

# %%
# spacy_embeds = spacy_sentence_embeddings_df['Embedding'].to_list()
cols_with_labels = spacy_sentence_embeddings_df.loc[:, ['Sentence Label', 'Author Type']]
cols_with_labels.head(3)

# %%
data_splits = DataProcessing.split_data(spacy_sentence_embeddings_df, cols_with_labels, stratify=True, stratify_by='Sentence Label')
data_splits
# print(f"{X_train_df.head(3)}")

# %%
X_train_df, X_test_df, y_sentence_train_df, y_sentence_test_df, y_author_train_df, y_author_test_df = data_splits
X_train_df.head(3)

# %%
# Example variables for demonstration (replace with actual lengths in your environment)
X_train_len = len(X_train_df)
X_test_len = len(X_test_df)
y_sentence_train_len = len(y_sentence_train_df)
y_sentence_test_len = len(y_sentence_test_df)
y_author_train_len = len(y_author_train_df)
y_author_test_len = len(y_author_test_df)

# Pretty print in a formatted table
print("{:<25} {:>10}".format("Dataset", "Count"))
print("-" * 37)
print("{:<25} {:>10}".format("X_train", X_train_len))
print("{:<25} {:>10}".format("X_test", X_test_len))
print("{:<25} {:>10}".format("y_sentence_train", y_sentence_train_len))
print("{:<25} {:>10}".format("y_sentence_test", y_sentence_test_len))
print("{:<25} {:>10}".format("y_author_train", y_author_train_len))
print("{:<25} {:>10}".format("y_author_test", y_author_test_len))


# %%
X_test_df.head(3)

# %%
save_df = True

if save_df == True:
    print("Save test set so we can pass these into LLMs")
    # save_path = os.path.join(base_data_path, 'combined_generated_fin_phrase_bank')
    DataProcessing.save_to_file(X_test_df, combine_data_path, 'x_test_set', 'csv')
    DataProcessing.save_to_file(y_sentence_test_df, combine_data_path, 'y_sentence_test_df', 'csv')
    DataProcessing.save_to_file(y_author_test_df, combine_data_path, 'y_author_test_df', 'csv')

# %%
y_train_sets = {
    cols_with_labels.columns.to_list()[0]: y_sentence_train_df.to_dict(),
    cols_with_labels.columns.to_list()[1]: y_author_train_df.to_dict()
}

y_train_sets

# %%
y_test_sets = {
    cols_with_labels.columns.to_list()[0]: y_sentence_test_df.to_dict(),
    cols_with_labels.columns.to_list()[1]: y_author_test_df.to_dict()
}

y_test_sets

# %%
X_train_df[embeddings_col_name].to_list()

# %% [markdown]
# ## Models

# %%
print("======= TRAIN x TEST MODELS =======")

# %% [markdown]
# > Track loss: try BCE (Binary Cross Entropy)

# %%
def build_models(factory, names):
    models = {}
    for name in names:
        models[name] = factory.select_model(name)
    return models

ml_model_names = [
    'perceptron',
    'sgd_classifier',
    'logistic_regression',
    'ridge_classifier',
    'decision_tree_classifier',
    'random_forest_classifier',
    'gradient_boosting_classifier',
]

# %%
model_checkpoint_save_path = os.path.join(base_data_path, 'model_checkpoint')
# for model_name, ml_model in ml_models.items():
#     specific_model_checkpoint_save_path = os.path.join(model_checkpoint_save_path, f"model_checkpoint-{model_name}.pkl")
#     # print(specific_model_checkpoint_save_path)
#     joblib.dump(ml_model, specific_model_checkpoint_save_path)


def save_ml_model(model_name, ml_model):
    specific_model_checkpoint_save_path = os.path.join(model_checkpoint_save_path, f"model_checkpoint-{model_name}.pkl")
    # print(specific_model_checkpoint_save_path)
    joblib.dump(ml_model, specific_model_checkpoint_save_path)

# %%
ml_models_with_predictions = {}  # {label_name: {model_name: preds}}

for y_train_set_name, y_train_set_values in y_train_sets.items():
    print(f"y_train_set_name: {y_train_set_name}")
    ml_models = build_models(SkLearnModelFactory, ml_model_names)
    # print(ml_models)

    X_train_list = X_train_df[embeddings_col_name].to_list()
    X_test_list  = X_test_df[embeddings_col_name].to_list()
    y_train_list = list(y_train_set_values.values())

    ml_models_with_predictions[y_train_set_name] = {}

    for model_name, ml_model in ml_models.items():
        print(f"Train -> Predict for {ml_model.get_model_name()} on {y_train_set_name}")
        ml_model.train_model(X_train_list, y_train_list)
        ml_model_predictions = ml_model.predict(X_test_list)
        ml_models_with_predictions[y_train_set_name][model_name] = ml_model_predictions

        save_ml_model(f"{model_name}-{y_train_set_name}", ml_model)

    print()

# %% [markdown]
# ### Align Test Sentences with Predicted Sentence Label from LLMs

# %%
test_and_model_results_df = X_test_df.copy()
for key, value in ml_models_with_predictions['Sentence Label'].items():
    test_and_model_results_df[key] = value.to_list()
test_and_model_results_df.head(3)

# %% [markdown]
# ## Save Output

# %%
DataProcessing.save_to_file(test_and_model_results_df, combine_data_path, 'ml_classifiers', '.csv')

# %% [markdown]
# ## Evaluation

# %%
print("======= EVALUATION/RESULTS =======")

# %%
get_metrics = EvaluationMetric()
get_metrics

# %% [markdown]
# > - Results may differ (from previous runs and even terminal runs) because we shuffle the data.

# %% [markdown]
# ![image.png](attachment:image.png)

# %%
eval_reports = {}
for col_name, model_name_with_results in ml_models_with_predictions.items():
    print(f"####### {col_name} #######")
    
    actual_label = y_sentence_test_df.values
    for ml_model_name, ml_model_predictions in model_name_with_results.items():
        print(f"### Model: {ml_model_name} ###")
    #     print(f"Actual Label:\t\t{actual_label}")
    #     ml_model_predictions = ml_models_with_predictions[ml_model_name].values
    #     print(f"{ml_model_name}:\t\t{ml_model_predictions}")
    #     print()
        eval_report = get_metrics.eval_classification_report(actual_label, ml_model_predictions)
        eval_reports[f"{col_name}-{ml_model_name}"] = eval_report
        print()

# %%
eval_reports_df = pd.DataFrame(eval_reports)
eval_reports_df.to_latex()

# %%
ml_models

# %% [markdown]
# ### Save model checkpoints

# %%
# model_checkpoint_save_path = os.path.join(base_data_path, 'model_checkpoint')
# for model_name, ml_model in ml_models.items():
#     specific_model_checkpoint_save_path = os.path.join(model_checkpoint_save_path, f"model_checkpoint-{model_name}.pkl")
#     # print(specific_model_checkpoint_save_path)
#     joblib.dump(ml_model, specific_model_checkpoint_save_path)

# %%



