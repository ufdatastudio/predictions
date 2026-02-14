
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from typing import Union
from spacy import displacy
from spacy.tokens import Doc

from IPython.display import HTML, display

class DataPlotting:
    """A class to plot data"""

    def plot_balancedness(X, y, method_name_to_balance, sampling_strategy: str, classes_names: list = ["negative", "positive"]):
        """
        Plot class balance before and after applying an oversampling method.

        This function visualizes a binary classification dataset in 2D (first two feature
        dimensions) before and after applying a resampling method such as
        `RandomOverSampler`, `SMOTE`, or `BorderlineSMOTE`. It produces a side-by-side
        scatter plot to illustrate how the chosen method affects class balance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Must be indexable with boolean masks and column slicing
            (e.g., a NumPy array or a SciPy sparse matrix converted to dense).
            **Note:** The plot uses only the first two features (columns 0 and 1). If
            your data is high-dimensional (e.g., TF-IDF or sentence embeddings), consider
            reducing to 2D with PCA/t-SNE/UMAP before calling this function.
        
        y : array-like of shape (n_samples,)
            Target labels containing exactly two classes encoded as 0 and 1.
            Must support boolean masking (e.g., NumPy array or Pandas Series).
        
        method_name_to_balance : callable
            A resampler class (not an instance) from `imblearn`, e.g.,
            `imblearn.over_sampling.SMOTE` or `imblearn.over_sampling.RandomOverSampler`.
            The function will instantiate it as:
                ``oversample = method_name_to_balance(sampling_strategy=sampling_strategy, random_state=42)``
            and then call:
                ``X_over, y_over = oversample.fit_resample(X, y)``.
        
        sampling_strategy : str or dict or float
            Sampling strategy passed directly to the resampler. Common options:
            - ``'auto'`` / ``'not majority'`` / ``'minority'`` / ``'all'`` (string aliases)
            - ``float`` in (0, 1] specifying the desired ratio of the minority class
            over the majority class after resampling (for binary problems)
            - ``dict`` mapping class label -> number of samples after resampling
            Refer to imbalanced-learn documentation for specifics of each method.
        
        classes_names : list of str with length 2
            Human-readable names for the two classes, in the order: `[name_for_class_0, name_for_class_1]`.
            Used only for plot legends and titles.

        Returns
        -------
        None
            This function produces a matplotlib figure with two subplots and displays it.

        Notes
        -----
        - Ensure ``X`` is a NumPy array (or convertible) if you plan to use boolean
        masking and column slicing. If you have a Python list of lists, convert via
        ``X = np.array(X)``.
        - If your data is textual embeddings or high-dimensional vectors, reduce to 2D
        first (e.g., PCA: ``PCA(n_components=2).fit_transform(X)``) so the scatter
        plot is meaningful.
        - The function assumes binary labels coded as 0 and 1. For multiclass, adapt
        the plotting logic accordingly.

        Raises
        ------
        ValueError
            If ``classes_names`` does not have length 2, or if ``y`` does not contain
            exactly the labels {0, 1}, or if ``X`` has fewer than 2 features/columns.
        TypeError
            If ``method_name_to_balance`` is not callable.

        """
        # if len(classes_names) == 0:
        #     class_0_name = "negative"
        #     class_1_name = "positive"
        if len(classes_names) == 2: 
            class_0_name = classes_names[0]
            class_1_name = classes_names[1]    
        # else:
        #     raise ValueError("`classes_names` must be an empty list or a list of exactly two class names.")

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label=f"Class 0: {class_0_name}", alpha=0.5, edgecolor="k")
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label=f"Class 1: {class_1_name}", alpha=0.5, edgecolor="k")
        plt.title("Original Imbalanced Dataset")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        # Apply oversampling
        oversample = method_name_to_balance(sampling_strategy=sampling_strategy, random_state=42)
        X_over, y_over = oversample.fit_resample(X, y)
        # Visualize after oversampling
        plt.subplot(1, 2, 2)
        plt.scatter(X_over[y_over == 0][:, 0], X_over[y_over == 0][:, 1], label=f"Class 0: {class_0_name}", alpha=0.5, edgecolor="k")
        plt.scatter(X_over[y_over == 1][:, 0], X_over[y_over == 1][:, 1], label=f"Class 1: {class_1_name}", alpha=0.5, edgecolor="k")
        plt.title(f"Dataset After Oversampling: {method_name_to_balance.__name__}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_balancedness_two(
        X: np.ndarray, 
        y: np.ndarray, 
        method_to_balance, 
        sampling_strategy: str = 'auto', 
        classes_names: list = None
    ) -> None:
        """
        Plot class balance before and after applying a resampling method.
        
        Visualizes class distribution using the first feature only.
        Creates a 1D scatter plot showing the distribution of samples.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix. Only the first feature is used for visualization.
        y : np.ndarray of shape (n_samples,)
            Target labels (0 and 1 for binary classification).
        method_to_balance : callable
            Resampler class from imblearn (e.g., SMOTE, RandomOverSampler).
        sampling_strategy : str or dict or float, default='auto'
            Sampling strategy passed to the resampler.
        classes_names : list of str, optional
            Names for [class_0, class_1]. Defaults to ["Class 0", "Class 1"].
            
        Returns
        -------
        None
            Displays a matplotlib figure with two subplots.
        """
        # Set default class names
        if classes_names is None or len(classes_names) != 2:
            classes_names = ["Class 0", "Class 1"]
        
        class_0_name, class_1_name = classes_names
        
        # Convert to numpy array if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Use only the first feature
        X_plot = X[:, 0]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot original dataset
        axes[0].scatter(
            X_plot[y == 0], np.zeros(sum(y == 0)), 
            label=class_0_name, alpha=0.5, edgecolor="k", s=100
        )
        axes[0].scatter(
            X_plot[y == 1], np.ones(sum(y == 1)), 
            label=class_1_name, alpha=0.5, edgecolor="k", s=100
        )
        axes[0].set_title(
            f"Original Dataset\n"
            f"(Class 0: {sum(y==0)}, Class 1: {sum(y==1)})"
        )
        axes[0].set_xlabel("Feature Value")
        axes[0].set_ylabel("Class")
        axes[0].set_yticks([0, 1])
        axes[0].set_yticklabels([class_0_name, class_1_name])
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Apply resampling
        resampler = method_to_balance(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        
        # Use only first feature for plotting
        X_resampled_plot = X_resampled[:, 0]
        
        # Plot resampled dataset
        axes[1].scatter(
            X_resampled_plot[y_resampled == 0], np.zeros(sum(y_resampled == 0)),
            label=class_0_name, alpha=0.5, edgecolor="k", s=100
        )
        axes[1].scatter(
            X_resampled_plot[y_resampled == 1], np.ones(sum(y_resampled == 1)),
            label=class_1_name, alpha=0.5, edgecolor="k", s=100
        )
        axes[1].set_title(
            f"After {method_to_balance.__name__}\n"
            f"(Class 0: {sum(y_resampled==0)}, Class 1: {sum(y_resampled==1)})"
        )
        axes[1].set_xlabel("Feature Value")
        axes[1].set_ylabel("Class")
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels([class_0_name, class_1_name])
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class DataVisualizing:

    def _ensure_doc(text_or_doc: Union[str, Doc], nlp) -> Doc:
        """Return a spaCy Doc. If input is str -> nlp(str); if Doc -> return as is."""
        return nlp(text_or_doc) if isinstance(text_or_doc, str) else text_or_doc

    def spacy_pos_dep(sentence: Union[str, Doc], spacy_nlp_model):
        doc = DataVisualizing._ensure_doc(sentence, spacy_nlp_model)
        html = displacy.render(doc, style="dep", jupyter=False)
        display(HTML(html))

    def spacy_ner_ent(sentence: Union[str, Doc], spacy_nlp_model):
        doc = DataVisualizing._ensure_doc(sentence, spacy_nlp_model)
        html = displacy.render(doc, style="ent", jupyter=False)
        display(HTML(html))
    
    def spacy_dep_ent(sentence: Union[str, Doc], spacy_nlp_model, mode: str = "both"):
        mode = mode.lower()
        if mode == "pos_dep":
            return DataVisualizing.extract_pos_features(sentence, spacy_nlp_model)

        if mode == "ner_ent":
            return DataVisualizing.extract_ner_features(sentence, spacy_nlp_model)

        # both
        DataVisualizing.spacy_pos_dep(sentence, spacy_nlp_model)
        DataVisualizing.spacy_ner_ent(sentence, spacy_nlp_model)

    def spacy_vis_all(self, *args, **kwargs):
        """Alias to preserve backward compatibility. Visualize pos_dep, dep_ent."""
        return self.extract_features(*args, **kwargs)

    def visualize_before_after_resampling(df_before: pd.DataFrame,
                                      df_after: pd.DataFrame,
                                      label_col: str = 'Label',
                                      class_names: list = ['Class 0', 'Class 1'],
                                      feature_cols: list = ['Feature_1', 'Feature_2'],
                                      method_name: str = 'Resampling') -> None:
        """Visualize class distribution before and after resampling."""
        fig = plt.figure(figsize=(14, 10))
        
        # Before resampling - bar chart
        ax1 = plt.subplot(2, 2, 1)
        counts_before = df_before[label_col].value_counts().sort_index()
        total_before = len(df_before[label_col])
        
        bars_before = ax1.bar([0, 1], counts_before.values, color=['#1f77b4', '#ff7f0e'],
                            edgecolor='black', width=0.6)
        
        for bar, count in zip(bars_before, counts_before.values):
            percentage = (count / total_before) * 100
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'n={count}\n({percentage:.1f}%)', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(class_names)
        ax1.set_ylabel('Count')
        ax1.set_title(f'Before {method_name} - Class Distribution')
        ax1.set_ylim(0, max(counts_before.values) * 1.30)
        ax1.grid(axis='y', alpha=0.3)
        
        # After resampling - bar chart
        ax2 = plt.subplot(2, 2, 2)
        counts_after = df_after[label_col].value_counts().sort_index()
        total_after = len(df_after[label_col])
        
        bars_after = ax2.bar([0, 1], counts_after.values, color=['#1f77b4', '#ff7f0e'],
                            edgecolor='black', width=0.6)
        
        for bar, count in zip(bars_after, counts_after.values):
            percentage = (count / total_after) * 100
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'n={count}\n({percentage:.1f}%)', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(class_names)
        ax2.set_ylabel('Count')
        ax2.set_title(f'After {method_name} - Class Distribution')
        ax2.set_ylim(0, max(counts_after.values) * 1.30)
        ax2.grid(axis='y', alpha=0.3)
        
        # Before resampling - scatter plot
        ax3 = plt.subplot(2, 2, 3)
        X_before = df_before[feature_cols].values
        y_before = df_before[label_col].values
        
        ax3.scatter(X_before[y_before == 0][:, 0], X_before[y_before == 0][:, 1],
                label=class_names[0], alpha=0.5, edgecolor='k')
        ax3.scatter(X_before[y_before == 1][:, 0], X_before[y_before == 1][:, 1],
                label=class_names[1], alpha=0.5, edgecolor='k')
        ax3.set_title(f'Before {method_name} - Feature Space\n(n={len(df_before)})')
        ax3.set_xlabel(feature_cols[0])
        ax3.set_ylabel(feature_cols[1])
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # After resampling - scatter plot
        ax4 = plt.subplot(2, 2, 4)
        X_after = df_after[feature_cols].values
        y_after = df_after[label_col].values
        
        ax4.scatter(X_after[y_after == 0][:, 0], X_after[y_after == 0][:, 1],
                label=class_names[0], alpha=0.5, edgecolor='k')
        ax4.scatter(X_after[y_after == 1][:, 0], X_after[y_after == 1][:, 1],
                label=class_names[1], alpha=0.5, edgecolor='k')
        ax4.set_title(f'After {method_name} - Feature Space\n(n={len(df_after)})')
        ax4.set_xlabel(feature_cols[0])
        ax4.set_ylabel(feature_cols[1])
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
