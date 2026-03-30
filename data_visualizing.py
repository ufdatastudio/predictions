import os

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from spacy import displacy
from spacy.tokens import Doc
from typing import Union, Optional

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from IPython.display import HTML, display

from data_processing import DataProcessing
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

class DataVisualizing:
    """A class to visualize spaCy NLP outputs."""

    def plot_class_distribution(
        df: pd.DataFrame,
        label_col: str = 'Prediction',
        class_names: list = ['Non-Prediction', 'Prediction'],
        title: str = 'Class Distribution',
        save_path: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the label column.
        label_col : str, default 'Prediction'
            Column name holding binary class labels (0 and 1).
        class_names : list, default ['Non-Prediction', 'Prediction']
            Human-readable names for [class_0, class_1].
        title : str, default 'Class Distribution'
            Title of the bar chart.
        save_path : str, optional
            If provided, saves the figure to this path.

        Notes
        -----
        General-purpose bar chart for any binary label column.
        Annotates each bar with count and percentage.

        Returns
        -------
        None
        """
        counts = df[label_col].value_counts().sort_index()
        total = len(df)

        fig, ax = plt.subplots(figsize=(7, 5))

        bars = ax.bar(
            [0, 1],
            counts.values,
            color=['#1f77b4', '#ff7f0e'],
            edgecolor='black',
            width=0.5,
        )

        # Annotate each bar with count and percentage
        for bar, count in zip(bars, counts.values):
            pct = (count / total) * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'n={count}\n({pct:.1f}%)',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold',
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(class_names)
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.set_ylim(0, max(counts.values) * 1.30)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            DataProcessing.save_to_file(
                None, save_path, 'class_distribution', 'png', include_version=True
            )

        plt.show()
        plt.close()

    def plot_balancedness(
        X: np.ndarray,
        y: np.ndarray,
        method_name_to_balance,
        sampling_strategy: str,
        classes_names: list = ['Non-Prediction', 'Prediction'],
    ) -> None:
        """
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix. Only the first two features are used for plotting.
        y : np.ndarray of shape (n_samples,)
            Binary labels (0 and 1).
        method_name_to_balance : callable
            Resampler class from imblearn (e.g., SMOTE, RandomOverSampler).
        sampling_strategy : str or dict or float
            Sampling strategy passed to the resampler.
        classes_names : list, default ['Non-Prediction', 'Prediction']
            Human-readable names for [class_0, class_1].

        Notes
        -----
        Scatter plot of the first two features before and after resampling.

        Returns
        -------
        None
        """
        class_0_name, class_1_name = classes_names

        plt.figure(figsize=(12, 5))

        # --- Before resampling ---
        plt.subplot(1, 2, 1)
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1],
                    label=f"Class 0: {class_0_name}", alpha=0.5, edgecolor='k')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1],
                    label=f"Class 1: {class_1_name}", alpha=0.5, edgecolor='k')
        plt.title(f"Original Dataset\n(Class 0: {sum(y==0)}, Class 1: {sum(y==1)})")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(alpha=0.3)

        # Apply resampling
        resampler = method_name_to_balance(sampling_strategy=sampling_strategy, random_state=42)
        X_over, y_over = resampler.fit_resample(X, y)

        # --- After resampling ---
        plt.subplot(1, 2, 2)
        plt.scatter(X_over[y_over == 0][:, 0], X_over[y_over == 0][:, 1],
                    label=f"Class 0: {class_0_name}", alpha=0.5, edgecolor='k')
        plt.scatter(X_over[y_over == 1][:, 0], X_over[y_over == 1][:, 1],
                    label=f"Class 1: {class_1_name}", alpha=0.5, edgecolor='k')
        plt.title(
            f"After {method_name_to_balance.__name__}\n"
            f"(Class 0: {sum(y_over==0)}, Class 1: {sum(y_over==1)})"
        )
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_balancedness_before_after(
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        label_col: str = 'Prediction',
        class_names: list = ['Non-Prediction', 'Prediction'],
        feature_cols: list = ['Feature_1', 'Feature_2'],
        method_name: str = 'Resampling',
    ) -> None:
        """
        Parameters
        ----------
        df_before : pd.DataFrame
            DataFrame before resampling.
        df_after : pd.DataFrame
            DataFrame after resampling.
        label_col : str, default 'Prediction'
            Column name holding binary labels.
        class_names : list, default ['Non-Prediction', 'Prediction']
            Human-readable names for [class_0, class_1].
        feature_cols : list, default ['Feature_1', 'Feature_2']
            Two feature column names used for the scatter plot.
        method_name : str, default 'Resampling'
            Name of the resampling method (used in titles).

        Notes
        -----
        2x2 grid: bar charts (top) and scatter plots (bottom),
        before and after resampling.

        Returns
        -------
        None
        """
        fig = plt.figure(figsize=(14, 10))

        # ---- Top row: bar charts ----------------------------------------
        for col_idx, (df, stage) in enumerate(
            [(df_before, 'Before'), (df_after, 'After')], start=1
        ):
            ax = plt.subplot(2, 2, col_idx)
            counts = df[label_col].value_counts().sort_index()
            total = len(df)

            bars = ax.bar(
                [0, 1], counts.values,
                color=['#1f77b4', '#ff7f0e'],
                edgecolor='black', width=0.6,
            )
            for bar, count in zip(bars, counts.values):
                pct = (count / total) * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'n={count}\n({pct:.1f}%)',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold',
                )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(class_names)
            ax.set_ylabel('Count')
            ax.set_title(f'{stage} {method_name} – Class Distribution')
            ax.set_ylim(0, max(counts.values) * 1.30)
            ax.grid(axis='y', alpha=0.3)

        # ---- Bottom row: scatter plots ----------------------------------
        for col_idx, (df, stage) in enumerate(
            [(df_before, 'Before'), (df_after, 'After')], start=3
        ):
            ax = plt.subplot(2, 2, col_idx)
            X = df[feature_cols].values
            y = df[label_col].values

            ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1],
                       label=class_names[0], alpha=0.5, edgecolor='k')
            ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1],
                       label=class_names[1], alpha=0.5, edgecolor='k')
            ax.set_title(f'{stage} {method_name} – Feature Space\n(n={len(df)})')
            ax.set_xlabel(feature_cols[0])
            ax.set_ylabel(feature_cols[1])
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.close()

    def confusion_matrix(
            model_name: str,
            confusion_mat: np.ndarray,
            save_path: str,
            class_names: list = ['Non-Prediction', 'Prediction'],
            include_version: bool = False
            ) -> None:
        """
        Parameters
        ----------
        model_name : str
            Name of the model (used in title and filename).
        confusion_mat : np.ndarray
            2x2 confusion matrix.
        save_path : str
            Directory path to save the PNG file.
        class_names : list, default ['Non-Prediction', 'Prediction']
            Human-readable names for [class_0, class_1].
        include_version : bool, default False
            If True, appends a version suffix (-v1, -v2, …) to the filename.

        Notes
        -----
        Saves a colored heatmap of the confusion matrix.

        Returns
        -------
        None
        """
        plt.figure(figsize=(8, 6))

        sns.heatmap(
            confusion_mat,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=True,
            square=True,
            xticklabels=class_names,
            yticklabels=class_names,
        )

        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix – {model_name}')
        plt.tight_layout()

        DataProcessing.save_to_file(
            None,
            save_path,
            f'confusion_matrix_{model_name}',
            'png',
            include_version=include_version,
        )

        plt.show()
        plt.close()

    def roc_curve(
            model_name: str,
            model: str,
            X_test: np.ndarray,
            y_test: str,
            save_path: str,
            include_version: bool = False
            ) -> None:
        """
        Parameters
        ----------
        model_name : str
            Name of the model (used in title and filename).
        save_path : str
            Directory path to save the PNG file.
        include_version : bool, default False
            If True, appends a version suffix (-v1, -v2, …) to the filename.

        Notes
        -----
        Saves a colored heatmap of the confusion matrix.

        Returns
        -------
        None
        """
        plt.figure(figsize=(8, 6))

        RocCurveDisplay.from_estimator(model.classifer, X_test, y_test)
        plt.title("ROC Curve")

        DataProcessing.save_to_file(
            None,
            save_path,
            f'roc_curve_{model_name}',
            'png',
            include_version=include_version,
        )

        plt.show()
        plt.close()

    def pr_curve(
            model_name: str,
            model: str,
            X_test: np.ndarray,
            y_test: str,
            save_path: str,
            include_version: bool = False
            ) -> None:
        """
        Parameters
        ----------
        model_name : str
            Name of the model (used in title and filename).
        save_path : str
            Directory path to save the PNG file.
        include_version : bool, default False
            If True, appends a version suffix (-v1, -v2, …) to the filename.

        Notes
        -----
        Saves a colored heatmap of the confusion matrix.

        Returns
        -------
        None
        """
        plt.figure(figsize=(8, 6))

        PrecisionRecallDisplay.from_estimator(model.classifer, X_test, y_test)
        plt.title("Precision-Recall Curve")

        DataProcessing.save_to_file(
            None,
            save_path,
            f'pr_curve_{model_name}',
            'png',
            include_version=include_version,
        )

        plt.show()
        plt.close()
    
    def get_shap_plot(
        shap_values,
        plot_type: str = 'waterfall',
        sample_idx: int = 0,
        model_name: str = 'model',
        save_path: str = None,
        include_version: bool = False,
        **kwargs
    ) -> None:
        """
        Generate and save SHAP plots for model explainability.
        
        Parameters
        ----------
        shap_values : shap.Explanation
            SHAP values from explainer(X)
        plot_type : str
            Type of SHAP plot: 'waterfall', 'beeswarm', 'bar', 'heatmap'
            Default: 'waterfall'
        sample_idx : int
            Sample index for waterfall plot. Default: 0
        model_name : str
            Name of the model (used in filename). Default: 'model'
        save_path : str, optional
            Directory path to save the PNG file
        include_version : bool, default False
            If True, appends version suffix (-v1, -v2, …) to filename
        **kwargs
            Additional arguments passed to shap plot functions
        
        Notes
        -----
        plot_type options:
        - 'waterfall': Single prediction explanation (uses sample_idx)
        - 'beeswarm': Global feature importance across all samples
        - 'bar': Mean absolute SHAP values per feature
        - 'heatmap': SHAP values across all samples and features
        
        Returns
        -------
        None
        """
        import shap
        
        plt.figure()
        
        if plot_type == 'waterfall':
            shap.plots.waterfall(shap_values[sample_idx], show=False, **kwargs)
        elif plot_type == 'beeswarm':
            shap.plots.beeswarm(shap_values, show=False, **kwargs)
        elif plot_type == 'bar':
            shap.plots.bar(shap_values, show=False, **kwargs)
        elif plot_type == 'heatmap':
            shap.plots.heatmap(shap_values, show=False, **kwargs)
        else:
            raise ValueError(
                f"Unknown plot_type: '{plot_type}'. "
                f"Choose from: 'waterfall', 'beeswarm', 'bar', 'heatmap'"
            )
        
        plt.tight_layout()
        
        if save_path:
            DataProcessing.save_to_file(
                None,
                save_path,
                f'shap_{plot_type}_{model_name}',
                'png',
                include_version=include_version
            )
        
        plt.show()
        plt.close()

    def plot_kmeans_tsne(
        df,
        text_column,
        embedding_col_name,
        n_clusters,
        show_sentences_per_cluster,
        sentence_label
    ):
        # Check if the sentence label column exists in the DataFrame
        show_sentence_label = sentence_label in df.columns

        # Create an empty list to store our embedding vectors
        embeddings_list = []
        embeddings_raw = df[embedding_col_name].values

        # Loop through each embedding in the DataFrame
        for embedding in embeddings_raw:
            # Check if the embedding is stored as a string (e.g., "[-0.12  0.24  ...]")
            if isinstance(embedding, str):
                # Step 1: Remove the surrounding brackets [ ]
                emb_stripped = embedding.strip('[]')
                # Step 2: Convert the string of numbers into a numpy array
                emb_array = np.fromstring(emb_stripped, sep=' ', dtype=np.float64)
            else:
                # If it's already a numpy array, use it directly
                emb_array = np.array(embedding, dtype=np.float64)

            # Add the embedding vector to our list
            embeddings_list.append(emb_array)

        # Convert the list of arrays into a single 2D numpy array
        embeddings = np.array(embeddings_list)

        # KMeans clustering on full dimensional data
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(embeddings)
        labels = kmeans.labels_

        # t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=2, random_state=0).fit_transform(embeddings)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis', s=100, alpha=0.6)
        plt.title(f'KMeans Clustering with t-SNE Visualization\nClusters: {n_clusters}, Total points: {len(embeddings)}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.show()

        # Print sample sentences and label distribution per cluster
        print(f"\n--- Sample Sentences Per Cluster ---")
        for cluster_id in sorted(set(labels)):
            cluster_indices = np.where(labels == cluster_id)[0]
            sample_indices = np.random.choice(
                cluster_indices,
                size=min(show_sentences_per_cluster, len(cluster_indices)),
                replace=False
            )

            print(f"\nCluster {cluster_id} ({len(cluster_indices)} sentences):")

            # Print label distribution (0s and 1s) for this cluster
            if show_sentence_label:
                cluster_df = df.iloc[cluster_indices]
                label_counts = cluster_df[sentence_label].value_counts().sort_index()
                for label_val, count in label_counts.items():
                    if label_val == 0:
                        print(f"  Label {label_val} (Non-Prediction): {count} sentences ({count / len(cluster_indices) * 100:.1f}%)")
                    else:
                        print(f"  Label {label_val} (Prediction): {count} sentences ({count / len(cluster_indices) * 100:.1f}%)")

            # Print sample sentences
            print(f"\n  Sample Sentences:")
            for idx in sample_indices:
                if show_sentence_label:
                    print(f"  - [{df.iloc[idx][sentence_label]}] {df.iloc[idx][text_column]}")
                else:
                    print(f"  - {df.iloc[idx][text_column]}")
    
    def _ensure_doc(text_or_doc: Union[str, Doc], nlp) -> Doc:
        """Return a spaCy Doc – parse if string, pass through if already a Doc."""
        return nlp(text_or_doc) if isinstance(text_or_doc, str) else text_or_doc

    def spacy_pos_dep(sentence: Union[str, Doc], spacy_nlp_model) -> None:
        """Render the dependency parse for a sentence."""
        doc = DataVisualizing._ensure_doc(sentence, spacy_nlp_model)
        html = displacy.render(doc, style='dep', jupyter=False)
        display(HTML(html))

    def spacy_ner_ent(sentence: Union[str, Doc], spacy_nlp_model) -> None:
        """Render the named-entity visualisation for a sentence."""
        doc = DataVisualizing._ensure_doc(sentence, spacy_nlp_model)
        html = displacy.render(doc, style='ent', jupyter=False)
        # display(HTML(html))

        display(HTML(f"""
        <div style="max-width:{200}px;">
            {html}
        </div>
        """))

    def spacy_dep_ent(
        sentence: Union[str, Doc],
        spacy_nlp_model,
        mode: str = 'both',
    ) -> None:
        """
        Parameters
        ----------
        sentence : str or Doc
            Input sentence or spaCy Doc.
        spacy_nlp_model : spacy.Language
            Loaded spaCy model.
        mode : {'pos_dep', 'ner_ent', 'both'}, default 'both'
            Which visualisation to render.

        Notes
        -----
        Thin dispatcher that calls the appropriate renderer(s).

        Returns
        -------
        None
        """
        mode = mode.lower()
        if mode == 'pos_dep':
            DataVisualizing.spacy_pos_dep(sentence, spacy_nlp_model)
        elif mode == 'ner_ent':
            DataVisualizing.spacy_ner_ent(sentence, spacy_nlp_model)
        else:
            # Default: render both dependency and entity visualisations
            DataVisualizing.spacy_pos_dep(sentence, spacy_nlp_model)
            DataVisualizing.spacy_ner_ent(sentence, spacy_nlp_model)
