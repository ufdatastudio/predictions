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
        label_col: str,
        title: str = None,
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
        
        # Auto-detect if binary (0,1) for better labeling
        if set(counts.index) == {0, 1}:
            labels = ['Non-Prediction', 'Prediction']
            colors = ['#1f77b4', '#ff7f0e']
        else:
            labels = counts.index
            colors = plt.cm.Set3(range(len(counts)))
        
        fig, ax = plt.subplots(figsize=(max(8, len(counts)*1.5), 6))
        bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor='black')
        
        # Annotate bars
        for bar, count in zip(bars, counts.values):
            pct = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'n={count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels, rotation=45 if len(counts) > 3 else 0, ha='right')
        ax.set_ylabel('Count')
        ax.set_title(title or f'{label_col} Distribution')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            DataProcessing.save_to_file(None, save_path, f'{label_col}_distribution', 'png', include_version=True)
        plt.show()
        plt.close()

    def plot_stacked_distribution(
        df: pd.DataFrame,
        category_col: str = 'Dataset Name',
        label_col: str = 'Sentence Label',
        title: str = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot stacked bar chart showing predictions/non-predictions per category."""
        import matplotlib.pyplot as plt
        
        # Create crosstab for stacked data
        cross_tab = pd.crosstab(df[category_col], df[label_col])
        
        # Ensure we have both 0 and 1 columns
        if 0 not in cross_tab.columns:
            cross_tab[0] = 0
        if 1 not in cross_tab.columns:
            cross_tab[1] = 0
        
        cross_tab = cross_tab[[0, 1]]  # Order: non-predictions, predictions
        
        fig, ax = plt.subplots(figsize=(max(8, len(cross_tab)*1.5), 6))
        
        # Create stacked bars
        bars1 = ax.bar(range(len(cross_tab)), cross_tab[0], 
                    color='#1f77b4', label='Non-Predictions (0)', edgecolor='black')
        bars2 = ax.bar(range(len(cross_tab)), cross_tab[1], 
                    bottom=cross_tab[0], color='#ff7f0e', label='Predictions (1)', edgecolor='black')
        
        # Annotate each segment
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            # Non-predictions annotation
            if cross_tab.iloc[i, 0] > 0:
                ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height()/2,
                        f'{cross_tab.iloc[i, 0]}', ha='center', va='center', fontweight='bold')
            
            # Predictions annotation  
            if cross_tab.iloc[i, 1] > 0:
                ax.text(bar2.get_x() + bar2.get_width()/2, 
                        cross_tab.iloc[i, 0] + bar2.get_height()/2,
                        f'{cross_tab.iloc[i, 1]}', ha='center', va='center', fontweight='bold')
            
            # Total on top
            total = cross_tab.iloc[i, 0] + cross_tab.iloc[i, 1]
            
            ax.text(bar2.get_x() + bar2.get_width()/2, total + cross_tab.values.max() * 0.02,
                    f'Total: {total}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_xticks(range(len(cross_tab)))
        ax.set_xticklabels(cross_tab.index, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title(title or f'{category_col} Distribution (Stacked by {label_col})')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            DataProcessing.save_to_file(None, save_path, f'{category_col}_stacked_distribution', 'png', include_version=True)
        plt.show()
        plt.close()
        
        # Print the crosstab
        print(f"\n{category_col} vs {label_col} Breakdown:")
        print(cross_tab)
        
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
        label_col: str = 'Label',
        class_names: list = ['Non-Prediction', 'Prediction'],
        feature_cols: list = ['Feature_1', 'Feature_2'],
        method_name: str = 'Resampling',
        title: str = None,
        save: bool = False,
        save_path: str = None
    ) -> None:
        """
        Parameters
        ----------
        # ... (rest of your docstring) ...
        save : bool, default False
            Whether to save the figure to disk.
        save_path : str, optional
            The directory path where the figure should be saved.
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
            
            # Format the title dynamically
            if title is None:
                chart_title = f'{stage} {method_name} – Class Distribution'
            else:
                chart_title = f'{stage}: {title} – Class Distribution'
                
            ax.set_title(chart_title)
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
                       
            # Format the title dynamically
            if title is None:
                scatter_title = f'{stage} {method_name} – Feature Space\n(n={len(df)})'
            else:
                scatter_title = f'{stage}: {title} – Feature Space\n(n={len(df)})'
                
            ax.set_title(scatter_title)
            ax.set_xlabel(feature_cols[0])
            ax.set_ylabel(feature_cols[1])
            ax.legend()
            ax.grid(alpha=0.3)
            
        plt.tight_layout()
        
        # Save logic
        if save and save_path:
            # Create a clean file prefix
            base_prefix = title.lower().replace(' ', '_').replace('(', '').replace(')', '') if title else method_name.lower()
            
            # Use your DataProcessing save method (it uses plt.savefig internally if 'png' is passed)
            DataProcessing.save_to_file(
                data=fig, 
                path=save_path, 
                prefix=f'resampling_visual_{base_prefix}', 
                save_file_type='png',
                include_version=False
            )
            
        # plt.show() # Optional depending on if you are running headless
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

    def print_cluster_samples(df, labels, sentence_label, text_column, show_sentences_per_cluster):
        """
        Prints label distribution and sample sentences per cluster.
        Used by both plot_kmeans_tsne and plot_kmeans_tsne_filtered.
        """
        show_sentence_label = sentence_label in df.columns

        # Add Cluster column to df copy — 0 or 1 based on KMeans label
        df = df.copy()
        df['Cluster'] = labels

        print(f"\n--- Sample Sentences Per Cluster ---")
        cluster_dfs = []
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

            cluster_dfs.append(df.iloc[cluster_indices])

        # Return OUTSIDE the for loop — after all clusters are processed
        return df, cluster_dfs


    def plot_kmeans_tsne_filtered(df, x_axis_filter, y_axis_filter, tsne, labels, sentence_label, text_column, show_sentences_per_cluster):

        # Build combined mask based on filters provided
        mask = np.ones(len(tsne), dtype=bool)
        title_parts = []

        if x_axis_filter is not None:
            x_min, x_max = x_axis_filter
            mask &= (tsne[:, 0] >= x_min) & (tsne[:, 0] <= x_max)
            title_parts.append(f"x={x_min} to {x_max}")

        if y_axis_filter is not None:
            y_min, y_max = y_axis_filter
            mask &= (tsne[:, 1] >= y_min) & (tsne[:, 1] <= y_max)
            title_parts.append(f"y={y_min} to {y_max}")

        # Apply mask
        filtered_tsne = tsne[mask]
        filtered_labels = labels[mask]
        filtered_df = df[mask].reset_index(drop=True)

        # Add coordinates and cluster to filtered_df
        filtered_df = filtered_df.copy()
        filtered_df['Cluster'] = filtered_labels
        filtered_df['tsne_x'] = filtered_tsne[:, 0]
        filtered_df['tsne_y'] = filtered_tsne[:, 1]

        # Plot full filtered region
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            filtered_tsne[:, 0], filtered_tsne[:, 1],
            c=filtered_labels, cmap='viridis', s=100, alpha=0.6
        )
        plt.title(f'KMeans Clustering with t-SNE Visualization\n(Filtered: {", ".join(title_parts)})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.show()

        print(f"\nFiltered points: {len(filtered_tsne)} out of {len(tsne)}")

        # Plot per cluster separately
        for cluster_id in sorted(set(filtered_labels)):
            cluster_mask = filtered_labels == cluster_id
            cluster_tsne = filtered_tsne[cluster_mask]

            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                cluster_tsne[:, 0], cluster_tsne[:, 1],
                c=[cluster_id] * len(cluster_tsne), cmap='viridis',
                vmin=0, vmax=max(filtered_labels),
                s=100, alpha=0.6
            )
            plt.title(f'KMeans Clustering with t-SNE Visualization\n(Filtered: {", ".join(title_parts)}, Cluster {cluster_id})')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.colorbar(scatter, label='Cluster')
            plt.tight_layout()
            plt.show()

        # Print samples using shared function
        filtered_df, cluster_dfs = DataVisualizing.print_cluster_samples(
            df=filtered_df,
            labels=filtered_labels,
            sentence_label=sentence_label,
            text_column=text_column,
            show_sentences_per_cluster=show_sentences_per_cluster
        )

        return filtered_df, cluster_dfs


    def plot_kmeans_tsne(
        df,
        text_column,
        embedding_col_name,
        n_clusters,
        show_sentences_per_cluster,
        sentence_label,
        filter_x_axis=None,
        filter_y_axis=None
    ):
        # Create an empty list to store our embedding vectors
        embeddings_list = []
        embeddings_raw = df[embedding_col_name].values

        # Loop through each embedding in the DataFrame
        for embedding in embeddings_raw:
            if isinstance(embedding, str):
                emb_stripped = embedding.strip('[]')
                emb_array = np.fromstring(emb_stripped, sep=' ', dtype=np.float64)
            else:
                emb_array = np.array(embedding, dtype=np.float64)
            embeddings_list.append(emb_array)

        # Convert to 2D numpy array
        embeddings = np.array(embeddings_list)

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(embeddings)
        labels = kmeans.labels_

        # t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=2, random_state=0).fit_transform(embeddings)

        # Add coordinates and cluster to df
        df = df.copy()
        df['Cluster'] = labels
        df['tsne_x'] = tsne[:, 0]
        df['tsne_y'] = tsne[:, 1]

        # Plot full dataset
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap='viridis', s=100, alpha=0.6)
        plt.title(f'KMeans Clustering with t-SNE Visualization\nClusters: {n_clusters}, Total points: {len(embeddings)}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.show()

        # Plot per cluster separately
        for cluster_id in sorted(set(labels)):
            cluster_mask = labels == cluster_id
            cluster_tsne = tsne[cluster_mask]

            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                cluster_tsne[:, 0], cluster_tsne[:, 1],
                c=[cluster_id] * len(cluster_tsne), cmap='viridis',
                vmin=0, vmax=max(labels),
                s=100, alpha=0.6
            )
            plt.title(f'KMeans Clustering with t-SNE Visualization\nCluster {cluster_id}, Total points: {len(cluster_tsne)}')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.colorbar(scatter, label='Cluster')
            plt.tight_layout()
            plt.show()

        # Print samples using shared function
        df, cluster_dfs = DataVisualizing.print_cluster_samples(
            df=df,
            labels=labels,
            sentence_label=sentence_label,
            text_column=text_column,
            show_sentences_per_cluster=show_sentences_per_cluster
        )

        # Apply filter if provided
        if filter_x_axis is not None or filter_y_axis is not None:
            filtered_df, cluster_dfs = DataVisualizing.plot_kmeans_tsne_filtered(
                df=df,
                x_axis_filter=filter_x_axis,
                y_axis_filter=filter_y_axis,
                tsne=tsne,
                labels=labels,
                sentence_label=sentence_label,
                text_column=text_column,
                show_sentences_per_cluster=show_sentences_per_cluster
            )
            return filtered_df, cluster_dfs

        return df, cluster_dfs
    
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
