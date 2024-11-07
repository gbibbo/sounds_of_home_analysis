# scripts/advanced_analysis.py

"""
Advanced Sound Analysis Script

This script performs multiple analyses on sound event data:

1. Time Series Analysis:
   - Plots the occurrence of sound events over time.
   - Helps observe daily trends, activity peaks, and common patterns.

2. Basic Statistical Analysis:
   - Provides statistical summaries (mean, median, standard deviation, etc.) of event counts.

3. Correlation Analysis:
   - Calculates the Pearson correlation coefficient between different sound classes.
   - Identifies classes that tend to increase or decrease together.

4. Principal Component Analysis (PCA):
   - Reduces data dimensionality to identify common patterns.
   - Highlights combinations of classes that explain the greatest variation over time.

5. Heatmaps and Clustering:
   - Visualizes the intensity of events by class and hour using heatmaps.
   - Applies hierarchical clustering to group similar hours or classes.

6. Peak Activity Analysis:
   - Identifies the hour with peak activity for each sound class.
   - Provides insights into when certain sound events are most frequent.

You can select which analyses to perform by setting the corresponding flags in the `selected_analyses` dictionary.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import src.config as config
from src.data_processing.load_data import (
    load_ontology,
    build_mappings,
    build_parent_child_mappings,
    load_class_labels
)
from src.data_processing.utils import get_all_subclasses, get_class_id

def save_plot(fig, name):
    """
    Save plot to assets/images directory
    Args:
        fig: Matplotlib figure object
        name (str): Name for the output file
    """
    save_dir = os.path.join('assets', 'images')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{name}.png')
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {save_path}")

def process_data(data_counts, selected_classes, class_mappings, hierarchy_mappings):
    """
    Processes the data to create a DataFrame suitable for analysis.

    Args:
        data_counts (dict): Nested dictionary with counts per recorder and hour
        selected_classes (list): List of class names to analyze
        class_mappings (tuple): Contains mappings between class labels and IDs
        hierarchy_mappings (tuple): Contains parent-child relationships

    Returns:
        pandas.DataFrame: Processed data with hours as index and classes as columns
    """
    class_label_to_id, id_to_class, name_to_class_id = class_mappings
    parent_to_children, child_to_parents = hierarchy_mappings

    # First aggregate counts across selected recorders
    counts_per_hour = {}
    for recorder in config.SELECTED_RECORDERS:
        recorder_data = data_counts.get(recorder, {})
        for hour in recorder_data:
            if hour not in counts_per_hour:
                counts_per_hour[hour] = {}
            for class_id, count in recorder_data[hour].items():
                if class_id not in counts_per_hour[hour]:
                    counts_per_hour[hour][class_id] = 0
                counts_per_hour[hour][class_id] += count

    # Get the hours from the actual data
    all_hours = sorted(counts_per_hour.keys(), key=lambda x: int(x))
    print(f"\nDebug: Hours from data: {all_hours}")

    # Process each class for each hour
    df_data = {}
    for hour in all_hours:
        df_data[hour] = {}
        for class_name in selected_classes:
            class_id = get_class_id(class_name, class_label_to_id, name_to_class_id)
            if class_id:
                # Get all subclasses using the same function as plot_results.py
                all_related_ids = get_all_subclasses(class_id, parent_to_children)
                class_count = sum(counts_per_hour[hour].get(cid, 0)
                                  for cid in all_related_ids)
                df_data[hour][class_name] = class_count
            else:
                print(f"Warning: Could not find ID for class {class_name}")
                df_data[hour][class_name] = 0

    return pd.DataFrame(df_data).T

def run_advanced_analysis(data_counts, selected_classes, threshold_str, recorder_info, normalize=False):
    """
    Main function to run advanced sound analyses.

    Args:
        data_counts (dict): Raw data counts per hour and class
        selected_classes (list): List of class names to analyze
        threshold_str (str): Confidence threshold used for detection
        recorder_info (str): Information about recorder configuration
        normalize (bool): Whether to normalize counts by active recorders
    """
    print("\n=== Advanced Sound Analysis Report ===")
    print(f"Analysis Parameters:")
    print(f"- Selected Classes: {selected_classes}")
    print(f"- Confidence Threshold: {threshold_str}")
    print(f"- Recorder Configuration: {recorder_info}")

    # Select which analyses to perform
    selected_analyses = {
        'time_series': True,
        'basic_stats': True,
        'correlation': True,
        'pca': True,  # Set to False to disable PCA analysis
        'heatmap_clustering': True,  # Set to False to disable heatmap and clustering
        'peak_activity': True,
    }

    try:
        # Load necessary ontology and mapping information
        ontology = load_ontology(config.ONTOLOGY_PATH)
        class_label_to_id, class_id_to_label = load_class_labels(config.CLASS_LABELS_CSV_PATH)
        id_to_class, name_to_class_id = build_mappings(ontology)
        parent_to_children, child_to_parents = build_parent_child_mappings(ontology)

        # Load active recorders information
        with open('analysis_results/recording_times/recorders_active_per_hour.json', 'r') as f:
            recorders_active = json.load(f)

        # Process data using the same logic as plot_results.py
        df = process_data(
            data_counts,
            selected_classes,
            (class_label_to_id, id_to_class, name_to_class_id),
            (parent_to_children, child_to_parents)
        )

        print("\nProcessed Data Sample:")
        print(df.head())
        print("\nData Shape:", df.shape)

        if df.empty:
            print("No data available for the selected classes.")
            return

        # Remove columns with constant values (e.g., all zeros)
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            print(f"\nThe following columns have constant values and will be excluded from certain analyses: {constant_cols}")
            df = df.drop(columns=constant_cols)

        if df.empty:
            print("No data available after removing constant columns.")
            return

        # Calculate active recorders ratio for x-axis labels
        total_recorders = len(config.SELECTED_RECORDERS)
        x_labels = []
        for hour in df.index:
            active_recorders_list = recorders_active.get(hour, [])
            active_recorders = [rec for rec in active_recorders_list if rec in config.SELECTED_RECORDERS]
            num_active = len(active_recorders)
            x_labels.append(f"{hour}\n({num_active}/{total_recorders})")

        # 1. Time Series Analysis
        if selected_analyses['time_series']:
            fig, ax = plt.subplots(figsize=(12, 6))
            for column in df.columns:
                ax.plot(range(len(df.index)), df[column], marker='o', label=column)
            ax.set_title(f'Sound Events per Hour\n{recorder_info}, Threshold: {threshold_str}')
            ax.set_xlabel('Hour (Active Recorders)')
            ax.set_ylabel('Event Count')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xticks(range(len(df.index)))
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            plt.tight_layout()
            save_plot(fig, 'temporal_distribution')
            plt.show()

        # 2. Basic Statistical Analysis
        if selected_analyses['basic_stats']:
            print("\nStatistical Analysis:")
            print("-----------------")
            stats = df.describe()
            print(stats)

        # 3. Correlation Analysis between sound classes
        if selected_analyses['correlation']:
            active_cols = df.columns[df.sum() > 0]
            if len(active_cols) > 1:
                correlation_matrix = df[active_cols].corr()

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix,
                            annot=True,
                            cmap='coolwarm',
                            center=0,
                            vmin=-1,
                            vmax=1,
                            square=True,
                            fmt=".2f",
                            ax=ax)
                ax.set_title('Sound Class Correlation Matrix')
                plt.tight_layout()
                save_plot(fig, 'correlation_matrix')
                plt.show()

                print("\nSignificant Correlations:")
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i + 1, len(correlation_matrix.columns)):
                        corr = correlation_matrix.iloc[i, j]
                        class1 = correlation_matrix.columns[i]
                        class2 = correlation_matrix.columns[j]
                        if abs(corr) > 0.5:
                            strength = "strong" if abs(corr) > 0.7 else "moderate"
                            direction = "positive" if corr > 0 else "negative"
                            print(f"- {class1} vs {class2}: {corr:.2f} ({strength} {direction} correlation)")

        # 4. Principal Component Analysis (PCA)
        if selected_analyses['pca']:
            from sklearn.decomposition import PCA

            # Standardize the data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)

            pca = PCA()
            pca_components = pca.fit_transform(df_scaled)

            # Plot cumulative explained variance ratio
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(1, len(pca.explained_variance_ratio_) + 1),
                    np.cumsum(pca.explained_variance_ratio_), marker='o')
            ax.set_title('Cumulative Explained Variance by PCA Components')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Explained Variance Ratio')
            ax.grid(True)
            save_plot(fig, 'pca_explained_variance')
            plt.show()

            print("\nPCA Analysis:")
            print("-----------------")
            for i, ratio in enumerate(pca.explained_variance_ratio_):
                print(f"Component {i+1}: {ratio:.4f} explained variance")

            # Print PCA loadings
            loadings = pd.DataFrame(pca.components_.T, index=df.columns, columns=[f'PC{i+1}' for i in range(len(df.columns))])
            print("\nPCA Loadings:")
            print(loadings)

            # Plot the first two principal components
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(pca_components[:, 0], pca_components[:, 1], c=range(len(df.index)), cmap='viridis')
            ax.set_title('PCA - First Two Principal Components')
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Hour Index')
            # Annotate points with hours
            for i, hour in enumerate(df.index):
                ax.annotate(hour, (pca_components[i, 0], pca_components[i, 1]))
            save_plot(fig, 'pca_scatter')
            plt.show()

        # 5. Heatmaps and Clustering
        if selected_analyses['heatmap_clustering']:
            cluster_grid = sns.clustermap(df.T, method='average', cmap='viridis', figsize=(12, 8), dendrogram_ratio=0.1)
            cluster_grid.fig.suptitle('Heatmap and Clustering of Sound Events', fontsize=16)
            # Adjust the height of the rows (classes)
            cluster_grid.ax_heatmap.set_yticklabels(cluster_grid.ax_heatmap.get_yticklabels(), rotation=0)
            cluster_grid.ax_heatmap.set_xticklabels(cluster_grid.ax_heatmap.get_xticklabels(), rotation=45)
            # Add units to colorbar
            cluster_grid.cax.set_title('Event Counts', fontsize=12)
            plt.tight_layout()
            save_path = os.path.join('assets', 'images', 'heatmap_clustering.png')
            cluster_grid.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to: {save_path}")
            plt.show()

        # 6. Peak Activity Analysis for each sound class
        if selected_analyses['peak_activity']:
            print("\nPeak Activity Analysis:")
            print("----------------------")
            for column in df.columns:
                if df[column].sum() > 0:
                    peak_hour = df[column].idxmax()
                    peak_value = df[column].max()
                    mean_value = df[column].mean()
                    print(f"\n{column}:")
                    print(f"- Peak hour: {peak_hour}")
                    print(f"- Maximum events: {peak_value:,}")
                    if mean_value > 0:
                        print(f"- Ratio to average: {peak_value/mean_value:.2f}x")

    except Exception as e:
        print(f"\nERROR: An error occurred during analysis:")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n=== End of Analysis Report ===")