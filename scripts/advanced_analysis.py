# scripts/advanced_analysis.py

"""
This script performs multiple analyses on sound event data:
1. Time Series Analysis
2. Basic Statistical Analysis
3. Correlation Analysis
4. Distribution Analysis
5. Peak Activity Analysis
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
 
def save_plot(plt, name):
    """
    Save plot to assets/images directory
    Args:
        plt: matplotlib.pyplot object
        name (str): Name for the output file
    """
    save_dir = os.path.join('assets', 'images')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{name}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {save_path}")
 
def process_data(data_counts, selected_classes, class_mappings, hierarchy_mappings):
    """    
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
        
        # Calculate active recorders ratio for x-axis labels
        total_recorders = len(config.SELECTED_RECORDERS)
        x_labels = []
        for hour in df.index:
            active_recorders_list = recorders_active.get(hour, [])
            active_recorders = [rec for rec in active_recorders_list if rec in config.SELECTED_RECORDERS]
            num_active = len(active_recorders)
            x_labels.append(f"{hour}\n({num_active}/{total_recorders})")
 
        # 1. Time Series Analysis
        plt.figure(figsize=(12, 6))
        for column in df.columns:
            plt.plot(range(len(df.index)), df[column], marker='o', label=column)
        plt.title(f'Sound Events per Hour ({recorder_info}, Threshold: {threshold_str})')
        plt.xlabel('Hour')
        plt.ylabel('Event Count')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(range(len(df.index)), x_labels, rotation=45, ha='right')
        plt.tight_layout()
        save_plot(plt, 'temporal_distribution')
        plt.show()
 
        # 2. Basic Statistical Analysis
        print("\nStatistical Analysis:")
        print("-----------------")
        stats = df.describe()
        print(stats)
 
        # 3. Correlation Analysis between sound classes
        active_cols = df.columns[df.sum() > 0]
        if len(active_cols) > 1:
            correlation_matrix = df[active_cols].corr()
           
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix,
                       annot=True,
                       cmap='coolwarm',
                       center=0,
                       vmin=-1,
                       vmax=1,
                       square=True)
            plt.title('Sound Class Correlation Matrix')
            plt.tight_layout()
            save_plot(plt, 'correlation_matrix')
            plt.show()
 
            print("\nSignificant correlations:")
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    corr = correlation_matrix.iloc[i, j]
                    class1 = correlation_matrix.columns[i]
                    class2 = correlation_matrix.columns[j]
                    strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
                    direction = "positive" if corr > 0 else "negative"
                    print(f"{class1} vs {class2}: {corr:.2f} ({strength} {direction} correlation)")
 
        # 4. Distribution Analysis using box plots
        plt.figure(figsize=(12, 6))
        df.boxplot()
        plt.title('Statistical Distribution of Sound Events by Class')
        plt.xticks(rotation=45)
        plt.ylabel('Event Count')
        plt.tight_layout()
        save_plot(plt, 'distribution_boxplot')
        plt.show()
 
        # 5. Peak Activity Analysis for each sound class
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