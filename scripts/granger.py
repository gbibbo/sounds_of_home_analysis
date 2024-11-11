# scripts/home_sounds_advanced_analysis.py

"""
scripts/home_sounds_advanced_analysis.py

This script performs advanced analyses on the selected HOME_SOUNDS classes.
It utilizes existing functions from utils.py and load_data.py to work
with the ontology and data.

The analyses include:
- Aggregating event counts per class over time
- Time Series Analysis using ARIMA models
- Cross-Correlation Functions with Lag Analysis
- Granger Causality Tests
- Principal Component Analysis (PCA) for Dimensionality Reduction
- Visualizing time series, correlation matrices, and cross-correlations

Instructions:
- Ensure that utils.py and load_data.py are correctly set up and accessible.
- Run this script after configuring the parameters as needed.
"""

import os
import sys
import json
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path to import utils and load_data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import PREDICTIONS_ROOT_DIR, ONTOLOGY_PATH, CLASS_LABELS_CSV_PATH
from src.data_processing.utils import (
    get_class_id,
    get_all_subclasses,
    extract_datetime_from_filename,
)
from src.data_processing.load_data import (
    load_ontology,
    load_class_labels,
    build_mappings,
    build_parent_child_mappings,
)

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

def main():
    # Configuration
    predictions_root_dir = PREDICTIONS_ROOT_DIR  # Update if necessary
    confidence_threshold = 0.5  # Adjust as needed
    bin_size = 60  # Time bin size in seconds (e.g., 60 seconds for per-minute aggregation)
    selected_recorders = []  # List of recorder IDs to include; empty list means all
    normalize_by_recorders = False  # Set to True if you want to normalize counts

    # HOME_SOUNDS classes provided
    HOME_SOUNDS = {
        "Human voice": {},
        "Domestic sounds, home sounds": {},
        "Domestic animals, pets": {},
        "Mechanisms": {},
        "Liquid": {},
        #########################################
        "Speech": {},
        "Door": {},
        "Doorbell": {},
        "Clock": {},
        "Telephone": {},
        "Pour": {},
        "Glass": {},
        "Splash, splatter": {},
        "Cat": {},
        "Dog": {},
        "Power tool": {},
        #########################################
        "Television": {},
        "Radio": {},
        "Conversation": {},
        "Water tap, faucet": {},
        "Dishes, pots, and pans": {},
        "Cutlery, silverware": {},
        "Walk, footsteps": {},
        "Toilet flush": {},
        "Sink (filling or washing)": {},
        "Cupboard open or close": {},
        "Drawer open or close": {},
        "Microwave oven": {},
        "Frying (food)": {},
        "Chopping (food)": {},
        "Bathtub (filling or washing)": {},
        "Alarm clock": {},
        "Whispering": {},
        "Air conditioning": {},
        "Mechanical fan": {},
        "Hair dryer": {},
        "Vacuum cleaner": {},
        "Blender": {},
        "Stir": {},
        "Keys jangling": {},
        "Computer keyboard": {},
        "Writing": {},
        "Electric shaver, electric razor": {},
        "Zipper (clothing)": {}
    }

    # Step 1: Load ontology and class labels
    print("Loading ontology and class labels...")
    ontology = load_ontology(ONTOLOGY_PATH)
    class_label_to_id, class_id_to_label = load_class_labels(CLASS_LABELS_CSV_PATH)
    id_to_class, name_to_class_id = build_mappings(ontology)
    parent_to_children, child_to_parents = build_parent_child_mappings(ontology)

    # Step 2: Map HOME_SOUNDS classes to class IDs
    print("Mapping HOME_SOUNDS classes to class IDs...")
    selected_class_ids = set()
    for class_name in HOME_SOUNDS.keys():
        class_id = get_class_id(class_name, class_label_to_id, name_to_class_id)
        if class_id:
            # Get all subclasses (descendants) including the class itself
            subclasses = get_all_subclasses(class_id, parent_to_children)
            selected_class_ids.update(subclasses)
        else:
            print(f"Class '{class_name}' not found in ontology.")

    # Map class IDs back to class names for reference
    selected_class_id_to_name = {class_id: id_to_class[class_id]['name'] for class_id in selected_class_ids}

    print(f"Total selected classes (including descendants): {len(selected_class_ids)}")

    # Step 3: Process the dataset
    print("Processing dataset...")
    data_dir = predictions_root_dir
    df_all = process_all_files(
        data_dir,
        selected_class_ids,
        class_id_to_label,
        confidence_threshold,
        bin_size,
        selected_recorders
    )

    # Check if data is available
    if df_all.empty:
        print("No data available after processing. Please check your configuration.")
        return

    # Step 4: Perform data analysis
    print("Performing data analysis...")
    # Aggregate data over all recorders and time bins
    df_agg = df_all.groupby(['time_bin'])[selected_class_id_to_name.values()].sum().reset_index()

    # Normalize by number of recorders if needed
    if normalize_by_recorders and selected_recorders:
        num_recorders = len(selected_recorders)
        df_agg[selected_class_id_to_name.values()] /= num_recorders

    # Ensure the time series is sorted by time
    df_agg.sort_values('time_bin', inplace=True)

    # Set 'time_bin' as index
    df_agg.set_index('time_bin', inplace=True)

    # Step 5: Advanced Analyses

    # Analysis 1: Time Series Analysis using ARIMA models
    arima_models = fit_arima_models(df_agg, selected_class_id_to_name)

    # Analysis 2: Cross-Correlation Functions with Lag Analysis
    cross_correlations = compute_cross_correlations(df_agg, selected_class_id_to_name)

    # Analysis 3: Granger Causality Tests
    granger_results = perform_granger_causality_tests(df_agg, selected_class_id_to_name)

    # Analysis 4: Principal Component Analysis (PCA)
    pca_results = perform_pca(df_agg, selected_class_id_to_name)

    # Step 6: Visualization
    plot_time_series(df_agg, selected_class_id_to_name)
    correlation_matrix = compute_correlations(df_agg, selected_class_id_to_name)
    plot_correlation_matrix(correlation_matrix)
    plot_cross_correlations(cross_correlations)
    plot_pca_results(pca_results, selected_class_id_to_name)

    # Additional analyses and visualizations can be added here

def process_all_files(data_dir, selected_class_ids, class_id_to_label, confidence_threshold, bin_size, selected_recorders):
    df_list = []
    # Loop through recorder directories
    for recorder in os.listdir(data_dir):
        recorder_dir = os.path.join(data_dir, recorder)
        if os.path.isdir(recorder_dir):
            if selected_recorders and recorder not in selected_recorders:
                continue  # Skip recorders not in the selected list
            print(f"Processing recorder: {recorder}")
            # Loop through JSON files in recorder directory
            for filename in os.listdir(recorder_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(recorder_dir, filename)
                    frames_data = load_and_filter_predictions(
                        file_path,
                        selected_class_ids,
                        confidence_threshold,
                        class_id_to_label
                    )
                    if frames_data:
                        df_agg = aggregate_frames(frames_data, bin_size)
                        if df_agg is not None:
                            df_list.append(df_agg)
    if df_list:
        df_all = pd.concat(df_list, ignore_index=True)
    else:
        df_all = pd.DataFrame()
    return df_all

def load_and_filter_predictions(file_path, selected_class_ids, confidence_threshold, class_id_to_label):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract datetime from filename
    filename = os.path.basename(file_path)
    file_datetime = extract_datetime_from_filename(filename)
    if not file_datetime:
        return None

    frames_data = []
    for frame in data:
        if 'predictions' in frame:
            frame_time = frame['time']
            predictions = frame['predictions']
            # Filter predictions based on selected class IDs and confidence threshold
            filtered_preds = {
                class_id_to_label[pred['class']]: pred['prob']
                for pred in predictions
                if pred['class'] in selected_class_ids and pred['prob'] >= confidence_threshold
            }
            if filtered_preds:
                # Adjust time to absolute datetime
                absolute_time = file_datetime + datetime.timedelta(seconds=frame_time)
                frames_data.append({'time': absolute_time, **filtered_preds})
    return frames_data

def aggregate_frames(frames_data, bin_size):
    df = pd.DataFrame(frames_data)
    if df.empty:
        return None
    # Create time bins
    df['time_bin'] = df['time'].dt.floor(f'{bin_size}S')
    # Sum probabilities within each bin
    aggregation_columns = df.columns.difference(['time', 'time_bin'])
    df_agg = df.groupby('time_bin')[aggregation_columns].sum().reset_index()
    return df_agg

def fit_arima_models(df_agg, selected_class_id_to_name):
    from statsmodels.tsa.arima.model import ARIMA
    arima_models = {}
    print("\n--- Analysis 1: Time Series Analysis using ARIMA models ---")
    for cls in selected_class_id_to_name.values():
        try:
            ts = df_agg[cls]
            # Differencing to make the series stationary if needed
            ts_diff = ts.diff().dropna()
            # Fit ARIMA model (order can be tuned based on data)
            model = ARIMA(ts_diff, order=(1, 0, 1))
            model_fit = model.fit()
            arima_models[cls] = model_fit
            print(f"ARIMA model fitted for class '{cls}'.")
        except Exception as e:
            print(f"Could not fit ARIMA model for '{cls}': {e}")
    return arima_models

def compute_cross_correlations(df_agg, selected_class_id_to_name, max_lag=10):
    from statsmodels.tsa.stattools import ccf
    cross_correlations = {}
    print("\n--- Analysis 2: Cross-Correlation Functions with Lag Analysis ---")
    class_names = list(selected_class_id_to_name.values())
    for i in range(len(class_names)):
        for j in range(i+1, len(class_names)):
            cls1 = class_names[i]
            cls2 = class_names[j]
            ts1 = df_agg[cls1] - df_agg[cls1].mean()
            ts2 = df_agg[cls2] - df_agg[cls2].mean()
            # Compute cross-correlation function
            ccf_values = ccf(ts1, ts2)[:max_lag]
            lags = np.arange(max_lag)
            cross_correlations[(cls1, cls2)] = (lags, ccf_values)
            print(f"Computed cross-correlation between '{cls1}' and '{cls2}'.")
    return cross_correlations

def perform_granger_causality_tests(df_agg, selected_class_id_to_name, maxlag=5):
    from statsmodels.tsa.stattools import grangercausalitytests
    granger_results = {}
    print("\n--- Analysis 3: Granger Causality Tests ---")
    class_names = list(selected_class_id_to_name.values())
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                cls1 = class_names[i]
                cls2 = class_names[j]
                data = df_agg[[cls1, cls2]].dropna()
                # Perform Granger causality test
                try:
                    test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                    granger_results[(cls1, cls2)] = test_result
                    print(f"Granger causality test performed between '{cls1}' and '{cls2}'.")
                except Exception as e:
                    print(f"Granger causality test failed between '{cls1}' and '{cls2}': {e}")
    return granger_results

def perform_pca(df_agg, selected_class_id_to_name):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    print("\n--- Analysis 4: Principal Component Analysis (PCA) ---")
    class_names = list(selected_class_id_to_name.values())
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_agg[class_names])
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by the first two principal components: {explained_variance}")
    pca_results = {
        'principal_components': principal_components,
        'explained_variance': explained_variance,
        'pca_model': pca
    }
    return pca_results

def compute_correlations(df_agg, selected_class_id_to_name):
    class_names = list(selected_class_id_to_name.values())
    correlation_matrix = df_agg[class_names].corr(method='pearson')
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    return correlation_matrix

def plot_time_series(df_agg, selected_class_id_to_name):
    class_names = list(selected_class_id_to_name.values())
    plt.figure(figsize=(12, 6))
    for cls in class_names:
        plt.plot(df_agg.index, df_agg[cls], label=cls)
    plt.xlabel('Time')
    plt.ylabel('Aggregated Probability')
    plt.title('Event Probabilities Over Time')
    plt.legend()
    plt.show()

def plot_correlation_matrix(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_cross_correlations(cross_correlations):
    print("\nPlotting Cross-Correlation Functions...")
    for (cls1, cls2), (lags, ccf_values) in cross_correlations.items():
        plt.figure()
        plt.stem(lags, ccf_values, use_line_collection=True)
        plt.xlabel('Lag')
        plt.ylabel('Cross-correlation')
        plt.title(f'Cross-correlation between {cls1} and {cls2}')
        plt.show()

def plot_pca_results(pca_results, selected_class_id_to_name):
    principal_components = pca_results['principal_components']
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
    plt.title('PCA of Selected Classes')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
