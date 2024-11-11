# scripts/granger.py

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
import time

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

# Define base directory
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'granger')
os.makedirs(BASE_DIR, exist_ok=True)

def main():
    import psutil
    from pathlib import Path
    print("\n=== Home Sounds Advanced Analysis ===")
    print("\nInitializing...")
    # Configuration
    predictions_root_dir = PREDICTIONS_ROOT_DIR  # Update if necessary
    confidence_threshold = 0.0  # Adjust as needed
    bin_size = 60  # Time bin size in seconds (e.g., 60 seconds for per-minute aggregation)
    selected_recorders = []  # List of recorder IDs to include; empty list means all
    normalize_by_recorders = False  # Set to True if you want to normalize counts
    start_time = time.time()

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

    print(f"Starting analysis at: {datetime.datetime.now()}")
    print(f"Expected completion in: {1344/4:.1f} hours")

    # Agregar checkpoints de tiempo
    def log_progress(msg):
        elapsed = (time.time() - start_time) / 3600
        print(f"\n[{elapsed:.1f}h] {msg}")

    def save_checkpoint(data, name):
        checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)  # Usar os.makedirs en lugar de mkdir
        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_{name}.pkl')  # Usar os.path.join en lugar de /
        pd.to_pickle(data, checkpoint_file, compression='gzip')
        print(f"\nSaved checkpoint: {checkpoint_file}")
    
    def check_memory():
        memory = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"\nCurrent memory usage: {memory:.0f} MB")

    # Step 1: Load ontology and class labels
    print("\n=== Loading Data ===")
    ontology = load_ontology(ONTOLOGY_PATH)
    class_label_to_id, class_id_to_label = load_class_labels(CLASS_LABELS_CSV_PATH)
    id_to_class, name_to_class_id = build_mappings(ontology)
    parent_to_children, child_to_parents = build_parent_child_mappings(ontology)

    # Step 2: Map HOME_SOUNDS classes to class IDs
    #print("Mapping HOME_SOUNDS classes to class IDs...")
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

    # Create a set of selected class names
    selected_class_names = set(selected_class_id_to_name.values())

    print(f"Total selected classes (including descendants): {len(selected_class_ids)}")

    # Step 3: Process the dataset
    print("Processing dataset...")
    log_progress("Starting data processing...")
    data_dir = predictions_root_dir
    df_all, available_classes = process_all_files(
        data_dir,
        selected_class_names,
        confidence_threshold,
        bin_size,
        selected_recorders
    )

    def validate_results(df_all, available_classes):
        # Validate unrealistic high frequencies
        suspicious_classes = []
        for cls in available_classes:
            occurrence_rate = (df_all[cls] > 0).mean() * 100
            if occurrence_rate > 75:  # Más del 75% del tiempo
                suspicious_classes.append((cls, occurrence_rate))
        
        if suspicious_classes:
            print("\nWARNING: Suspiciously high occurrence rates:")
            for cls, rate in suspicious_classes:
                print(f"- {cls}: {rate:.1f}% of time bins")
                
        # Validate memory scaling
        total_hours = len(df_all) / 60  # assuming 60s bins
        memory_per_hour = psutil.Process().memory_info().rss / (1024 * 1024 * total_hours)
        projected_memory = memory_per_hour * 1344  # total dataset hours
        
        print(f"\nMemory Projection:")
        print(f"Current: {memory_per_hour:.1f} MB/hour")
        print(f"Projected for full dataset: {projected_memory/1024:.1f} GB")
        
        if projected_memory/1024 > 100:  # Si proyección > 100GB
            print("WARNING: Memory usage might be too high for full dataset")

    validate_results(df_all, available_classes)

    # Add here:
    check_memory()
    save_checkpoint({'df_all': df_all, 'available_classes': available_classes}, 
                   'after_processing')

    # Check if data is available
    if df_all.empty:
        print("No data available after processing. Please check your configuration.")
        return
    
    # Analyze class frequencies
    class_frequencies = {}
    for cls in available_classes:
        non_zero_counts = (df_all[cls] > 0).sum()
        class_frequencies[cls] = non_zero_counts
    
    print("\nClass Occurrence Statistics:")
    for cls, freq in sorted(class_frequencies.items(), key=lambda x: x[1], reverse=True):
        print(f"{cls}: {freq} occurrences ({(freq/len(df_all))*100:.2f}% of time bins)")
    
    # Filter out sparse classes
    min_occurrence_threshold = len(df_all) * 0.05  # 5% threshold
    active_classes = [cls for cls, freq in class_frequencies.items() 
                     if freq >= min_occurrence_threshold]
    
    print(f"\nFiltering classes: {len(available_classes)} -> {len(active_classes)}")
    available_classes = set(active_classes)

    # Filter selected_class_id_to_name to only include available classes
    selected_class_id_to_name = {
        class_id: name 
        for class_id, name in selected_class_id_to_name.items() 
        if name in available_classes
    }

    if not selected_class_id_to_name:
        print("No selected classes found in the data. Please check class names.")
        return

    print(f"\nProceeding with analysis for {len(selected_class_id_to_name)} available classes:")
    for name in sorted(selected_class_id_to_name.values()):
        print(f"  - {name}")

    # Step 4: Perform data analysis
    print("\nPerforming data analysis...")
    # Aggregate data over all recorders and time bins
    available_columns = list(available_classes)
    df_agg = df_all.groupby(['time_bin'])[available_columns].sum().reset_index()

    # Normalize by number of recorders if needed
    if normalize_by_recorders and selected_recorders:
        num_recorders = len(selected_recorders)
        df_agg[selected_class_id_to_name.values()] /= num_recorders

    # Ensure the time series is sorted by time
    df_agg.sort_values('time_bin', inplace=True)

    # Set 'time_bin' as index
    df_agg.set_index('time_bin', inplace=True)

    # Step 5: Advanced Analyses
    log_progress("Starting statistical analysis...")
    # Analysis 1: Time Series Analysis using ARIMA models
    arima_models = fit_arima_models(df_agg, selected_class_id_to_name)

    # Analysis 2: Cross-Correlation Functions with Lag Analysis
    cross_correlations = compute_cross_correlations(df_agg, selected_class_id_to_name)

    # Analysis 3: Granger Causality Tests
    granger_results = perform_granger_causality_tests(df_agg, selected_class_id_to_name)

    # Analysis 4: Principal Component Analysis (PCA)
    pca_results = perform_pca(df_agg, selected_class_id_to_name)
    save_checkpoint({'pca_results': pca_results}, 'pca')

    # Step 6: Visualization
    log_progress("Generating visualizations...")
    plot_time_series(df_agg, selected_class_id_to_name)
    correlation_matrix = compute_correlations(df_agg, selected_class_id_to_name)
    save_checkpoint({'correlation_matrix': correlation_matrix}, 'correlations')
    plot_correlation_matrix(correlation_matrix)
    plot_cross_correlations(cross_correlations)
    plot_pca_results(pca_results, selected_class_id_to_name)

    # Save analysis summary
    summary = {
        'dataset': {
            'total_classes': len(available_classes),
            'classes_analyzed': list(available_classes)
        },
        'time_series': {
            'bin_size': bin_size,
            'total_bins': len(df_agg),
            'non_empty_bins_percent': (df_agg > 0).mean().to_dict()
        },
        'models': {
            'successful_arima_fits': len(arima_models),
            'pca_variance': float(pca_results['explained_variance'][0])
        }
    }
    
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
    with open(os.path.join(BASE_DIR, 'results', 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    total_time = (time.time() - start_time) / 3600
    print(f"\nFinal processing time: {total_time:.1f} hours")

def process_all_files(data_dir, selected_class_names, confidence_threshold, bin_size, selected_recorders):
    from pathlib import Path
    import gc  # Para garbage collection
    
    total_hours = 0
    total_files = 0
    df_buffer = []  # Buffer para almacenar DataFrames temporales
    buffer_size = min(50, total_files // 20)  # Máximo 20 chunks
    available_classes = set()
    
    # First pass: count total files
    print("\nCounting files...")
    total_files = sum(1 for recorder in os.listdir(data_dir) 
                     if os.path.isdir(os.path.join(data_dir, recorder))
                     for _ in os.listdir(os.path.join(data_dir, recorder))
                     if _.endswith('.json'))
    
    processed_files = 0
    print(f"\nProcessing {total_files} files...")
    
    # Loop through recorder directories
    for recorder in os.listdir(data_dir):
        recorder_dir = os.path.join(data_dir, recorder)
        if os.path.isdir(recorder_dir):
            if selected_recorders and recorder not in selected_recorders:
                continue
                
            print(f"\nProcessing recorder: {recorder}")
            
            # Create a temporary list for this recorder
            recorder_data = []
            
            for filename in os.listdir(recorder_dir):
                if filename.endswith('.json'):
                    processed_files += 1
                    file_path = os.path.join(recorder_dir, filename)
                    
                    # Update progress every 10 files
                    if processed_files % 10 == 0:
                        print(f"Progress: {processed_files}/{total_files} files ({processed_files/total_files*100:.1f}%)")
                        
                    try:
                        frames_data = load_and_filter_predictions(
                            file_path,
                            selected_class_names,
                            confidence_threshold
                        )
                        
                        if frames_data:
                            # Update available classes
                            for frame in frames_data:
                                available_classes.update(k for k in frame.keys() if k != 'time')
                            
                            df_agg = aggregate_frames(frames_data, bin_size)
                            if df_agg is not None:
                                recorder_data.append(df_agg)
                                
                            # Process buffer if full
                            if len(recorder_data) >= buffer_size:
                                temp_df = pd.concat(recorder_data, ignore_index=True)
                                df_buffer.append(temp_df)
                                recorder_data = []  # Clear buffer
                                gc.collect()  # Force garbage collection
                                
                    except Exception as e:
                        print(f"\nError processing file {file_path}: {str(e)}")
                        continue
            
            # Process remaining files in recorder
            if recorder_data:
                temp_df = pd.concat(recorder_data, ignore_index=True)
                df_buffer.append(temp_df)
                recorder_data = []
                gc.collect()
    
    # Final processing
    if df_buffer:
        print("\nConcatenating all processed data...")
        df_all = pd.concat(df_buffer, ignore_index=True)
        del df_buffer  # Free memory
        gc.collect()
        
        print(f"Found {len(available_classes)} classes in data")
        print("\nClasses found:")
        for cls in sorted(available_classes):
            print(f"- {cls}")
        return df_all, available_classes
    else:
        print("\nNo data was processed successfully")
        return pd.DataFrame(), set()

def load_and_filter_predictions(file_path, selected_class_names, confidence_threshold):
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
            # Filter predictions based on selected class names and confidence threshold
            filtered_preds = {
                pred['class']: pred['prob']
                for pred in predictions
                if pred['class'] in selected_class_names and pred['prob'] >= confidence_threshold
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
        
    # Add adaptive binning based on data sparsity
    total_duration = (df['time'].max() - df['time'].min()).total_seconds()
    if total_duration < 3600:  # Less than 1 hour of data
        print("Warning: Limited data detected in file")
        
    # Create time bins
    df['time_bin'] = df['time'].dt.floor(f'{bin_size}S')
    
    # Count number of samples per bin
    samples_per_bin = df.groupby('time_bin').size()
    
    # Print bin statistics for this file
    print(f"\nBin statistics for file:")
    print(f"- Average samples per bin: {samples_per_bin.mean():.2f}")
    print(f"- Empty bins: {(samples_per_bin == 0).sum()}")
    print(f"- Total bins: {len(samples_per_bin)}")
    
    # Sum probabilities within each bin
    aggregation_columns = df.columns.difference(['time', 'time_bin'])
    df_agg = df.groupby('time_bin')[aggregation_columns].sum().reset_index()
    
    return df_agg

def fit_arima_models(df_agg, selected_class_id_to_name):
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    arima_models = {}
    print("\nTime Series Analysis (frame interval: {:.3f}s)".format((320 * 32) / 48000))

    df_agg.index = pd.date_range(start=df_agg.index[0], periods=len(df_agg), freq='213333U')  # microseconds
    
    for cls in selected_class_id_to_name.values():
        ts = df_agg[cls].fillna(0)
        
        # Check minimum data requirements
        if ts.std() == 0:
            print(f"✗ {cls} - No variation in data")
            continue
            
        # Check for sufficient non-zero values
        non_zero_percent = (ts > 0).sum() / len(ts) * 100
        if non_zero_percent < 5:  # Less than 5% non-zero
            print(f"✗ {cls} - Too sparse ({non_zero_percent:.1f}% non-zero)")
            continue
        
        try:
            # Check stationarity
            adf_result = adfuller(ts)
            is_stationary = adf_result[1] < 0.05
            
            # Fit model
            order = (1,0,0) if is_stationary else (1,1,0)
            model = ARIMA(ts, order=order)
            model_fit = model.fit()
            arima_models[cls] = model_fit
            print(f"✓ {cls} - {'Stationary' if is_stationary else 'Non-stationary'}")
        except Exception as e:
            print(f"✗ {cls}")
            
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
    return cross_correlations

def perform_granger_causality_tests(df_agg, selected_class_id_to_name, maxlag=5):
    from statsmodels.tsa.stattools import grangercausalitytests
    granger_results = {}
    print("\n3. Granger Causality Tests")
    
    for i, cls1 in enumerate(selected_class_id_to_name.values()):
        for j, cls2 in enumerate(selected_class_id_to_name.values()):
            if i != j:
                data = df_agg[[cls1, cls2]].fillna(0)
                # Solo testar si hay suficiente variación
                if data[cls1].std() > 0 and data[cls2].std() > 0:
                    try:
                        test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                        granger_results[(cls1, cls2)] = test_result
                    except:
                        continue
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
    os.makedirs(os.path.join(BASE_DIR, 'figures'), exist_ok=True)
    class_names = list(selected_class_id_to_name.values())
    plt.figure(figsize=(12, 6))
    for cls in class_names:
        plt.plot(df_agg.index, df_agg[cls], label=cls)
    plt.xlabel('Time')
    plt.ylabel('Aggregated Probability')
    plt.title('Event Probabilities Over Time')
    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, 'figures', 'time_series.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(correlation_matrix):
    os.makedirs(os.path.join(BASE_DIR, 'figures'), exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(BASE_DIR, 'figures', 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_cross_correlations(cross_correlations):
    print("\nAnalyzing cross-correlations...")
    
    # Calculate significance threshold
    conf_int = 1.96/np.sqrt(10) * 1.5  # 50% más estricto
    
    # Create summary of significant correlations
    significant_correlations = []
    
    for (cls1, cls2), (lags, ccf_values) in cross_correlations.items():
        max_corr = float(np.max(np.abs(ccf_values)))  # Convert to Python float
        max_lag = int(lags[np.argmax(np.abs(ccf_values))])  # Convert to Python int
        
        if max_corr > conf_int:
            significant_correlations.append({
                'class1': str(cls1),
                'class2': str(cls2),
                'correlation': float(max_corr),  # Ensure it's a Python float
                'lag': int(max_lag)  # Ensure it's a Python int
            })
    
    # Sort by correlation strength
    significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    # Save summary
    os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
    with open(os.path.join(BASE_DIR, 'results', 'significant_correlations.json'), 'w') as f:
        summary = {
            'confidence_threshold': float(conf_int),  # Convert to Python float
            'significant_pairs': significant_correlations
        }
        json.dump(summary, f, indent=2)
    
    # Print summary of top correlations
    print(f"\nFound {len(significant_correlations)} significant correlations")
    print("\nTop 10 strongest correlations:")
    for i, corr in enumerate(significant_correlations[:10]):
        print(f"{i+1}. {corr['class1']} - {corr['class2']}: "
              f"correlation = {corr['correlation']:.3f} at lag {corr['lag']}")

def plot_pca_results(pca_results, selected_class_id_to_name):
    os.makedirs(os.path.join(BASE_DIR, 'figures'), exist_ok=True)
    principal_components = pca_results['principal_components']
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
    plt.title('PCA of Selected Classes')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.savefig(os.path.join(BASE_DIR, 'figures', 'pca_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    from contextlib import redirect_stdout
    
    # Create logs directory
    log_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'analysis_log_{timestamp}.txt')
    
    with open(log_file, 'w') as f:
        with redirect_stdout(f):
            main()
            
    # Print to console that the log was saved
    print(f"\nLog saved to: {log_file}")