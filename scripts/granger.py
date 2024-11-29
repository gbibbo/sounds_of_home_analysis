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
from collections import defaultdict, Counter
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap.umap_ as umap

# Add the parent directory to sys.path to import utils and load_data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import CONFIDENCE_THRESHOLD, ONTOLOGY_PATH, CLASS_LABELS_CSV_PATH
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

from data_preparation_granger import load_prepared_data

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Define base directory
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'granger')
AGGREGATION_LEVELS = ['all', 'by_day', 'by_recorder']
os.makedirs(BASE_DIR, exist_ok=True)

def main():
    import psutil
    from pathlib import Path
    print("\n=== Home Sounds Advanced Analysis ===")
    print("\nInitializing...")
    # Configuration
    bin_size = 60  # Time bin size in seconds (e.g., 60 seconds for per-minute aggregation)
    selected_recorders = []  # List of recorder IDs to include; empty list means all
    normalize_by_recorders = False  # Set to True if you want to normalize counts
    start_time = time.time()

    # Agregar checkpoints de tiempo
    def log_progress(msg):
        elapsed = (time.time() - start_time) / 3600
        print(f"\n[{elapsed:.1f}h] {msg}")

    def save_checkpoint(data, name):
        checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints', aggregation_level)
        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_{name}_{aggregation_level}.pkl')
        os.makedirs(checkpoint_dir, exist_ok=True)  # Usar os.makedirs en lugar de mkdir
        pd.to_pickle(data, checkpoint_file, compression='gzip')
        print(f"\nSaved checkpoint: {checkpoint_file}")

    # HOME_SOUNDS classes provided
    HOME_SOUNDS = {
        # "Human voice": {},
        # "Domestic sounds, home sounds": {},
        # "Domestic animals, pets": {},
        # "Mechanisms": {},
        # "Liquid": {},
        # #########################################
        # "Speech": {},
        # "Door": {},
        # "Doorbell": {},
        # "Clock": {},
        # "Telephone": {},
        # "Pour": {},
        # "Glass": {},
        # "Splash, splatter": {},
        # "Cat": {},
        # "Dog": {},
        # "Power tool": {},
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

    # Step 1: Load ontology and class labels
    print("\n=== Loading Data ===")
    ontology = load_ontology(ONTOLOGY_PATH)
    class_label_to_id, class_id_to_label = load_class_labels(CLASS_LABELS_CSV_PATH)

    # Overwrite HOME_SOUNDS to include all classes
    HOME_SOUNDS = {class_name: {} for class_name in class_label_to_id.keys()}

    print(f"Starting analysis at: {datetime.datetime.now()}")

    # Step 1: Load ontology and class labels
    print("\n=== Loading Data ===")
    #ontology = load_ontology(ONTOLOGY_PATH)
    class_label_to_id, class_id_to_label = load_class_labels(CLASS_LABELS_CSV_PATH)
    #id_to_class, name_to_class_id = build_mappings(ontology)
    #parent_to_children, child_to_parents = build_parent_child_mappings(ontology)

    for aggregation_level in AGGREGATION_LEVELS:
        print("\n" + "=" * 80)
        print(f"Starting analysis for aggregation level: {aggregation_level}")
        print(f"Timestamp: {datetime.datetime.now()}")
        print("=" * 80 + "\n")

        # Step 2: Map HOME_SOUNDS classes to class IDs
        selected_class_id_to_name = class_id_to_label
        selected_class_names = set(class_id_to_label.values())
        print(f"Total selected classes: {len(selected_class_names)}")

        # selected_class_ids = set()
        # for class_name in HOME_SOUNDS.keys():
        #     class_id = get_class_id(class_name, class_label_to_id, name_to_class_id)
        #     if class_id:
        #         subclasses = get_all_subclasses(class_id, parent_to_children)
        #         selected_class_ids.update(subclasses)
        #     else:
        #         print(f"Class '{class_name}' not found in ontology.")
        # Map class IDs back to class names
        #selected_class_id_to_name = {class_id: id_to_class[class_id]['name'] for class_id in selected_class_ids}
        #selected_class_names = set(selected_class_id_to_name.values())
        #print(f"Total selected classes (including descendants): {len(selected_class_ids)}")

        # Step 3: Load the prepared data
        print("Loading prepared dataset...")
        log_progress("Starting data loading...")

        #aggregation_level = 'all'  # 'all', 'by_day', 'by_recorder'
        normalize_data = True      # True para normalizar datos

        # Cargar los datos preparados
        df_agg = load_prepared_data(aggregation_level, normalize=normalize_data)

        # Verificar si los datos están disponibles
        if df_agg.empty:
            print("No data available after loading. Please check your configuration.")
            return

        # Rename class ID columns to class names
        df_agg.rename(columns=class_id_to_label, inplace=True)

        # Set index name to 'time_bin'
        df_agg.index.name = 'time_bin'

        # Update 'available_classes' based on the columns present in df_agg
        available_classes = [col for col in df_agg.columns if col in selected_class_names]

        # Filter df_agg to include only the selected classes
        df_agg = df_agg[available_classes]

        # Ensure the time series is sorted by time
        df_agg.sort_index(inplace=True)

        # Calcular presence_stats directamente desde df_agg
        presence_stats = {}
        total_time_bins = len(df_agg)
        for class_name in available_classes:
            total_detections = df_agg[class_name].sum()
            presence_ratio = (df_agg[class_name] > 0).sum() / total_time_bins
            presence_stats[class_name] = {
                'total_detections': int(total_detections),
                'presence_ratio': float(presence_ratio)
            }

        print(f"Aggregation Level: {aggregation_level}")
        if aggregation_level == 'by_day':
            print("Data is aggregated by day.")
            # Optionally, print the days included in the analysis
            print(f"Days included in analysis: {df_agg.index.get_level_values('date').unique()}")
        elif aggregation_level == 'by_recorder':
            print("Data is aggregated by day and recorder.")
            # Optionally, print the recorders included
            print(f"Recorders included in analysis: {df_agg.index.get_level_values('recorder_id').unique()}")
        else:
            print("Data is aggregated across all data.")

        # Imprimir estadísticas detalladas
        print("\nSound Presence Statistics:")
        print("=" * 80)
        print(f"{'Class Name':<30} {'Presence %':<12} {'Total Detections'}")
        print("-" * 80)

        for class_name, stats in sorted(
            presence_stats.items(), 
            key=lambda x: x[1]['presence_ratio'], 
            reverse=True
        ):
            print(f"{class_name:<30} {stats['presence_ratio']*100:>8.2f}%    {stats['total_detections']:>8}")

        # Continuar con el análisis
        print(f"\nProceeding with analysis for {len(available_classes)} available classes:")
        for name in sorted(available_classes):
            print(f"  - {name}")

        # Step 4: Perform data analysis
        print("\nPerforming data analysis...")

        # Normalizar si es necesario
        if normalize_by_recorders and selected_recorders:
            num_recorders = len(selected_recorders)
            df_agg[available_classes] /= num_recorders

        # Asegurarse de que la serie temporal esté ordenada por tiempo
        df_agg.sort_index(inplace=True)

        # Set 'time_bin' as index
        print("Rango de tiempo en df_agg:")
        print(f"Inicio: {df_agg.index.min()}")
        print(f"Fin: {df_agg.index.max()}")
        print(f"Total de registros: {len(df_agg)}")
        print(f"Tipo de índice de df_agg: {type(df_agg.index)}")
        print(f"Frecuencia del índice antes de inferir: {df_agg.index.freq}")
        print(f"Primeros 5 valores del índice:\n{df_agg.index[:5]}")
        print(f"Diferencia entre los dos primeros índices: {df_agg.index[1] - df_agg.index[0]}")
        print("Conteo de registros por marca de tiempo en df_agg:")
        print(df_agg.index.value_counts())
        df_agg.index.freq = pd.infer_freq(df_agg.index)
        print(f"Frecuencia del índice después de inferir: {df_agg.index.freq}")

        # Step 5: Advanced Analyses
        log_progress("Starting statistical analysis...")
        summary = {}
        # Analysis 1: Time Series Analysis using ARIMA models
        arima_models = fit_arima_models(df_agg, selected_class_id_to_name)
        print(f"df_agg index after fit_arima_models:\n{df_agg.index}")

        # Analysis 2: Cross-Correlation Functions with Lag Analysis
        cross_correlations = compute_cross_correlations(df_agg, selected_class_id_to_name)
        cross_corr_results = plot_cross_correlations(cross_correlations, len(df_agg), aggregation_level)
        if cross_corr_results is None:
            print("Warning: plot_cross_correlations did not return results.")
            cross_corr_results = {
                'significant_correlations': [],
                'confidence_threshold': None
            }
        # Add correlation analysis to summary
        summary['correlation_analysis'] = {
            'significant_correlations': cross_corr_results['significant_correlations'],
            'total_correlations_found': len(cross_corr_results['significant_correlations']),
            'confidence_threshold': cross_corr_results['confidence_threshold']
        }

        # Analysis 3: Granger Causality Tests
        granger_results = perform_granger_causality_tests(df_agg, selected_class_id_to_name, aggregation_level)

        # Analysis 4: Principal Component Analysis (PCA)
        pca_results = perform_pca(df_agg, selected_class_id_to_name)
        save_checkpoint({'pca_results': pca_results}, 'pca')

        # Analysis 5: UMAP
        umap_results = perform_umap(df_agg, selected_class_id_to_name)
        save_checkpoint({'umap_results': umap_results}, 'umap')

        # Analysis 6: t-SNE
        tsne_results = perform_tsne(df_agg, selected_class_id_to_name)
        save_checkpoint({'tsne_results': tsne_results}, 'tsne')

        # Step 6: Visualization
        log_progress("Generating visualizations...")
        plot_time_series(df_agg, selected_class_id_to_name, aggregation_level)
        correlation_matrix = compute_correlations(df_agg, selected_class_id_to_name)
        save_checkpoint({'correlation_matrix': correlation_matrix}, 'correlations')
        plot_correlation_matrix(correlation_matrix, aggregation_level)
        plot_cross_correlations(cross_correlations, len(df_agg), aggregation_level)
        plot_pca_results(pca_results, selected_class_id_to_name, aggregation_level)
        plot_umap_results(umap_results, df_agg, selected_class_id_to_name, aggregation_level)
        plot_tsne_results(tsne_results, df_agg, selected_class_id_to_name, aggregation_level)

        #df_agg = df_agg.asfreq('T')
        # Create UMAP animation over time
        print(f"df_agg before animation:\n{df_agg.head()}")
        print(f"df_agg index frequency: {df_agg.index.freq}")
        ani = create_umap_animation(df_agg, selected_class_id_to_name, aggregation_level, window_minutes=15)
        if ani is not None:
            # Save the animation in GIF format
            ani.save(os.path.join(BASE_DIR, 'figures', aggregation_level, f'umap_animation_{aggregation_level}.gif'), writer='imagemagick')
        else:
            print("UMAP animation was not generated due to lack of frames.")

        # Save analysis summary
        summary.update({
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
        })
        # Add presence statistics with thresholds
        summary['presence_stats'] = {
            class_name: {
                'presence_ratio': float(stats['presence_ratio']),
                'total_detections': int(stats['total_detections'])
            }
            for class_name, stats in presence_stats.items()
        }

        # Add correlation analysis
        summary['correlation_analysis'] = {
            'significant_correlations': cross_corr_results['significant_correlations'],
            'total_correlations_found': len(cross_corr_results['significant_correlations']),
            'confidence_threshold': cross_corr_results['confidence_threshold']
        }

        # Add Granger causality results
        if granger_results:
            summary['granger_analysis'] = {
                'total_relationships_tested': granger_results['summary']['total_tests'],
                'significant_relationships': granger_results['summary']['significant_relationships'],
                'most_causal_sounds': [k for k, v in Counter(
                    rel['cause'] for rel in granger_results['significant_relationships']
                ).most_common(5)],
                'most_dependent_sounds': [k for k, v in Counter(
                    rel['effect'] for rel in granger_results['significant_relationships']
                ).most_common(5)]
            }
        
        os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
        with open(os.path.join(BASE_DIR, 'results', f'analysis_summary_{aggregation_level}.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        total_time = (time.time() - start_time) / 3600
        print(f"\nFinal processing time: {total_time:.1f} hours")

def fit_arima_models(df_agg, selected_class_id_to_name):
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    arima_models = {}
    print("\nTime Series Analysis")
    print(f"Resolution: {len(df_agg)} minute-level samples")

    df_local = df_agg.copy()
    df_local.index = pd.date_range(start=df_agg.index[0], periods=len(df_agg), freq='T')  # 'T' for minute frequency

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

def perform_umap(df_agg, selected_class_id_to_name, n_neighbors=15, min_dist=0.1):
    """
    Realiza UMAP en los datos agregados.
    """
    print("\n--- Análisis 5: Uniform Manifold Approximation and Projection (UMAP) ---")
    class_names = list(selected_class_id_to_name.values())
    data = df_agg[class_names].fillna(0).values

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(data)

    umap_results = {
        'embedding': embedding,
        'reducer': reducer
    }
    return umap_results

def perform_tsne(df_agg, selected_class_id_to_name, perplexity=30, n_iter=1000):
    """
    Realiza t-SNE en los datos agregados.
    """
    print("\n--- Análisis 6: t-Distributed Stochastic Neighbor Embedding (t-SNE) ---")
    class_names = list(selected_class_id_to_name.values())
    data = df_agg[class_names].fillna(0).values

    tsne = TSNE(perplexity=perplexity, n_iter=n_iter, random_state=42)
    embedding = tsne.fit_transform(data)

    tsne_results = {
        'embedding': embedding,
        'model': tsne
    }
    return tsne_results

def perform_granger_causality_tests(df_agg, selected_class_id_to_name, aggregation_level, maxlag=5):
    from statsmodels.tsa.stattools import grangercausalitytests
    granger_results = {}
    print("\n3. Granger Causality Tests")
    print("Testing for causal relationships between sound events...")
    
    # Create a results structure to store significant relationships
    significant_relationships = []
    
    for i, cls1 in enumerate(selected_class_id_to_name.values()):
        for j, cls2 in enumerate(selected_class_id_to_name.values()):
            if i != j:
                data = df_agg[[cls1, cls2]].fillna(0)
                # Solo testar si hay suficiente variación
                if data[cls1].std() > 0 and data[cls2].std() > 0:
                    try:
                        test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                        
                        # Check for significance at any lag
                        min_pvalue = float('inf')
                        best_lag = None
                        for lag in range(1, maxlag + 1):
                            # Get p-value from F-test
                            pvalue = test_result[lag][0]['ssr_chi2test'][1]
                            if pvalue < min_pvalue:
                                min_pvalue = pvalue
                                best_lag = lag
                        
                        # If significant at 0.05 level
                        if min_pvalue < 0.05:
                            significant_relationships.append({
                                'cause': cls1,
                                'effect': cls2,
                                'p_value': min_pvalue,
                                'best_lag': best_lag
                            })
                            
                    except:
                        continue
    
    # Sort by p-value
    significant_relationships.sort(key=lambda x: x['p_value'])
    
    # Print and save results
    print("\nSignificant Granger Causality Relationships:")
    print(f"{'Cause':<25} {'Effect':<25} {'P-value':<10} {'Best Lag':<10}")
    print("-" * 70)
    
    for rel in significant_relationships:
        print(f"{rel['cause']:<25} {rel['effect']:<25} {rel['p_value']:.4f}    {rel['best_lag']}")
    
    # Save detailed results
    results_dir = os.path.join(BASE_DIR, 'results', aggregation_level)
    os.makedirs(results_dir, exist_ok=True)
    
    granger_output = {
        'parameters': {
            'max_lag': maxlag,
            'significance_level': 0.05
        },
        'significant_relationships': significant_relationships,
        'summary': {
            'total_tests': len(selected_class_id_to_name) * (len(selected_class_id_to_name) - 1),
            'significant_relationships': len(significant_relationships)
        }
    }
    
    with open(os.path.join(results_dir, f'granger_causality_results_{aggregation_level}.json'), 'w') as f:
        json.dump(granger_output, f, indent=2)
    
    print(f"\nTotal number of significant causal relationships found: {len(significant_relationships)}")
    print(f"Full results saved to: {os.path.join(results_dir, f'granger_causality_results_{aggregation_level}.json')}")
    
    # Add some interpretation of findings
    if significant_relationships:
        print("\nKey Findings:")
        # Find sounds that are most often causal
        cause_counts = Counter(rel['cause'] for rel in significant_relationships)
        effect_counts = Counter(rel['effect'] for rel in significant_relationships)
        
        print("\nMost influential sounds (appear most as causes):")
        for cause, count in cause_counts.most_common(5):
            print(f"- {cause}: influences {count} other sounds")
            
        print("\nMost dependent sounds (appear most as effects):")
        for effect, count in effect_counts.most_common(5):
            print(f"- {effect}: influenced by {count} other sounds")
    
    return granger_output

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

def plot_time_series(df_agg, selected_class_id_to_name, aggregation_level):
    os.makedirs(os.path.join(BASE_DIR, 'figures', aggregation_level), exist_ok=True)
    class_names = list(selected_class_id_to_name.values())
    plt.figure(figsize=(12, 6))
    for cls in class_names:
        plt.plot(df_agg.index, df_agg[cls], label=cls)
    plt.xlabel('Time')
    plt.ylabel('Aggregated Probability')
    plt.title(f'Event Probabilities Over Time ({aggregation_level})')
    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, 'figures', f'time_series_{aggregation_level}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(correlation_matrix, aggregation_level):
    os.makedirs(os.path.join(BASE_DIR, 'figures', aggregation_level), exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f'Correlation Matrix ({aggregation_level})')
    plt.savefig(os.path.join(BASE_DIR, 'figures', f'correlation_matrix_{aggregation_level}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_cross_correlations(cross_correlations, sample_size, aggregation_level):
    print("\nAnalyzing cross-correlations...")
    
    # Calculate significance threshold
    conf_int = 1.96 / np.sqrt(sample_size) * 1.5  # 50% stricter
    
    # Store significant correlations
    significant_correlations = []
    try:
        for (cls1, cls2), (lags, ccf_values) in cross_correlations.items():
            max_corr = float(np.max(np.abs(ccf_values)))
            max_lag = int(lags[np.argmax(np.abs(ccf_values))])
            
            if max_corr > conf_int:
                significant_correlations.append({
                    'class1': cls1,
                    'class2': cls2,
                    'correlation': max_corr,
                    'lag': max_lag
                })
        
        # Sort by correlation strength and keep top 20 for visualization
        significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        top_correlations = significant_correlations[:20]
        
        # Save results
        os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)
        with open(os.path.join(BASE_DIR, 'results', f'significant_correlations_{aggregation_level}.json'), 'w') as f:
            json.dump({
                'confidence_threshold': conf_int,
                'significant_pairs': significant_correlations
            }, f, indent=2)
        
        # Plot top correlations if any
        if top_correlations:
            plt.figure(figsize=(15, 10))
            classes = sorted(list(set(
                [corr['class1'] for corr in top_correlations] +
                [corr['class2'] for corr in top_correlations]
            )))
            
            matrix = np.zeros((len(classes), len(classes)))
            for corr in top_correlations:
                i = classes.index(corr['class1'])
                j = classes.index(corr['class2'])
                matrix[i, j] = corr['correlation']
                matrix[j, i] = corr['correlation']
            
            sns.heatmap(matrix,
                        xticklabels=classes,
                        yticklabels=classes,
                        annot=True,
                        cmap='coolwarm',
                        center=0,
                        vmin=-1,
                        vmax=1)
            
            plt.title(f'Top 20 Strongest Cross-Correlations ({aggregation_level})')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_DIR, 'figures', 'top_correlations.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Print summary
        print(f"\nFound {len(significant_correlations)} significant correlations")
        print("\nTop 10 strongest correlations:")
        for i, corr in enumerate(significant_correlations[:10]):
            print(f"{i+1}. {corr['class1']} - {corr['class2']}: "
                  f"correlation = {corr['correlation']:.3f} at lag {corr['lag']}")
        
        return {
            'significant_correlations': significant_correlations,
            'confidence_threshold': conf_int
        }
    except Exception as e:
        print(f"An error occurred in plot_cross_correlations: {e}")
        # Ensure that the function returns the expected keys even in case of error
        return {
            'significant_correlations': significant_correlations,
            'confidence_threshold': conf_int
        }


def plot_pca_results(pca_results, selected_class_id_to_name, aggregation_level):
    os.makedirs(os.path.join(BASE_DIR, 'figures', aggregation_level), exist_ok=True)
    principal_components = pca_results['principal_components']
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
    plt.title(f'PCA of Selected Classes ({aggregation_level})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.savefig(os.path.join(BASE_DIR, 'figures', 'pca_results.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_umap_results(umap_results, df_agg, selected_class_id_to_name, aggregation_level):
    os.makedirs(os.path.join(BASE_DIR, 'figures', aggregation_level), exist_ok=True)
    embedding = umap_results['embedding']
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, c='blue', s=10)
    plt.title(f'UMAP of Selected Classes ({aggregation_level})')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True)
    plt.savefig(os.path.join(BASE_DIR, 'figures', f'umap_results_{aggregation_level}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_tsne_results(tsne_results, df_agg, selected_class_id_to_name, aggregation_level):
    os.makedirs(os.path.join(BASE_DIR, 'figures', aggregation_level), exist_ok=True)
    embedding = tsne_results['embedding']
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, c='green', s=10)
    plt.title(f't-SNE of Selected Classes ({aggregation_level})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True)
    plt.savefig(os.path.join(BASE_DIR, 'figures', f'tsne_results_{aggregation_level}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_umap_animation(df_agg, selected_class_id_to_name, aggregation_level, window_minutes=15):
    """
    Creates an animated visualization of UMAP embeddings over time and saves intermediate data.

    Args:
        df_agg: DataFrame with time series data.
        selected_class_id_to_name: Dictionary mapping class IDs to names.
        window_minutes: Size of sliding window in minutes (default: 15).
    """
    import matplotlib.animation as animation
    import numpy as np
    import pickle  # For saving intermediate data

    # Initialize lists to store intermediate data
    embeddings_list = []
    time_bins = []
    window_indices = []
    additional_info = []

    # Convert index to datetime if needed and sort
    df_agg.index = pd.to_datetime(df_agg.index)
    df_agg = df_agg.sort_index()
    df_agg.index.freq = pd.infer_freq(df_agg.index)

    print("\nTime range in data:")
    print(f"Start: {df_agg.index.min()}")
    print(f"End: {df_agg.index.max()}")
    print(f"Total records: {len(df_agg)}")
    print(f"First 5 indices:\n{df_agg.index[:5]}")

    # Configure UMAP
    reducer = umap.UMAP(n_neighbors=3, min_dist=0.1, random_state=42, n_components=2)

    # Create windows based on index
    min_time = df_agg.index.min()
    max_time = df_agg.index.max()
    window_td = pd.Timedelta(minutes=window_minutes)
    step_td = pd.Timedelta(minutes=15)  # Step size for sliding window (Updated to 15 minutes)

    # Debug: Print time ranges
    print("\nTime window configuration:")
    print(f"Window size: {window_td}")
    print(f"Step size: {step_td}")
    print(f"Data spans: {max_time - min_time}")

    windows = []
    current_time = min_time

    while current_time < max_time:
        window_end = current_time + window_td
        # Get data for current window
        mask = (df_agg.index >= current_time) & (df_agg.index < window_end)
        window_data = df_agg.loc[mask]

        if len(window_data) >= 3:  # Ensure enough samples for UMAP
            windows.append((current_time, window_data))

        current_time += step_td

    print(f"\nTotal number of time windows: {len(windows)}")
    if windows:
        print("\nExample windows:")
        for i, (t, data) in enumerate(windows[:3]):
            print(f"Window {i+1}: Start={t}, End={t+window_td}, Samples={len(data)}")

    if not windows:
        print("\nNo valid windows found. Check data distribution.")
        return None

    # Get feature names
    features = list(selected_class_id_to_name.values())
    print(f"\nFeatures being used: {features}")

    # Generate UMAP embeddings and collect intermediate data
    print("\nGenerating UMAP embeddings for each window...")

    for time_bin, window_data in tqdm(windows):
        try:
            # Prepare data and generate embedding
            data = window_data[features].fillna(0)
            embedding = reducer.fit_transform(data.values)

            # Store embeddings and additional info
            embeddings_list.append(embedding)
            time_bins.append(time_bin)
            window_indices.append(window_data.index)

            # Calculate additional info, e.g., dominant class per point
            dominant_classes = data.idxmax(axis=1)
            additional_info.append(dominant_classes)

        except Exception as e:
            print(f"Error processing window at {time_bin}: {e}")
            continue

    if not embeddings_list:
        print("\nNo valid embeddings generated.")
        return None

    # Calculate global axis limits for consistent scaling
    all_embeddings = np.vstack(embeddings_list)
    global_min_x = np.min(all_embeddings[:, 0])
    global_max_x = np.max(all_embeddings[:, 0])
    global_min_y = np.min(all_embeddings[:, 1])
    global_max_y = np.max(all_embeddings[:, 1])

    # Add margins to the axis limits
    margin_x = (global_max_x - global_min_x) * 0.1
    margin_y = (global_max_y - global_min_y) * 0.1

    global_xmin = global_min_x - margin_x
    global_xmax = global_max_x + margin_x
    global_ymin = global_min_y - margin_y
    global_ymax = global_max_y + margin_y

    # Save intermediate data to a pickle file, including global axis limits
    intermediate_data = {
        'embeddings_list': embeddings_list,
        'time_bins': time_bins,
        'window_indices': window_indices,
        'additional_info': additional_info,
        'features': features,
        'global_axis_limits': {
            'xmin': global_xmin,
            'xmax': global_xmax,
            'ymin': global_ymin,
            'ymax': global_ymax
        }
    }

    intermediate_file = os.path.join(BASE_DIR, 'results', aggregation_level, f'umap_intermediate_data_{aggregation_level}.pkl')
    os.makedirs(os.path.dirname(intermediate_file), exist_ok=True)
    with open(intermediate_file, 'wb') as f:
        pickle.dump(intermediate_data, f)

    print(f"\nIntermediate data saved to: {intermediate_file}")

    # Optionally, generate frames and video immediately
    # generate_video_from_intermediate_data(intermediate_file)
    # Comment out the above line if you don't want to generate the video now

    return None  # Return None as we are focusing on saving intermediate data


def generate_video_from_intermediate_data(intermediate_file, aggregation_level, title="UMAP Animation", legend=True):
    """
    Generates a video from saved UMAP embeddings and allows customization.

    Args:
        intermediate_file: Path to the saved intermediate data file.
        title: Title for the animation frames.
        legend: Whether to include a legend in the plots.
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import pickle

    # Load intermediate data
    with open(intermediate_file, 'rb') as f:
        data = pickle.load(f)

    embeddings_list = data['embeddings_list']
    time_bins = data['time_bins']
    additional_info = data['additional_info']
    features = data['features']
    axis_limits = data.get('global_axis_limits', None)

    # Create frames directory
    frames_dir = os.path.join(BASE_DIR, 'figures', 'umap_frames')
    os.makedirs(frames_dir, exist_ok=True)
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))

    print("\nGenerating frames from intermediate data...")

    # Map unique classes to colors
    unique_classes = set()
    for info in additional_info:
        unique_classes.update(info.unique())
    class_to_color = {cls: plt.cm.tab20(i % 20) for i, cls in enumerate(sorted(unique_classes))}

    frames = []
    for idx, embedding in enumerate(embeddings_list):
        time_bin = time_bins[idx]
        dominant_classes = additional_info[idx]

        # Map classes to colors for the current frame
        colors = dominant_classes.map(class_to_color)

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.7, s=10)

        # Customize the title
        plt.title(f"{title} - {time_bin.strftime('%Y-%m-%d %H:%M')}")

        # Add legend if desired
        if legend:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', label=cls,
                                      markerfacecolor=col, markersize=8)
                               for cls, col in class_to_color.items()]
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True)

        # Set consistent axis limits if available
        if axis_limits:
            plt.xlim(axis_limits['xmin'], axis_limits['xmax'])
            plt.ylim(axis_limits['ymin'], axis_limits['ymax'])

        frame_filename = os.path.join(frames_dir, f'frame_{idx:04d}.png')
        plt.savefig(frame_filename, dpi=300, bbox_inches='tight')
        plt.close()
        frames.append(frame_filename)

    print(f"\nTotal frames generated: {len(frames)}")

    if len(frames) > 1:
        print("\nCreating animated visualization...")
        fig = plt.figure(figsize=(8, 6))
        plt.axis('off')
        ims = []

        for frame_file in sorted(frames):
            img = plt.imread(frame_file)
            im = plt.imshow(img, animated=True)
            ims.append([im])

        # Create animation
        ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)

        # Save the MP4 video
        video_filename = os.path.join(BASE_DIR, 'figures', aggregation_level, 'umap_animation_custom.mp4')
        ani.save(video_filename, writer='ffmpeg', fps=2)
        print(f"Video saved to: {video_filename}")

        # Save the GIF file
        gif_filename = os.path.join(BASE_DIR, 'figures', aggregation_level, 'umap_animation_custom.gif')
        ani.save(gif_filename, writer='pillow', fps=2)
        print(f"GIF saved to: {gif_filename}")

        return ani
    else:
        print("\nInsufficient frames generated for animation.")
        return None



if __name__ == '__main__':
    from contextlib import redirect_stdout
    import datetime
    import os

    # Create logs directory
    log_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Create log file with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'analysis_log_{timestamp}.txt')

    # Variable to control whether to perform full analysis or only generate video
    perform_full_analysis = True  # Set to False to only generate the video

    if perform_full_analysis:
        with open(log_file, 'w') as f:
            with redirect_stdout(f):
                main()

        # Print to console that the log was saved
        print(f"\nLog saved to: {log_file}")

        # After full analysis, generate video for each aggregation level
        for aggregation_level in AGGREGATION_LEVELS:
            intermediate_file = os.path.join(BASE_DIR, 'results', aggregation_level, f'umap_intermediate_data_{aggregation_level}.pkl')
            generate_video_from_intermediate_data(intermediate_file, aggregation_level, title=f"UMAP Animation - {aggregation_level}", legend=True)
    else:
        # Only generate the video from intermediate data for each aggregation level
        for aggregation_level in AGGREGATION_LEVELS:
            intermediate_file = os.path.join(BASE_DIR, 'results', aggregation_level, f'umap_intermediate_data_{aggregation_level}.pkl')
            generate_video_from_intermediate_data(intermediate_file, aggregation_level, title=f"UMAP Animation - {aggregation_level}", legend=True)