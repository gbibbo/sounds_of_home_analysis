# scripts/plot_results.py

import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Import configurations and utility functions
import src.config as config
from src.data_processing.load_data import (
    load_ontology,
    build_mappings,
    build_parent_child_mappings,
    load_class_labels
)
from src.data_processing.utils import get_all_subclasses, get_class_id

def plot_results(data_counts, selected_recorders, selected_classes, threshold_str, recorder_info, normalize_option):
    """
    Generates a stacked area plot for the selected classes and recorders.

    Args:
        data_counts (dict): Nested dictionary with counts per recorder and hour.
        selected_recorders (list): List of selected recorder IDs.
        selected_classes (list): List of selected class names.
        threshold_str (str): Threshold value used, as a string.
        recorder_info (str): Information about the recorders used.
    """
    # Load ontology and mappings
    ontology = load_ontology(config.ONTOLOGY_PATH)
    class_label_to_id, class_id_to_label = load_class_labels(config.CLASS_LABELS_CSV_PATH)
    id_to_class, name_to_class_id = build_mappings(ontology)
    parent_to_children, child_to_parents = build_parent_child_mappings(ontology)

    # Leer el archivo JSON con los grabadores activos por hora
    recorders_active_per_hour_path = 'analysis_results/recording_times/recorders_active_per_hour.json'
    with open(recorders_active_per_hour_path, 'r') as f:
        recorders_active_per_hour = json.load(f)

    # Prepare data for plotting
    counts_per_hour = {}

    # Aggregate counts over selected recorders
    for hour in data_counts:
        counts_per_hour[hour] = {}
        for class_id, count in data_counts[hour].items():
            counts_per_hour[hour][class_id] = count

    # Prepare data for plotting
    all_hours = sorted(counts_per_hour.keys(), key=lambda x: int(x))
    counts_per_class_per_hour = {hour: {} for hour in all_hours}

    # Total number of selected recorders
    total_recorders = len(selected_recorders)
    num_recorders_active_per_hour = {}
    for hour in all_hours:
        active_recorders = recorders_active_per_hour.get(hour, [])
        # Filtrar solo los grabadores seleccionados
        num_active = len(set(active_recorders) & set(selected_recorders))
        num_recorders_active_per_hour[hour] = num_active

    # For each hour, compute the counts for each selected class
    for hour in all_hours:
        num_recorders_active = num_recorders_active_per_hour.get(hour, 0)
        if normalize_option:
            if num_recorders_active == 0:
                weight = 1
            else:
                weight = total_recorders / num_recorders_active
        else:
            weight = 1

        for class_name in selected_classes:
            class_id = get_class_id(class_name, class_label_to_id, name_to_class_id)
            if class_id:
                # Get all subclasses (including the class itself)
                all_related_ids = get_all_subclasses(class_id, parent_to_children)
                class_count = sum(counts_per_hour[hour].get(cid, 0) for cid in all_related_ids)
                adjusted_class_count = class_count * weight
                counts_per_class_per_hour[hour][class_name] = adjusted_class_count
            else:
                counts_per_class_per_hour[hour][class_name] = 0

    # Prepare data arrays for plotting
    hours = sorted(all_hours, key=lambda x: int(x))
    x_values = range(len(hours))
    class_counts_list = []
    for class_name in selected_classes:
        counts = [counts_per_class_per_hour[hour].get(class_name, 0) for hour in hours]
        class_counts_list.append(counts)
    class_counts = np.array(class_counts_list)

    # Create the stacked area plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot stacked areas
    bottom = np.zeros(len(hours))
    for i, class_name in enumerate(selected_classes):
        counts = class_counts[i]
        ax.fill_between(x_values, bottom, bottom + counts, label=class_name, alpha=0.7)
        bottom += counts

    ax.set_xlabel('Hour')
    ax.set_ylabel('Event Counts')
    ax.set_title(f'Sound Events per Hour ({recorder_info}, Threshold: {threshold_str})')

    if not normalize_option:
        # Añadir fracción de grabadores activos en las etiquetas del eje x
        xtick_labels = []
        for hour in hours:
            num_recorders_active = num_recorders_active_per_hour.get(hour, 0)
            fraction = f"{num_recorders_active}/{total_recorders}"
            label = f"{hour}\n({fraction})"
            xtick_labels.append(label)
        ax.set_xticks(x_values)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    else:
        # Etiquetas originales del eje x
        ax.set_xticks(x_values)
        ax.set_xticklabels(hours, rotation=45, ha='right')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs('assets/images', exist_ok=True)
    plt.savefig('assets/images/plot.png')
    plt.show()