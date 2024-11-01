# scripts/plot_results.py

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Normalization Explanation:
# The normalization adjusts the event counts to account for the varying number of active recorders in each hour.
# For each hour, we calculate a normalization factor defined as:
#     norm_factor = total_number_of_selected_recorders / number_of_active_selected_recorders_in_that_hour
# This factor scales the event counts so that the results represent what would have been detected if all selected recorders were active.
# The adjusted event count for each class and hour is then calculated as:
#     adjusted_event_count = original_event_count * norm_factor
# This method ensures that hours with fewer active recorders are not underrepresented in the analysis,
# allowing for a fair comparison of event counts across hours with different recorder availability.

# Import configurations and utility functions
import src.config as config
from src.data_processing.load_data import (
    load_ontology,
    build_mappings,
    build_parent_child_mappings,
    load_class_labels
)
from src.data_processing.utils import get_all_subclasses, get_class_id

def plot_results(data_counts, selected_recorders, selected_classes, threshold_str, recorder_info, normalize=False):
    """
    Generates a stacked area plot for the selected classes and recorders.

    Args:
        data_counts (dict): Nested dictionary with counts per recorder and hour.
        selected_recorders (list): List of selected recorder IDs.
        selected_classes (list): List of selected class names.
        threshold_str (str): Threshold value used, as a string.
        recorder_info (str): Information about the recorders used.
        normalize (bool): Whether to normalize counts by number of active recorders.
    """
    # Load ontology and mappings
    ontology = load_ontology(config.ONTOLOGY_PATH)
    class_label_to_id, class_id_to_label = load_class_labels(config.CLASS_LABELS_CSV_PATH)
    id_to_class, name_to_class_id = build_mappings(ontology)
    parent_to_children, child_to_parents = build_parent_child_mappings(ontology)

    # Load active recorders information
    try:
        with open('analysis_results/recording_times/recorders_active_per_hour.json', 'r') as f:
            recorders_active = json.load(f)
    except FileNotFoundError:
        print("Warning: recorders_active_per_hour.json not found.")
        recorders_active = {}

    # Prepare data for plotting
    counts_per_hour = {}

    # Aggregate counts over selected recorders
    for recorder in selected_recorders:
        recorder_data = data_counts.get(recorder, {})
        for hour in recorder_data:
            if hour not in counts_per_hour:
                counts_per_hour[hour] = {}
            for class_id, count in recorder_data[hour].items():
                if class_id not in counts_per_hour[hour]:
                    counts_per_hour[hour][class_id] = 0
                counts_per_hour[hour][class_id] += count

    # Prepare data for plotting
    all_hours = sorted(counts_per_hour.keys(), key=lambda x: int(x))
    counts_per_class_per_hour = {hour: {} for hour in all_hours}

    # Total number of selected recorders
    total_recorders = len(selected_recorders)
    active_recorders_per_hour = {}
    norm_factors = {}  # Dictionary to store normalization factors per hour

    for hour in all_hours:
        hour_str = str(hour).zfill(2)
        active_recorders_list = recorders_active.get(hour_str, [])
        # Filter to only selected recorders
        active_recorders = [rec for rec in active_recorders_list if rec in selected_recorders]
        num_active_recorders = len(active_recorders)
        active_recorders_per_hour[hour] = num_active_recorders

        # Calculate normalization factor if needed
        norm_factor = 1
        if normalize:
            if num_active_recorders > 0:
                norm_factor = total_recorders / num_active_recorders
            else:
                norm_factor = 1  # Avoid division by zero
        norm_factors[hour] = norm_factor  # Store the normalization factor

        for class_name in selected_classes:
            class_id = get_class_id(class_name, class_label_to_id, name_to_class_id)
            if class_id:
                # Get all subclasses (including the class itself)
                all_related_ids = get_all_subclasses(class_id, parent_to_children)
                class_count = sum(counts_per_hour[hour].get(cid, 0) for cid in all_related_ids)
                # Apply normalization if enabled
                counts_per_class_per_hour[hour][class_name] = class_count * norm_factor
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
    title = 'Normalized ' if normalize else ''
    ax.set_title(f'{title}Sound Events per Hour ({recorder_info}, Threshold: {threshold_str})')

    # Prepare x-axis labels
    xtick_labels = []
    for hour in hours:
        hour_label = str(hour).zfill(2)
        if normalize:
            # Add normalization factor to x-axis labels
            norm_factor = norm_factors.get(hour, 1)
            label = f'{hour_label}\n(x{norm_factor:.2f})'
        else:
            # Add fraction of active recorders to x-axis labels
            num_active = active_recorders_per_hour.get(hour, 0)
            fraction = f'{num_active}/{total_recorders}'
            label = f'{hour_label}\n({fraction})'
        xtick_labels.append(label)

    ax.set_xticks(x_values)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Ensure the output directory exists
    os.makedirs('assets/images', exist_ok=True)
    plt.savefig('assets/images/plot.png')
    plt.show()
