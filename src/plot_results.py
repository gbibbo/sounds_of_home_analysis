# src/plot_results.py

import os
import json
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
import json

# Add the root directory of the project to the PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configurations and utility functions
import src.config as config
from src.data_processing.load_data import load_ontology, build_mappings, build_parent_child_mappings
from src.data_processing.utils import get_all_subclasses, get_class_id

def main():
    # Specify the classes to plot
    classes_to_plot = [
        'Channel, environment and background',
        'Acoustic environment',
        'Noise',
        'Sound reproduction'
    ]

    # Specify the threshold to use
    threshold = 0.2

    # Path to the results directory and the specific JSON file
    results_dir = 'analysis_results'
    input_file = os.path.join(results_dir, f'analysis_results_threshold_{threshold}.json')

    if not os.path.exists(input_file):
        print(f"The analysis results file '{input_file}' does not exist.")
        return

    # Load the analysis results from the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    data_counts = data['data_counts']

    # Load ontology and mappings
    ontology = load_ontology(config.ONTOLOGY_PATH)
    id_to_class, name_to_class_id = build_mappings(ontology)
    parent_to_children, child_to_parents = build_parent_child_mappings(ontology)

    # Prepare data for plotting
    all_hours = sorted(data_counts.keys())
    counts_per_hour = {hour: {} for hour in all_hours}

    # For each hour, compute the counts for each class
    for hour in all_hours:
        total_events = sum(data_counts[hour].values())
        for class_name in classes_to_plot:
            class_id = get_class_id(class_name, {}, name_to_class_id)
            if class_id:
                all_related_ids = get_all_subclasses(class_id, parent_to_children)
                class_count = sum(data_counts[hour].get(cid, 0) for cid in all_related_ids)
                counts_per_hour[hour][class_name] = class_count
            else:
                counts_per_hour[hour][class_name] = 0

    # Prepare data for plotting
    hours = [hour.split(' ')[1] for hour in all_hours]  # Extract time part
    num_hours = len(hours)
    num_classes = len(classes_to_plot)
    bar_width = 0.8 / num_classes
    index = np.arange(num_hours)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, class_name in enumerate(classes_to_plot):
        class_counts = [counts_per_hour[hour].get(class_name, 0) for hour in all_hours]
        positions = index + i * bar_width
        ax.bar(positions, class_counts, bar_width, label=class_name)

    ax.set_xlabel('Hour')
    ax.set_ylabel('Event Counts')
    ax.set_title(f'Sound Events per Hour (Threshold: {threshold})')
    ax.set_xticks(index + bar_width * (num_classes - 1) / 2)
    ax.set_xticklabels(hours, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
