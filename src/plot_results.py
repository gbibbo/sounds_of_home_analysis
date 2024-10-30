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
        'Sound reproduction',
        'Human sounds',
        'Sounds of things',
        'Channel, environment and background'
    ]

    threshold = 0.3
    results_dir = 'analysis_results'
    input_file = os.path.join(results_dir, f'analysis_results_threshold_{threshold}.json')

    if not os.path.exists(input_file):
        print(f"The analysis results file '{input_file}' does not exist.")
        return

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
        for class_name in classes_to_plot:
            class_id = get_class_id(class_name, {}, name_to_class_id)
            if class_id:
                all_related_ids = get_all_subclasses(class_id, parent_to_children)
                class_count = sum(data_counts[hour].get(cid, 0) for cid in all_related_ids)
                counts_per_hour[hour][class_name] = class_count
            else:
                counts_per_hour[hour][class_name] = 0

    # Prepare data for plotting
    hours = all_hours
    class_counts = []
    for class_name in classes_to_plot:
        counts = [counts_per_hour[hour].get(class_name, 0) for hour in hours]
        class_counts.append(counts)

    # Create the area plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot stacked areas
    ax.fill_between(hours, 0, class_counts[0], label=classes_to_plot[0], alpha=0.7)
    for i in range(1, len(class_counts)):
        bottom = np.sum(class_counts[:i], axis=0)
        ax.fill_between(hours, bottom, bottom + class_counts[i], 
                       label=classes_to_plot[i], alpha=0.7)

    ax.set_xlabel('Hour')
    ax.set_ylabel('Event Counts')
    ax.set_title(f'Sound Events per Hour (Threshold: {threshold})')
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels(hours, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
    plt.savefig('output_plot.png')

if __name__ == '__main__':
    main()