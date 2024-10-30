# src/events_statistics.py

"""
The Sounds of Home Event Statistics Generator

This script processes and analyzes sound event predictions based on the AudioSet ontology.
It generates statistical information about the occurrence of different sound categories
and their hierarchical relationships in the analyzed audio recordings.

Key Features:
- Processes predictions from multiple JSON files
- Handles hierarchical relationships using AudioSet ontology
- Applies quality-based confidence thresholds
- Generates statistics at category, subcategory, and hierarchical levels
- Creates visualization plots

Input:
- JSON files containing sound event predictions
- AudioSet ontology file
- Class labels mapping file
- Quality estimates for AudioSet classes

Output:
- JSON file with detailed statistics
- Visualization plots for categories and subcategories
- Console output with comprehensive statistics

Required directory structure:
├── metadata/
│   ├── ontology.json
│   ├── class_labels_indices.csv
│   └── audioset_data.csv
├── assets/
│   └── images/
└── src/
    └── events_statistics.py

Configuration:
- Adjust PREDICTIONS_ROOT_DIR for input data location
- Set confidence thresholds and visualization preferences using global variables
- Customize category organization through CUSTOM_CATEGORIES dictionary
"""

import os
import json
import csv
import pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# Root directory containing the recorder folders
PREDICTIONS_ROOT_DIR = '/vol/research/datasets/audio/AI4S_SoH/VITALISE data light/Cnn14_DecisionLevelAtt_light'

# Paths to the ontology and class labels files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY_PATH = os.path.join(BASE_DIR, '../metadata/ontology.json')
CLASS_LABELS_CSV_PATH = os.path.join(BASE_DIR, '../metadata/class_labels_indices.csv')
LABEL_QUALITY_ESTIMATES_PATH = os.path.join(BASE_DIR, '../metadata/audioset_data.csv')

# Output paths
OUTPUT_JSON_PATH = 'events_statistics_results.json'
IMAGES_OUTPUT_DIR = os.path.join(BASE_DIR, '../assets/images')

# Ensure images directory exists
os.makedirs(IMAGES_OUTPUT_DIR, exist_ok=True)

# Configuration thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.2  # Default threshold value
USE_LABEL_QUALITY_THRESHOLDS = True  # Set to True to adjust thresholds based on label quality
GENERATE_GRAPHS = True  # Set to False if you don't want to generate graphs

# Custom categories provided by the user
CUSTOM_CATEGORIES = {
    'Human sounds': [
        'Human voice',
        'Whistling',
        'Respiratory sounds',
        'Human locomotion',
        'Digestive',
        'Hands',
        'Heart sounds, heartbeat',
        'Otoacoustic emission',
        'Human group actions'
    ],
    'Source-ambiguous sounds': [
        'Generic impact sounds',
        'Surface contact',
        'Deformable shell',
        'Onomatopoeia',
        'Silence',
        'Other sourceless'
    ],
    'Animal': [
        'Animal',
        'Domestic animals, pets',
        'Livestock, farm animals, working animals',
        'Wild animals'
    ],
    'Sounds of things': [
        'Vehicle',
        'Engine',
        'Domestic sounds, home sounds',
        'Bell',
        'Alarm',
        'Mechanisms',
        'Tools',
        'Explosion',
        'Wood',
        'Glass',
        'Liquid',
        'Miscellaneous sources',
        'Specific impact sounds'
    ],
    'Music': [
        'Music',
        'Musical instrument',
        'Music genre',
        'Musical concepts',
        'Music role',
        'Music mood'
    ],
    'Natural sounds': [
        'Wind',
        'Thunderstorm',
        'Water',
        'Fire'
    ],
    'Channel, environment and background': [
        'Acoustic environment',
        'Noise',
        'Sound reproduction'
    ]
}

def load_quality_estimates(csv_path):
    """
    Loads quality estimates from the audioset_data.csv file.
    Returns a dictionary mapping class labels to their quality estimates.
    """
    quality_estimates = {}
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            quality_estimates[row['label']] = row['quality_estimate']
    except Exception as e:
        print(f"Error loading quality estimates: {e}")
        return {}
    return quality_estimates

# Label quality estimates
LABEL_QUALITY_ESTIMATES = load_quality_estimates(LABEL_QUALITY_ESTIMATES_PATH)

def load_ontology(ontology_path):
    """Loads the ontology from a JSON file."""
    with open(ontology_path, 'r') as f:
        ontology = json.load(f)
    return ontology

def load_class_labels(csv_path):
    """Loads class labels and their corresponding IDs from a CSV file."""
    class_label_to_id = {}
    class_id_to_label = {}
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            display_name = row['display_name'].strip('"').strip()
            class_id = row['mid']
            class_label_to_id[display_name.lower()] = class_id
            class_id_to_label[class_id] = display_name
    return class_label_to_id, class_id_to_label

def build_mappings(ontology):
    """
    Builds necessary mappings from the ontology.
    Returns:
        id_to_class: Mapping from class ID to class details.
        name_to_class_id: Mapping from class name to class ID.
    """
    id_to_class = {item['id']: item for item in ontology}
    name_to_class_id = {item['name']: item['id'] for item in ontology}
    return id_to_class, name_to_class_id

def build_parent_child_mappings(ontology):
    """
    Builds parent-to-children and child-to-parents mappings.
    Returns:
        parent_to_children: Mapping from parent ID to list of child IDs.
        child_to_parents: Mapping from child ID to list of parent IDs.
    """
    parent_to_children = defaultdict(list)
    child_to_parents = defaultdict(list)
    for class_item in ontology:
        parent_id = class_item['id']
        for child_id in class_item.get('child_ids', []):
            parent_to_children[parent_id].append(child_id)
            child_to_parents[child_id].append(parent_id)
    return parent_to_children, child_to_parents

def get_class_id(class_name, class_label_to_id, name_to_class_id):
    """
    Attempts to find the class ID using multiple approaches.
    Args:
        class_name: Name of the class to look for
        class_label_to_id: Dictionary mapping lowercase display names to IDs
        name_to_class_id: Dictionary mapping exact ontology names to IDs
    Returns:
        class_id or None if not found
    """
    # Attempt to obtain the ID directly
    class_id = class_label_to_id.get(class_name.lower())
    if class_id:
        return class_id

    # If not found, search in name_to_class_id
    class_id = name_to_class_id.get(class_name)
    if class_id:
        return class_id

    # If still not found, search in a more flexible way
    for name, id in name_to_class_id.items():
        if class_name.lower() in name.lower():
            return id

    print(f"Warning: Could not find ID for class '{class_name}'")
    return None

def get_leaf_nodes(class_id, parent_to_children):
    """
    Recursively find all leaf nodes (nodes without children) under a given class ID.
    Args:
        class_id: ID of the class to find leaves for
        parent_to_children: Dictionary mapping parent IDs to their children IDs
    Returns:
        List of leaf node IDs
    """
    children = parent_to_children.get(class_id, [])
    if not children:
        return [class_id]
    leaves = []
    for child in children:
        leaves.extend(get_leaf_nodes(child, parent_to_children))
    return leaves

def get_ancestors(class_id, child_to_parents, memo=None):
    """
    Recursively obtains all ancestors of a given class ID.
    Uses memoization to optimize repeated searches.
    """
    if memo is None:
        memo = {}
    if class_id in memo:
        return memo[class_id]

    ancestors = set()
    parents = child_to_parents.get(class_id, [])
    for parent_id in parents:
        ancestors.add(parent_id)
        ancestors.update(get_ancestors(parent_id, child_to_parents, memo))
    memo[class_id] = ancestors
    return ancestors

def map_classes_to_categories(ontology, custom_categories, id_to_class, name_to_class_id, parent_to_children, child_to_parents):
    """
    Maps each class in the ontology to its corresponding category and subcategory.
    Returns:
        class_id_to_category_info: Mapping from class ID to its category and subcategory.
    """
    class_id_to_category_info = {}

    # Map main categories and subcategories to their IDs
    category_name_to_id = {}
    subcategory_id_to_category = {}
    for category_name, subcategory_names in custom_categories.items():
        category_id = name_to_class_id.get(category_name)
        if category_id:
            category_name_to_id[category_name] = category_id
            # Map subcategory IDs to the category name
            for sub_name in subcategory_names:
                sub_id = name_to_class_id.get(sub_name)
                if sub_id:
                    subcategory_id_to_category[sub_id] = (category_name, sub_name)
                else:
                    print(f"Warning: Subcategory '{sub_name}' not found in ontology.")
        else:
            print(f"Warning: Category '{category_name}' not found in ontology.")

    # Determine category and subcategory for each class
    for class_item in ontology:
        class_id = class_item['id']
        class_name = class_item['name']

        # Initialize category and subcategory
        category = None
        subcategory = None

        # Get all ancestors of the class
        ancestors = get_ancestors(class_id, child_to_parents)

        # Check if the class itself is a subcategory
        if class_id in subcategory_id_to_category:
            category, subcategory = subcategory_id_to_category[class_id]
        else:
            # Find matching subcategories among ancestors
            matching_subcategories = ancestors & set(subcategory_id_to_category.keys())
            if matching_subcategories:
                # Choose the most specific subcategory (deepest in the hierarchy)
                closest_sub_id = max(matching_subcategories, key=lambda x: len(get_ancestors(x, child_to_parents)))
                category, subcategory = subcategory_id_to_category[closest_sub_id]
            else:
                # Check if the class itself is a main category
                if class_id in category_name_to_id.values():
                    category = class_name
                    subcategory = class_name
                else:
                    # Find matching main categories among ancestors
                    matching_categories = ancestors & set(category_name_to_id.values())
                    if matching_categories:
                        category_id = next(iter(matching_categories))
                        category = id_to_class[category_id]['name']
                        subcategory = class_name  # Use the class itself as subcategory
                    else:
                        # If it doesn't match any category, mark as 'Uncategorized'
                        category = 'Uncategorized'
                        subcategory = class_name

        # Assign to mapping
        class_id_to_category_info[class_id] = {
            'category': category,
            'subcategory': subcategory
        }

    return class_id_to_category_info

def process_predictions(predictions_root_dir):
    """
    Traverses JSON prediction files and collects all predictions.
    Returns:
        all_predictions: List of prediction dictionaries with hierarchical information.
    """
    all_predictions = []
    json_file_count = 0  # Initialize counter for .json files processed

    for subdir_name in os.listdir(predictions_root_dir):
        subdir_path = os.path.join(predictions_root_dir, subdir_name)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.json') or filename.endswith('_light.json'):
                    json_file_count += 1  # Increment counter
                    if json_file_count % 100 == 0:
                        print(f"{json_file_count} JSON files processed so far...")
                    filepath = os.path.join(subdir_path, filename)
                    with open(filepath, 'r') as f:
                        try:
                            data = json.load(f)
                            for frame in data:
                                if isinstance(frame, dict) and ('predictions' in frame):
                                    for prediction in frame['predictions']:
                                        # Ensure consistent format for class_label and probability
                                        if 'class' in prediction:
                                            prediction['class_label'] = prediction['class']
                                        if 'prob' in prediction:
                                            prediction['probability'] = prediction['prob']
                                        all_predictions.append(prediction)
                        except json.JSONDecodeError:
                            print(f"Error: Failed to parse JSON file {filepath}.")

    # After processing, print the total number of JSON files processed
    print(f"Total JSON files processed: {json_file_count}")
    return all_predictions

def compute_class_thresholds(class_label_to_id, class_id_to_label):
    """
    Computes class-specific thresholds based on label quality estimates.
    For USE_LABEL_QUALITY_THRESHOLDS=True:
        - Threshold is calculated linearly between 0.2 (at 0% quality) and DEFAULT_CONFIDENCE_THRESHOLD (at 100% quality)
        - Formula: threshold = 0.2 + (DEFAULT_CONFIDENCE_THRESHOLD - 0.2) * (quality / 100)
    For USE_LABEL_QUALITY_THRESHOLDS=False:
        - Uses DEFAULT_CONFIDENCE_THRESHOLD for all classes
    Returns:
        class_thresholds: Dictionary mapping class labels to thresholds.
    """
    class_thresholds = {}

    if USE_LABEL_QUALITY_THRESHOLDS:
        # Calculate thresholds based on label quality estimates
        for class_label_lower, class_id in class_label_to_id.items():
            class_label = class_id_to_label[class_id]
            label_quality = LABEL_QUALITY_ESTIMATES.get(class_label, 50)  # Default to 50% if not found

            # Linear interpolation between 0.2 (at 0% quality) and DEFAULT_CONFIDENCE_THRESHOLD (at 100% quality)
            threshold = 0.2 + (DEFAULT_CONFIDENCE_THRESHOLD - 0.2) * (label_quality / 100)
            class_thresholds[class_label] = threshold
    else:
        # Use default confidence threshold for all classes
        for class_label_lower, class_id in class_label_to_id.items():
            class_label = class_id_to_label[class_id]
            class_thresholds[class_label] = DEFAULT_CONFIDENCE_THRESHOLD

    return class_thresholds

def filter_predictions(all_predictions, class_thresholds):
    """
    Filters predictions based on the selected threshold method.
    Returns:
        filtered_predictions: List of prediction dictionaries.
    """
    filtered_predictions = []
    for pred in all_predictions:
        class_label = (pred.get('class_label') or pred.get('class', '')).strip()
        probability = pred.get('probability') or pred.get('prob', 0)
        threshold = class_thresholds.get(class_label, DEFAULT_CONFIDENCE_THRESHOLD)

        if probability >= threshold:
            filtered_predictions.append(pred)
    return filtered_predictions

def map_predictions_to_categories(predictions, class_label_to_id, class_id_to_category_info):
    """
    Maps each prediction to its corresponding category and subcategory.
    Returns:
        predicted_categories: List of dictionaries with keys 'category' and 'subcategory'.
        unmapped_classes: Set of class labels that could not be mapped.
    """
    predicted_categories = []
    unmapped_classes = set()

    for pred in predictions:
        class_label = pred.get('class_label') or pred.get('class', '').strip()
        class_label_lower = class_label.lower()
        class_id = class_label_to_id.get(class_label_lower)
        if class_id:
            category_info = class_id_to_category_info.get(class_id)
            if category_info:
                predicted_categories.append(category_info)
                pred['mapped_category'] = category_info['category']
                pred['mapped_subcategory'] = category_info['subcategory'] or category_info['category']
            else:
                # Assign to 'Uncategorized' if category not found
                predicted_categories.append({'category': 'Uncategorized', 'subcategory': class_label})
                pred['mapped_category'] = 'Uncategorized'
                pred['mapped_subcategory'] = class_label
                unmapped_classes.add(class_label)
        else:
            # Assign to 'Uncategorized' if class label not found
            predicted_categories.append({'category': 'Uncategorized', 'subcategory': class_label})
            pred['mapped_category'] = 'Uncategorized'
            pred['mapped_subcategory'] = class_label
            unmapped_classes.add(class_label)

    return predicted_categories, unmapped_classes

def generate_statistics(predicted_categories):
    """
    Generates counts for categories and subcategories.
    Returns:
        category_counts: Counter object for categories.
        subcategory_counts: defaultdict of Counter objects for subcategories within each category.
    """
    category_counts = Counter()
    subcategory_counts = defaultdict(Counter)

    for entry in predicted_categories:
        category = entry['category']
        subcategory = entry['subcategory'] if entry['subcategory'] else entry['category']
        category_counts[category] += 1
        subcategory_counts[category][subcategory] += 1

    return category_counts, subcategory_counts

def aggregate_counts(data_counts, parent_to_children, child_to_parents, id_to_class, class_label_to_id):
    """
    Aggregate counts considering the complete hierarchical structure.
    Args:
        data_counts: Dictionary of hour-based counts for each class ID
        parent_to_children: Dictionary mapping parent IDs to their children IDs
        child_to_parents: Dictionary mapping child IDs to their parent IDs
        id_to_class: Dictionary mapping class IDs to their full information
        class_label_to_id: Dictionary mapping class labels to their IDs
    Returns:
        aggregated_counts: Dictionary mapping class names to their total counts
    """
    aggregated_counts = {}

    def aggregate_recursive(class_id):
        class_name = id_to_class[class_id]['name']

        # Get all leaf nodes under this class
        leaf_nodes = get_leaf_nodes(class_id, parent_to_children)

        # Sum counts from all leaf nodes
        total_count = 0
        for leaf_id in leaf_nodes:
            for hour in data_counts:
                if leaf_id in data_counts[hour]:
                    total_count += data_counts[hour][leaf_id]

        aggregated_counts[class_name] = total_count
        return total_count

    # Start aggregation from root nodes
    for class_id in id_to_class:
        if class_id not in child_to_parents:  # This is a root class
            aggregate_recursive(class_id)

    return aggregated_counts

# Plotting functions
def plot_main_categories(category_counts):
    """
    Generates a bar plot for main categories showing percentages.
    """
    total_count = sum(category_counts.values())
    sorted_categories = category_counts.most_common()
    categories = [cat for cat, _ in sorted_categories]
    counts = [count for _, count in sorted_categories]
    percentages = [count / total_count * 100 for count in counts]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, percentages, color='skyblue', edgecolor='black')
    plt.xlabel('Categories')
    plt.ylabel('Percentage of Samples (%)')
    plt.title('Sound Event Statistics by Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    for bar, percent in zip(bars, percentages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(percentages)*0.01,
                f'{percent:.1f}%', ha='center', va='bottom')

    # Save the plot
    plt.savefig(os.path.join(IMAGES_OUTPUT_DIR, 'main_categories.png'))
    plt.close()

def plot_subcategories(subcategory_counts):
    """
    Generates bar plots for each main category showing subcategory percentages.
    """
    for category, sub_counts in subcategory_counts.items():
        if category == 'Uncategorized':
            continue

        total_sub_count = sum(sub_counts.values())
        sorted_subcategories = sub_counts.most_common()
        subcategories = [sub for sub, _ in sorted_subcategories]
        counts = [count for _, count in sorted_subcategories]
        percentages = [count / total_sub_count * 100 for count in counts]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(subcategories, percentages, color='lightgreen', edgecolor='black')
        plt.xlabel('Subcategories')
        plt.ylabel('Percentage of Samples (%)')
        plt.title(f"Subcategory Breakdown in '{category}'")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        for bar, percent in zip(bars, percentages):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(percentages)*0.01,
                    f'{percent:.1f}%', ha='center', va='bottom')

        # Save the plot with category name in filename
        safe_category_name = category.lower().replace(' ', '_').replace("'", "")
        plt.savefig(os.path.join(IMAGES_OUTPUT_DIR, f'subcategories_{safe_category_name}.png'))
        plt.close()

def main():
    # Step 1: Load ontology and class labels
    ontology = load_ontology(ONTOLOGY_PATH)
    class_label_to_id, class_id_to_label = load_class_labels(CLASS_LABELS_CSV_PATH)

    # Step 2: Build necessary mappings
    id_to_class, name_to_class_id = build_mappings(ontology)
    parent_to_children, child_to_parents = build_parent_child_mappings(ontology)

    # Step 3: Map classes to categories and subcategories
    class_id_to_category_info = map_classes_to_categories(
        ontology, CUSTOM_CATEGORIES, id_to_class, name_to_class_id,
        parent_to_children, child_to_parents
    )

    # Step 4: Process predictions
    all_predictions = process_predictions(PREDICTIONS_ROOT_DIR)

    # Step 5: Compute class-specific thresholds
    class_thresholds = compute_class_thresholds(class_label_to_id, class_id_to_label)

    # Step 6: Filter predictions using the selected threshold method
    filtered_predictions = filter_predictions(all_predictions, class_thresholds)

    # Step 7: Organize predictions by hour and class
    data_counts = {}
    for pred in filtered_predictions:
        hour = pred.get('hour', '00')  # Default to '00' if hour not present
        if hour not in data_counts:
            data_counts[hour] = {}

        class_label = pred.get('class_label') or pred.get('class', '')
        class_id = get_class_id(class_label, class_label_to_id, name_to_class_id)

        if class_id:
            if class_id not in data_counts[hour]:
                data_counts[hour][class_id] = 0
            data_counts[hour][class_id] += 1

    # Step 8: Aggregate counts considering the complete hierarchy
    hierarchical_counts = aggregate_counts(data_counts, parent_to_children, child_to_parents,
                                         id_to_class, class_label_to_id)

    # Step 9: Map predictions to categories for category statistics
    predicted_categories, unmapped_classes = map_predictions_to_categories(
        filtered_predictions, class_label_to_id, class_id_to_category_info
    )

    # Step 10: Generate category statistics
    category_counts, subcategory_counts = generate_statistics(predicted_categories)

    # Step 11: Print statistics
    print("\nHierarchical Statistics:")
    for class_name, count in sorted(hierarchical_counts.items(), key=lambda x: -x[1]):
        print(f"{class_name}: {count}")

    print("\nCategory Statistics:")
    for category, count in category_counts.items():
        print(f"{category}: {count}")

    print("\nSubcategory Statistics:")
    for category, sub_counts in subcategory_counts.items():
        print(f"\nSubcategories in '{category}':")
        for subcategory, count in sub_counts.items():
            print(f"  {subcategory}: {count}")

    # Step 12: Save analysis results to JSON
    analysis_results = {
        'hierarchical_counts': hierarchical_counts,
        'category_counts': dict(category_counts),
        'subcategory_counts': {cat: dict(sub_counts) for cat, sub_counts in subcategory_counts.items()}
    }

    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(analysis_results, f, indent=4)
    print(f"\nAnalysis results saved to {OUTPUT_JSON_PATH}")

    # Step 13: Print unmapped classes if any
    if unmapped_classes:
        print("\nUnmapped Classes:")
        for cls in sorted(unmapped_classes):
            print(f"- '{cls}'")

    # Step 14: Print the threshold method used
    if USE_LABEL_QUALITY_THRESHOLDS:
        print("\nUsing label quality-based thresholds:")
        for class_label, threshold in class_thresholds.items():
            print(f"- '{class_label}': Threshold = {threshold}")
    else:
        print(f"\nUsing fixed confidence threshold: {DEFAULT_CONFIDENCE_THRESHOLD}")

    # Step 15: Generate graphs if enabled
    if GENERATE_GRAPHS:
        plot_main_categories(category_counts)
        plot_subcategories(subcategory_counts)
        print(f"\nGraphs saved to '{IMAGES_OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
