# src/batch_analysis.py

import sys
import os
import json

# Add the root directory of the project to the PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.config as config

# Import necessary functions
from src.gui.tkinter_interface import get_available_days, get_available_hours
from src.data_processing.load_data import load_ontology, build_mappings
from src.data_processing.process_data import load_and_process_data

def main():
    # Define the thresholds to analyze
    thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    # Get all available recorders
    available_recorders = [rec for rec in os.listdir(config.PREDICTIONS_ROOT_DIR)
                           if os.path.isdir(os.path.join(config.PREDICTIONS_ROOT_DIR, rec))]
    config.SELECTED_RECORDERS = available_recorders

    # Get all available days and hours
    available_days_dict = get_available_days(config.PREDICTIONS_ROOT_DIR, available_recorders)
    config.SELECTED_DAYS = list(available_days_dict.values())

    available_hours_dict = get_available_hours(config.PREDICTIONS_ROOT_DIR, available_recorders)
    config.SELECTED_HOURS = list(available_hours_dict.values())

    # Use classes and subclasses from CUSTOM_CATEGORIES in config.py
    config.SELECTED_CLASSES = []
    for category, subclasses in config.CUSTOM_CATEGORIES.items():
        config.SELECTED_CLASSES.append(category)
        config.SELECTED_CLASSES.extend(subclasses)

    # Create a directory to save the results
    results_dir = 'analysis_results'
    os.makedirs(results_dir, exist_ok=True)

    for threshold in thresholds:
        # Set the confidence threshold
        config.CONFIDENCE_THRESHOLD = threshold
        config.CONFIDENCE_THRESHOLD_STR = str(threshold)

        # Load and process the data
        result = load_and_process_data()
        if result:
            data_counts, class_label_to_id, id_to_class, parent_to_children, name_to_class_id = result

            # Include the threshold value in the filename
            output_file = os.path.join(results_dir, f'analysis_results_threshold_{threshold}.json')

            # Save the results along with the threshold value
            with open(output_file, 'w') as f:
                json.dump({
                    'threshold': threshold,
                    'data_counts': data_counts
                }, f)

            print(f"Analysis for threshold {threshold} completed. Results saved in '{output_file}'.")
        else:
            print(f"No data processed for threshold {threshold}.")

if __name__ == '__main__':
    main()
