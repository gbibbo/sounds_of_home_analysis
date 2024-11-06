# scripts/batch_analysis.py

import sys
import os
import json
import src.config as config

from src.data_processing.utils import get_available_days, get_available_hours
from src.data_processing.load_data import load_ontology, build_mappings
from src.data_processing.process_data import load_and_process_data

def main():
    # Define thresholds to analyze, including 'variable' for variable threshold
    thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 'variable']

    # Get all available recorders
    available_recorders = [rec for rec in os.listdir(config.PREDICTIONS_ROOT_DIR)
                           if os.path.isdir(os.path.join(config.PREDICTIONS_ROOT_DIR, rec))]
    config.SELECTED_RECORDERS = available_recorders

    # Obtain all available days
    available_days_dict = get_available_days(config.PREDICTIONS_ROOT_DIR, available_recorders)
    available_days_set = set()
    for days_list in available_days_dict.values():
        available_days_set.update(days_list)
    config.SELECTED_DAYS = sorted(available_days_set)

    # Obtain all available hours
    available_hours_dict = get_available_hours(config.PREDICTIONS_ROOT_DIR, available_recorders)
    available_hours_set = set()
    for hours_list in available_hours_dict.values():
        available_hours_set.update(hours_list)
    config.SELECTED_HOURS = sorted(available_hours_set)

    # Use classes and subclasses from CUSTOM_CATEGORIES in config.py
    config.SELECTED_CLASSES = []
    for category, subclasses in config.CUSTOM_CATEGORIES.items():
        config.SELECTED_CLASSES.append(category)
        config.SELECTED_CLASSES.extend(subclasses)

    # Create directories to save the results
    normal_results_dir = os.path.join('analysis_results', 'batch_analysis_results')
    muted_results_dir = os.path.join('analysis_results', 'batch_analysis_results_MUTED')
    os.makedirs(normal_results_dir, exist_ok=True)
    os.makedirs(muted_results_dir, exist_ok=True)

    for threshold in thresholds:
        if threshold == 'variable':
            # Configure to use variable thresholds
            config.USE_LABEL_QUALITY_THRESHOLDS = True
            config.CONFIDENCE_THRESHOLD = config.DEFAULT_CONFIDENCE_THRESHOLD
            config.CONFIDENCE_THRESHOLD_STR = 'variable'
        else:
            config.USE_LABEL_QUALITY_THRESHOLDS = False
            # Set the confidence threshold
            config.CONFIDENCE_THRESHOLD = threshold
            config.CONFIDENCE_THRESHOLD_STR = str(threshold)

        # Load and process the data
        result = load_and_process_data()
        if result:
            # Unpack normal and muted data counts
            data_counts_normal, data_counts_muted, class_label_to_id, id_to_class, parent_to_children, name_to_class_id = result

            # Include the threshold value in the filename
            if threshold == 'variable':
                filename = 'analysis_results_threshold_variable.json'
            else:
                filename = f'analysis_results_threshold_{threshold}.json'

            # Save the total results
            total_normal_dir = os.path.join(normal_results_dir, 'total')
            total_muted_dir = os.path.join(muted_results_dir, 'total')
            os.makedirs(total_normal_dir, exist_ok=True)
            os.makedirs(total_muted_dir, exist_ok=True)

            # Create data_counts_total_normal and data_counts_total_muted
            data_counts_total_normal = {}
            data_counts_total_muted = {}

            for recorder in data_counts_normal:
                data_counts_total_normal[recorder] = {}
                data_counts_total_muted[recorder] = {}

                # Aggregate counts over all dates for normal data
                for date_str in data_counts_normal[recorder]:
                    for hour in data_counts_normal[recorder][date_str]:
                        if hour not in data_counts_total_normal[recorder]:
                            data_counts_total_normal[recorder][hour] = {}
                        for class_id, count in data_counts_normal[recorder][date_str][hour].items():
                            if class_id not in data_counts_total_normal[recorder][hour]:
                                data_counts_total_normal[recorder][hour][class_id] = 0
                            data_counts_total_normal[recorder][hour][class_id] += count

                # Aggregate counts over all dates for muted data
                for date_str in data_counts_muted[recorder]:
                    for hour in data_counts_muted[recorder][date_str]:
                        if hour not in data_counts_total_muted[recorder]:
                            data_counts_total_muted[recorder][hour] = {}
                        for class_id, count in data_counts_muted[recorder][date_str][hour].items():
                            if class_id not in data_counts_total_muted[recorder][hour]:
                                data_counts_total_muted[recorder][hour][class_id] = 0
                            data_counts_total_muted[recorder][hour][class_id] += count

            output_file_normal = os.path.join(total_normal_dir, filename)
            output_file_muted = os.path.join(total_muted_dir, filename)

            # Save the normal total results
            with open(output_file_normal, 'w') as f:
                json.dump({
                    'threshold': config.CONFIDENCE_THRESHOLD_STR,
                    'data_counts': data_counts_total_normal
                }, f)

            # Save the muted total results
            with open(output_file_muted, 'w') as f:
                json.dump({
                    'threshold': config.CONFIDENCE_THRESHOLD_STR,
                    'data_counts': data_counts_total_muted
                }, f)

            # Save per-day results
            days_set = set()
            for recorder in data_counts_normal:
                days_set.update(data_counts_normal[recorder].keys())

            for day in days_set:
                # Create directories for the day
                day_normal_dir = os.path.join(normal_results_dir, day)
                day_muted_dir = os.path.join(muted_results_dir, day)
                os.makedirs(day_normal_dir, exist_ok=True)
                os.makedirs(day_muted_dir, exist_ok=True)

                # Extract data_counts for the day
                data_counts_normal_day = {}
                data_counts_muted_day = {}

                for recorder in data_counts_normal:
                    if day in data_counts_normal[recorder]:
                        if recorder not in data_counts_normal_day:
                            data_counts_normal_day[recorder] = {}
                        data_counts_normal_day[recorder][day] = data_counts_normal[recorder][day]

                    if day in data_counts_muted[recorder]:
                        if recorder not in data_counts_muted_day:
                            data_counts_muted_day[recorder] = {}
                        data_counts_muted_day[recorder][day] = data_counts_muted[recorder][day]

                # Save per-day normal results
                output_file_normal_day = os.path.join(day_normal_dir, filename)
                with open(output_file_normal_day, 'w') as f:
                    json.dump({
                        'threshold': config.CONFIDENCE_THRESHOLD_STR,
                        'data_counts': data_counts_normal_day
                    }, f)

                # Save per-day muted results
                output_file_muted_day = os.path.join(day_muted_dir, filename)
                with open(output_file_muted_day, 'w') as f:
                    json.dump({
                        'threshold': config.CONFIDENCE_THRESHOLD_STR,
                        'data_counts': data_counts_muted_day
                    }, f)

            print(f"Analysis for threshold {config.CONFIDENCE_THRESHOLD_STR} completed.")
            print(f"Total normal results saved in '{output_file_normal}'.")
            print(f"Total muted results saved in '{output_file_muted}'.")
            print(f"Per-day results saved in respective directories.")

        else:
            print(f"No data processed for threshold {config.CONFIDENCE_THRESHOLD_STR}.")

if __name__ == '__main__':
    main()