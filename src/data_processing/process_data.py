# src/data_processing/process_data.py

import os
import json
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import src.config as config
from src.data_processing.utils import get_ancestors
from src.data_processing.load_data import (
    load_ontology,
    load_class_labels,
    build_mappings,
    build_parent_child_mappings,
    map_classes_to_categories
)
from tqdm import tqdm
import pandas as pd

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

def compute_class_thresholds(class_label_to_id, class_id_to_label):
    """
    Computes class-specific thresholds based on label quality estimates.
    """
    class_thresholds = {}

    # Load label quality estimates
    LABEL_QUALITY_ESTIMATES = load_quality_estimates(config.LABEL_QUALITY_ESTIMATES_PATH)

    # Calculate thresholds based on label quality estimates
    for class_label_lower, class_id in class_label_to_id.items():
        class_label = class_id_to_label[class_id]
        label_quality = LABEL_QUALITY_ESTIMATES.get(class_label, 50)  # Default to 50% if not found

        # Linear interpolation between 0.2 and the default confidence threshold
        threshold = 0.2 + (config.DEFAULT_CONFIDENCE_THRESHOLD - 0.2) * (label_quality / 100)
        class_thresholds[class_label] = threshold

    return class_thresholds

def process_json_file(file_path, class_label_to_id, class_id_to_label, class_id_to_category_info,
                      parent_to_children, child_to_parents, selected_hours, confidence_threshold, class_thresholds, recorder):
    file_counts_normal = {}
    file_counts_muted = {}
    processed_hours = set()
    classes_with_data = set()
    is_light = '_light.json' in file_path

    try:
        filename = os.path.basename(file_path).replace('_light.json', '').replace('.json', '')
        file_datetime = datetime.datetime.strptime(filename, '%Y%m%d_%H%M%S')

        with open(file_path, 'r') as f:
            data = json.load(f)

            # Collect times to mute
            mute_times = []

            # First pass: Identify mute intervals
            for frame in data:
                if isinstance(frame, dict) and ('predictions' in frame):
                    frame_time = file_datetime + datetime.timedelta(seconds=float(frame.get('time', 0)))
                    hour = frame_time.strftime('%H')

                    if hour in selected_hours:
                        predictions = frame.get('predictions', [])

                        for pred in predictions:
                            class_label = pred.get('class_label') or pred.get('class')
                            score = pred.get('probability') or pred.get('prob')

                            if class_label and class_label in config.MUTE_LABELS:
                                if score >= config.MUTE_THRESHOLD:
                                    mute_times.append(frame_time.timestamp())

            # Create mute intervals (with 1 second before and after)
            mute_intervals = []
            for t in mute_times:
                start_time = t - 1.0
                end_time = t + 1.0
                mute_intervals.append((start_time, end_time))

            # Merge overlapping intervals
            mute_intervals.sort()
            merged_intervals = []
            for interval in mute_intervals:
                if not merged_intervals or interval[0] > merged_intervals[-1][1]:
                    merged_intervals.append(list(interval))
                else:
                    merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])

            # Second pass: Process frames and apply muting
            for frame in data:
                if isinstance(frame, dict) and ('predictions' in frame):
                    frame_time = file_datetime + datetime.timedelta(seconds=float(frame.get('time', 0)))
                    frame_timestamp = frame_time.timestamp()
                    hour = frame_time.strftime('%H')

                    if hour in selected_hours:
                        processed_hours.add(hour)
                        predictions = frame.get('predictions', [])

                        # Check if frame is muted
                        is_muted = any(start <= frame_timestamp <= end for start, end in merged_intervals)

                        # Initialize counts if not already
                        if hour not in file_counts_normal:
                            file_counts_normal[hour] = {}
                        if hour not in file_counts_muted:
                            file_counts_muted[hour] = {}

                        for pred in predictions:
                            class_label = pred.get('class_label') or pred.get('class')
                            score = pred.get('probability') or pred.get('prob')

                            if class_label:
                                class_id = class_label_to_id.get(class_label.lower())
                                if class_id:
                                    # Get the threshold for this class
                                    if class_thresholds:
                                        threshold = class_thresholds.get(class_label, confidence_threshold)
                                    else:
                                        threshold = confidence_threshold

                                    if score >= threshold:
                                        # Get ancestors
                                        ancestors = get_ancestors(class_id, child_to_parents)
                                        # Add the current class and its ancestors
                                        all_related_ids = ancestors | {class_id}

                                        increment = 32 if is_light else 1
                                        classes_with_data.update(all_related_ids)

                                        # Update normal counts
                                        for related_id in all_related_ids:
                                            if related_id not in file_counts_normal[hour]:
                                                file_counts_normal[hour][related_id] = 0
                                            file_counts_normal[hour][related_id] += increment

                                        # Update muted counts if not muted
                                        if not is_muted:
                                            for related_id in all_related_ids:
                                                if related_id not in file_counts_muted[hour]:
                                                    file_counts_muted[hour][related_id] = 0
                                                file_counts_muted[hour][related_id] += increment

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        raise

    return recorder, file_counts_normal, file_counts_muted, processed_hours, classes_with_data

def load_and_process_data():
    print(f"\nSelected confidence threshold: {config.CONFIDENCE_THRESHOLD_STR}")

    ontology = load_ontology(config.ONTOLOGY_PATH)
    class_label_to_id, class_id_to_label = load_class_labels(config.CLASS_LABELS_CSV_PATH)
    id_to_class, name_to_class_id = build_mappings(ontology)
    parent_to_children, child_to_parents = build_parent_child_mappings(ontology)
    class_id_to_category_info = map_classes_to_categories(
        ontology, config.CUSTOM_CATEGORIES, id_to_class, name_to_class_id,
        parent_to_children, child_to_parents
    )

    # Compute class thresholds if using variable thresholds
    if config.USE_LABEL_QUALITY_THRESHOLDS:
        class_thresholds = compute_class_thresholds(class_label_to_id, class_id_to_label)
    else:
        class_thresholds = None  # Not using variable thresholds

    # Prepare list of files to process and keep track of recorders
    files_to_process = []
    available_days = set()
    available_hours = set()
    processed_recorders = set()
    for recorder in config.SELECTED_RECORDERS:
        recorder_dir = os.path.join(config.PREDICTIONS_ROOT_DIR, recorder)
        if not os.path.isdir(recorder_dir):
            print(f"\nWarning: The recorder directory '{recorder}' does not exist.")
            continue
        else:
            processed_recorders.add(recorder)

        for filename in os.listdir(recorder_dir):
            if filename.endswith('.json'):
                # Extract day and hour information
                try:
                    filename_no_ext = filename.replace('_light.json', '').replace('.json', '')
                    file_datetime = datetime.datetime.strptime(filename_no_ext, '%Y%m%d_%H%M%S')
                    date_str = file_datetime.strftime('%Y%m%d')
                    hour_str = file_datetime.strftime('%Y-%m-%d %H:00')
                except ValueError as ve:
                    print(f"Error parsing date from file {filename}: {ve}")
                    continue

                available_days.add(date_str)
                available_hours.add(hour_str)

                # Add to list of files to process if date matches or if no specific date is selected
                if date_str in config.SELECTED_DAYS or not config.SELECTED_DAYS:
                    file_path = os.path.join(recorder_dir, filename)
                    files_to_process.append((file_path, class_label_to_id, class_id_to_label,
                                             class_id_to_category_info, parent_to_children,
                                             child_to_parents, config.SELECTED_HOURS,
                                             config.CONFIDENCE_THRESHOLD, class_thresholds,
                                             recorder))

    # Ensure days and hours are selected by default if available
    if not config.SELECTED_DAYS and available_days:
        config.SELECTED_DAYS = sorted(list(available_days))
    if not config.SELECTED_HOURS and available_hours:
        config.SELECTED_HOURS = sorted(list(available_hours))

    if not processed_recorders:
        print("\nNo valid recorder directories found. Exiting.")
        return None

    if not available_days:
        print("\nWarning: No data found for the selected days.")
        print("Please check the dates and ensure data is available.")
        return None

    if not files_to_process:
        print("\nNo files to process. Exiting.")
        return None

    # Process files using multiprocessing with progress bar
    data_counts_normal = {}
    data_counts_muted = {}
    processed_hours = set()
    classes_with_data = set()

    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = [
            executor.submit(process_json_file, *args)
            for args in files_to_process
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            recorder, file_counts_normal_rec, file_counts_muted_rec, file_hours, file_classes = future.result()
            # Merge normal results
            if recorder not in data_counts_normal:
                data_counts_normal[recorder] = {}
            for hour, counts in file_counts_normal_rec.items():
                if hour not in data_counts_normal[recorder]:
                    data_counts_normal[recorder][hour] = {}
                for class_id, count in counts.items():
                    if class_id not in data_counts_normal[recorder][hour]:
                        data_counts_normal[recorder][hour][class_id] = 0
                    data_counts_normal[recorder][hour][class_id] += count
            # Merge muted results
            if recorder not in data_counts_muted:
                data_counts_muted[recorder] = {}
            for hour, counts in file_counts_muted_rec.items():
                if hour not in data_counts_muted[recorder]:
                    data_counts_muted[recorder][hour] = {}
                for class_id, count in counts.items():
                    if class_id not in data_counts_muted[recorder][hour]:
                        data_counts_muted[recorder][hour][class_id] = 0
                    data_counts_muted[recorder][hour][class_id] += count

            processed_hours.update(file_hours)
            classes_with_data.update(file_classes)

    if not processed_hours:
        print("\nWarning: No hours were processed. No data available for the selected parameters.")
        return None

    if not classes_with_data:
        print("\nWarning: No data found for the selected classes in the processed data.")
        print("The classes may not be present in the data for the selected recorders and days.")

    # If there are hours selected that weren't processed, provide a warning
    unprocessed_hours = set(config.SELECTED_HOURS) - processed_hours
    if unprocessed_hours:
        print(f"\nWarning: The following hours were not found in the data and were not processed: {', '.join(sorted(unprocessed_hours))}")

    # Print a summary
    print("\nProcessing Summary:")
    print(f"Recorders processed: {', '.join(sorted(processed_recorders))}")
    print(f"Days processed: {', '.join(sorted(config.SELECTED_DAYS))}")
    print(f"Hours processed: {', '.join(sorted(processed_hours))}")

    return data_counts_normal, data_counts_muted, class_label_to_id, id_to_class, parent_to_children, name_to_class_id
