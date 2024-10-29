# src/data_processing/process_data.py

import os
import json
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import src.config as config
from src.data_processing.load_data import (
    load_ontology,
    load_class_labels,
    build_mappings,
    build_parent_child_mappings,
    map_classes_to_categories
)
from tqdm import tqdm  

def process_json_file(file_path, class_label_to_id, class_id_to_label, class_id_to_category_info,
                      parent_to_children, id_to_class, selected_hours, confidence_threshold):
    file_counts = {}
    processed_hours = set()
    classes_with_data = set()
    print('selected_hours = ', selected_hours)
    try:
        # Extract datetime from filename
        filename = os.path.basename(file_path).replace('_light.json', '').replace('.json', '')
        file_datetime = datetime.datetime.strptime(filename.replace('.json', ''), '%Y%m%d_%H%M%S')

        # Process the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
            for frame in data:
                frame_time = file_datetime + datetime.timedelta(seconds=float(frame.get('time', 0)))
                hour = frame_time.strftime('%H')
                if hour in selected_hours:
                    processed_hours.add(hour)
                    predictions = frame.get('predictions', [])
                    for pred in predictions:
                        class_label = pred.get('class_label') or pred.get('class', '').strip()
                        score = pred.get('probability') or pred.get('prob', 0)
                        #print(f"Processing prediction: {class_label} with score {score}")
                        if score >= confidence_threshold:
                            class_id = class_label_to_id.get(class_label.lower())
                            if class_id:
                                if hour not in file_counts:
                                    file_counts[hour] = {}
                                if class_id not in file_counts[hour]:
                                    file_counts[hour][class_id] = 0
                                file_counts[hour][class_id] += 1
                                classes_with_data.add(class_id)

        print(f"Processed file: {file_path}")
        for hour, counts in file_counts.items():
            print(f"  Hour: {hour}")
            for class_id, count in counts.items():
                print(f"    {class_id_to_label.get(class_id, 'Unknown')} : {count}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        raise
    return file_counts, processed_hours, classes_with_data


def aggregate_counts(data_counts, parent_to_children, child_to_parents, id_to_class, class_label_to_id):
    aggregated_counts = {}
    
    def aggregate_recursive(class_id):
        class_name = id_to_class[class_id]['name']
        direct_count = sum(sum(data_counts[hour].get(cid, 0) for cid in [class_id] + parent_to_children.get(class_id, [])) for hour in data_counts)
        
        child_count = sum(aggregate_recursive(child_id) for child_id in parent_to_children.get(class_id, []))
        
        total_count = direct_count + child_count
        aggregated_counts[class_name] = total_count
        
        return total_count
    
    for class_id in id_to_class:
        if class_id not in child_to_parents:  # This is a root class
            aggregate_recursive(class_id)
    
    return aggregated_counts


def load_and_process_data():
    print(f"\nSelected confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    ontology = load_ontology(config.ONTOLOGY_PATH)
    class_label_to_id, class_id_to_label = load_class_labels(config.CLASS_LABELS_CSV_PATH)
    id_to_class, name_to_class_id = build_mappings(ontology)
    parent_to_children, child_to_parents = build_parent_child_mappings(ontology)
    class_id_to_category_info = map_classes_to_categories(
        ontology, config.CUSTOM_CATEGORIES, id_to_class, name_to_class_id,
        parent_to_children, child_to_parents
    )

    # Prepare list of files to process
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
                    # Remove '_light.json' or '.json' from filename
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
                                 id_to_class, config.SELECTED_HOURS, config.CONFIDENCE_THRESHOLD))

    # Ensure days and hours are selected by default if available
    if not config.SELECTED_DAYS and available_days:
        config.SELECTED_DAYS = sorted(list(available_days))
    if not config.SELECTED_HOURS and available_hours:
        config.SELECTED_HOURS = sorted(list(available_hours))

    # Debug prints to verify available days and hours
    print(f"Available days: {config.SELECTED_DAYS}")
    print(f"Available hours: {config.SELECTED_HOURS}")

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
    data_counts = {}
    processed_hours = set()
    classes_with_data = set()

    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = [
            executor.submit(process_json_file, *args)
            for args in files_to_process
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            file_counts, file_hours, file_classes = future.result()
            # Merge results
            for hour, counts in file_counts.items():
                if hour not in data_counts:
                    data_counts[hour] = {}
                for class_id, count in counts.items():
                    if class_id not in data_counts[hour]:
                        data_counts[hour][class_id] = 0
                    data_counts[hour][class_id] += count
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
    #print(f"Classes with data: {', '.join(sorted([class_id_to_label[class_id] for class_id in classes_with_data]))}")

    # Aggregate counts based on the ontology
    aggregated_counts = aggregate_counts(data_counts, parent_to_children, child_to_parents, id_to_class, class_label_to_id)

    def print_hierarchy(class_name, level=0):
        indent = "  " * level
        class_id = class_label_to_id.get(class_name.lower())
        count = aggregated_counts.get(class_name, 0)
        print(f"{indent}{class_name}: {count} occurrences")
        
        if class_id:
            for child_id in parent_to_children.get(class_id, []):
                child_name = id_to_class[child_id]['name']
                print_hierarchy(child_name, level + 1)

    # Printing counts of unique classes detected
    print("\nDetailed hierarchy of selected classes:")
    for selected_class in config.SELECTED_CLASSES:
        print_hierarchy(selected_class)
        # Print all subclasses of Channel, environment and background
        if selected_class == "Channel, environment and background":
            for subclass in config.CUSTOM_CATEGORIES[selected_class]:
                print_hierarchy(subclass, 1)

    return data_counts, class_label_to_id, id_to_class, parent_to_children, name_to_class_id