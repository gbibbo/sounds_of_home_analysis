# src/data_processing/utils.py

import os
import datetime
import src.config as config 

def get_class_id(class_name, class_label_to_id, name_to_class_id):
    # Attempt to obtain the ID directly
    class_id = class_label_to_id.get(class_name.lower())
    if class_id:
        return class_id
    
    # If not found, search on name_to_class_id
    class_id = name_to_class_id.get(class_name)
    if class_id:
        return class_id
    
    # If still not found, search in a more flexible way
    for name, id in name_to_class_id.items():
        if class_name.lower() in name.lower():
            return id
    
    print(f"Warning: Could not find ID for class '{class_name}'")
    return None

def get_all_subclasses(class_id, parent_to_children):
    """Obtiene todos los subclases de una clase, incluyendo la clase misma."""
    subclasses = {class_id}
    for child in parent_to_children.get(class_id, []):
        subclasses.update(get_all_subclasses(child, parent_to_children))
    return subclasses

def get_ancestors(class_id, child_to_parents, memo=None):
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

def extract_datetime_from_filename(filename):
    # Remove file extension
    filename_no_ext, ext = os.path.splitext(filename)
    # Remove '_light' termination if present
    if filename_no_ext.endswith('_light'):
        filename_no_ext = filename_no_ext[:-6]
    # Extract datetime
    try:
        file_datetime = datetime.datetime.strptime(filename_no_ext, '%Y%m%d_%H%M%S')
    except ValueError as ve:
        print(f"Error parsing date from file {filename}: {ve}")
        return None
    return file_datetime


def get_available_days(predictions_root_dir, recorders):
    available_days = {}
    for recorder in recorders:
        recorder_dir = os.path.join(predictions_root_dir, recorder)
        days_set = set()
        for filename in os.listdir(recorder_dir):
            if filename.endswith('.json'):
                file_datetime = extract_datetime_from_filename(filename)
                if file_datetime:
                    date_str = file_datetime.strftime('%Y%m%d')
                    days_set.add(date_str)
        available_days[recorder] = sorted(days_set)
    return available_days

def get_available_hours(predictions_root_dir, recorders):
    available_hours = {}
    for recorder in recorders:
        recorder_dir = os.path.join(predictions_root_dir, recorder)
        hours_set = set()
        for filename in os.listdir(recorder_dir):
            if filename.endswith('.json'):
                file_datetime = extract_datetime_from_filename(filename)
                if file_datetime:
                    hour_str = file_datetime.strftime('%H')
                    hours_set.add(hour_str)
        available_hours[recorder] = sorted(hours_set)
    return available_hours

def get_num_recorders(dataset_dir):
    """
    Counts how many recorder directories in dataset_dir have at least one JSON file.
    A recorder directory is considered active if it contains at least one .json file.
    """
    import os
    recorder_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    active_count = 0
    for rec in recorder_folders:
        rec_path = os.path.join(dataset_dir, rec)
        # Check if there's any JSON file inside
        if any(fn.endswith('.json') for fn in os.listdir(rec_path) if os.path.isfile(os.path.join(rec_path, fn))):
            active_count += 1
    return active_count