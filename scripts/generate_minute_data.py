# scripts/generate_minute_data.py

"""
Processes audio event detection results with minute-level resolution.
Aggregates detection counts across all recorders and days, applying variable 
confidence thresholds for each sound class.

The output is a JSON file where each minute contains counts for all individual
AudioSet classes, without ontology aggregation.
"""

import os
import json
import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import src.config as config
from src.data_processing.utils import (
   get_available_hours,
   extract_datetime_from_filename
)
from src.data_processing.load_data import load_class_labels
from src.data_processing.process_data import compute_class_thresholds

def process_json_file(file_path, class_thresholds, class_label_to_id):
   """
   Process a single JSON prediction file, counting events per minute
   
   Args:
       file_path: Path to JSON file with predictions
       class_thresholds: Dictionary mapping class labels to their thresholds
       class_label_to_id: Dictionary mapping class labels to AudioSet IDs
   
   Returns:
       Dictionary with minute-level counts for each class
   """
   minute_counts = defaultdict(lambda: defaultdict(int))
   try:
       filename = os.path.basename(file_path).replace('_light.json', '').replace('.json', '')
       file_datetime = datetime.datetime.strptime(filename, '%Y%m%d_%H%M%S')
       
       with open(file_path, 'r') as f:
           data = json.load(f)

       is_light = '_light.json' in file_path
       increment = 32 if is_light else 1

       for frame in data:
           if isinstance(frame, dict) and ('predictions' in frame):
               frame_time = file_datetime + datetime.timedelta(seconds=float(frame.get('time', 0)))
               minute_key = frame_time.strftime('%H:%M')
               
               predictions = frame.get('predictions', [])
               for pred in predictions:
                   class_label = pred.get('class_label') or pred.get('class')
                   score = pred.get('probability') or pred.get('prob')
                   
                   if class_label:
                       threshold = class_thresholds.get(class_label, config.DEFAULT_CONFIDENCE_THRESHOLD)
                       
                       if score >= threshold:
                           class_id = class_label_to_id.get(class_label.lower())
                           if class_id:
                               minute_counts[minute_key][class_id] += increment

   except Exception as e:
       print(f"Error processing file {file_path}: {e}")
       raise
       
   return minute_counts

def generate_minute_data():
   """
   Main function to generate minute-level data for audio events.
   Processes all available recordings and saves results to JSON.
   """
   print("Loading AudioSet class labels...")
   
   # Load class labels and compute thresholds
   class_label_to_id, class_id_to_label = load_class_labels(config.CLASS_LABELS_CSV_PATH)
   class_thresholds = compute_class_thresholds(class_label_to_id, class_id_to_label)
   
   # Get available recorders
   available_recorders = [rec for rec in os.listdir(config.PREDICTIONS_ROOT_DIR) 
                        if os.path.isdir(os.path.join(config.PREDICTIONS_ROOT_DIR, rec))]
   
   if not available_recorders:
       print("No recorder directories found. Exiting.")
       return
       
   print(f"Found {len(available_recorders)} recorders")
   
   # Get available hours for filtering
   available_hours = get_available_hours(config.PREDICTIONS_ROOT_DIR, available_recorders)
   valid_hours = sorted(set(hour for recorder_hours in available_hours.values() 
                          for hour in recorder_hours))
   
   if not valid_hours:
       print("No valid hours found in the dataset. Exiting.")
       return
   
   print(f"Processing data for hours: {', '.join(valid_hours)}")
   
   # Prepare list of files to process
   files_to_process = []
   for recorder in available_recorders:
       recorder_dir = os.path.join(config.PREDICTIONS_ROOT_DIR, recorder)
       for filename in os.listdir(recorder_dir):
           if filename.endswith('.json'):
               file_datetime = extract_datetime_from_filename(filename)
               if file_datetime:
                   hour = file_datetime.strftime('%H')
                   if hour in valid_hours:
                       files_to_process.append(os.path.join(recorder_dir, filename))
               
   if not files_to_process:
       print("No JSON files found to process. Exiting.")
       return
       
   print(f"Processing {len(files_to_process)} files...")
   
   # Process files using ThreadPoolExecutor
   all_minute_counts = defaultdict(lambda: defaultdict(int))
   
   with ThreadPoolExecutor() as executor:
       future_to_file = {
           executor.submit(
               process_json_file,
               file_path,
               class_thresholds,
               class_label_to_id
           ): file_path 
           for file_path in files_to_process
       }
       
       for future in tqdm(as_completed(future_to_file), total=len(files_to_process)):
           file_path = future_to_file[future]
           try:
               minute_counts = future.result()
               # Aggregate counts across files
               for minute, counts in minute_counts.items():
                   for class_id, count in counts.items():
                       all_minute_counts[minute][class_id] += count
           except Exception as e:
               print(f"Error processing {file_path}: {e}")
               
   # Convert defaultdict to regular dict for JSON serialization
   output_data = {
       minute: dict(counts)
       for minute, counts in all_minute_counts.items()
   }
   
   # Create output directory if it doesn't exist
   output_dir = os.path.join('analysis_results', 'minute_analysis_results')
   os.makedirs(output_dir, exist_ok=True)
   
   # Save results
   output_path = os.path.join(output_dir, "minute_counts.json")
   with open(output_path, 'w') as f:
       json.dump(output_data, f)
       
   print(f"\nAnalysis complete. Results saved to: {output_path}")
   print(f"Total minutes processed: {len(output_data)}")
   print(f"Total classes detected: {len(set(class_id for counts in output_data.values() for class_id in counts))}")

if __name__ == "__main__":
   generate_minute_data()