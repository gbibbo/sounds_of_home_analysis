# scripts/analyze_recording_times.py

import os
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

def print_recording_times(times_by_recorder):
    """
    Print recording times for each recorder in a formatted way.
    
    Args:
        times_by_recorder (dict): Dictionary mapping recorder IDs to sets of recording times
    """
    print("\nRecording start times by recorder:")
    print("-" * 50)
    
    for recorder in sorted(times_by_recorder.keys()):
        # Convert times to datetime objects for proper sorting
        times = sorted(list(times_by_recorder[recorder]))
        formatted_recorder = f"{int(recorder):02d}"
        times_str = ", ".join(times)
        print(f"Recorder {formatted_recorder}: {times_str}")

def save_recorders_active_per_hour(times_by_recorder):
    """
    Save the recorders active per hour to a JSON file.

    Args:
        times_by_recorder (dict): Dictionary mapping recorder IDs to sets of recording times
    """
    recorders_active_per_hour = defaultdict(set)

    for recorder, times in times_by_recorder.items():
        for time in times:
            hour = time[:2]  # Extract the hour part (HH)
            recorders_active_per_hour[hour].add(recorder)

    # Convert sets to lists for JSON serialization
    recorders_active_per_hour = {hour: list(recorders) for hour, recorders in recorders_active_per_hour.items()}

    # Ensure the directory exists
    os.makedirs('analysis_results/recording_times', exist_ok=True)

    # Save to JSON file
    with open('analysis_results/recording_times/recorders_active_per_hour.json', 'w') as f:
        json.dump(recorders_active_per_hour, f, indent=4)

def analyze_recording_times(data_dir):
    """
    Analyze and visualize recording times for each recorder.
    
    Args:
        data_dir (str): Path to the root directory containing recorder folders
    """
    # Dictionary to store times by recorder
    times_by_recorder = defaultdict(set)
    
    # Walk through all directories
    for recorder_dir in sorted(os.listdir(data_dir)):
        recorder_path = os.path.join(data_dir, recorder_dir)
        if not os.path.isdir(recorder_path):
            continue
        
        # Process all JSON files in the recorder directory
        for filename in sorted(os.listdir(recorder_path)):
            if not filename.endswith('.json'):
                continue
                
            # Remove '_light.json' or '.json' extension
            base_name = filename.replace('_light.json', '').replace('.json', '')
            
            try:
                # Parse datetime from filename (format: YYYYMMDD_HHMMSS)
                timestamp = datetime.datetime.strptime(base_name, '%Y%m%d_%H%M%S')
                times_by_recorder[recorder_dir].add(timestamp.strftime('%H:%M'))
            except ValueError as e:
                print(f"Warning: Could not parse datetime from filename {filename}: {e}")

    # Print recording times
    print_recording_times(times_by_recorder)

    # Save recorders active per hour to JSON
    save_recorders_active_per_hour(times_by_recorder)
    
    # Create the visualization
    create_heatmap(times_by_recorder)

def create_heatmap(times_by_recorder):
    """
    Create a heatmap visualization of recording times.
    
    Args:
        times_by_recorder (dict): Dictionary mapping recorder IDs to sets of recording times
    """
    # Create figure and axis
    plt.figure(figsize=(15, 8))
    
    # Prepare data for heatmap
    recorders = sorted(times_by_recorder.keys())
    
    # Create time range manually to ensure proper formatting
    hours = range(6, 24)  # 6:00 to 23:00
    time_range = [f"{hour:02d}:00" for hour in hours]
    
    # Create binary matrix (1 for recording, 0 for no recording)
    data = np.zeros((len(recorders), len(time_range)))
    
    for i, recorder in enumerate(recorders):
        for j, time in enumerate(time_range):
            hour = int(time.split(':')[0])
            next_hour = (hour + 1) % 24
            next_time = f"{next_hour:02d}:00"
            
            # Check if there are any recordings in this hour
            recordings_in_hour = any(
                t >= time and (t < next_time or next_hour < hour)
                for t in times_by_recorder[recorder]
            )
            data[i, j] = 1 if recordings_in_hour else 0

    colors = ['#FFFACD', '#800000']  # Amarillo claro a burgundy
    custom_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('custom', colors)
    # Create heatmap with custom colors
    plt.imshow(data, aspect='auto', cmap=custom_cmap, interpolation='nearest')
    
    # Customize appearance
    plt.title('Recording Times Distribution by Recorder')
    plt.xlabel('Hour of Day')
    plt.ylabel('Recorder')
    
    # Set x-axis ticks
    plt.xticks(range(len(time_range)), time_range, rotation=0)
    
    # Set y-axis ticks
    plt.yticks(range(len(recorders)), [f"{int(r):02d}" for r in recorders])
    
    # Ensure the directory exists
    os.makedirs('analysis_results/recording_times', exist_ok=True)
    
    # Save the plot
    plt.savefig('analysis_results/recording_times/recording_times_heatmap.png', 
                bbox_inches='tight', 
                dpi=300,
                facecolor='white')
    plt.close()

def main():
    """Main function to run the analysis"""
    data_dir = '/vol/research/datasets/audio/AI4S_SoH/VITALISE data light/Cnn14_DecisionLevelAtt_light'
    analyze_recording_times(data_dir)

if __name__ == '__main__':
    main()