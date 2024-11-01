# scripts/analyze_recording_times.py

import os
import datetime
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_recording_times(data_dir):
    """
    Analyze recording start times from JSON files in the dataset.
    
    Args:
        data_dir (str): Path to the root directory containing recorder folders
    """
    # Dictionary to store times by recorder
    times_by_recorder = defaultdict(set)
    # Dictionary to store all files with their full timestamps
    file_timestamps = defaultdict(list)
    # Track unique hours and minutes
    unique_hours = set()
    unique_minutes = set()
    
    # Walk through all directories
    for recorder_dir in sorted(os.listdir(data_dir)):
        recorder_path = os.path.join(data_dir, recorder_dir)
        if not os.path.isdir(recorder_path):
            continue
            
        print(f"\nAnalyzing recorder {recorder_dir}...")
        
        # Process all JSON files in the recorder directory
        for filename in sorted(os.listdir(recorder_path)):
            if not filename.endswith('.json'):
                continue
                
            # Remove '_light.json' or '.json' extension
            base_name = filename.replace('_light.json', '').replace('.json', '')
            
            try:
                # Parse datetime from filename (format: YYYYMMDD_HHMMSS)
                timestamp = datetime.datetime.strptime(base_name, '%Y%m%d_%H%M%S')
                
                # Store the time for this recorder
                times_by_recorder[recorder_dir].add(timestamp.strftime('%H:%M'))
                
                # Store full timestamp information
                file_timestamps[recorder_dir].append({
                    'filename': filename,
                    'date': timestamp.strftime('%Y-%m-%d'),
                    'time': timestamp.strftime('%H:%M:%S'),
                    'hour': timestamp.hour,
                    'minute': timestamp.minute
                })
                
                # Track unique hours and minutes
                unique_hours.add(timestamp.hour)
                unique_minutes.add(timestamp.minute)
                
            except ValueError as e:
                print(f"Warning: Could not parse datetime from filename {filename}: {e}")
    
    # Generate analysis results
    results = {
        'summary': {
            'total_files': sum(len(files) for files in file_timestamps.values()),
            'total_recorders': len(times_by_recorder),
            'unique_hours': sorted(list(unique_hours)),
            'unique_minutes': sorted(list(unique_minutes))
        },
        'recorders': {}
    }
    
    # Analyze patterns for each recorder
    for recorder, timestamps in sorted(file_timestamps.items()):
        recorder_summary = {
            'total_files': len(timestamps),
            'unique_start_times': sorted(list(times_by_recorder[recorder])),
            'files_by_hour': defaultdict(int),
            'files_by_minute': defaultdict(int),
            'date_range': {
                'start': min(t['date'] for t in timestamps),
                'end': max(t['date'] for t in timestamps)
            }
        }
        
        # Count files by hour and minute
        for ts in timestamps:
            recorder_summary['files_by_hour'][ts['hour']] += 1
            recorder_summary['files_by_minute'][ts['minute']] += 1
        
        # Convert defaultdicts to regular dicts for JSON serialization
        recorder_summary['files_by_hour'] = dict(sorted(recorder_summary['files_by_hour'].items()))
        recorder_summary['files_by_minute'] = dict(sorted(recorder_summary['files_by_minute'].items()))
        
        results['recorders'][recorder] = recorder_summary
    
    # Save detailed analysis to JSON
    output_dir = 'analysis_results/recording_times'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'recording_times_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate visualizations
    plot_recording_patterns(results, output_dir)
    
    return results

def plot_recording_patterns(results, output_dir):
    """
    Generate visualizations of recording patterns.
    
    Args:
        results (dict): Analysis results dictionary
        output_dir (str): Directory to save plots
    """
    # Plot 1: Heatmap of recording times by recorder
    plt.figure(figsize=(15, 8))
    recorders = sorted(results['recorders'].keys())
    hours = sorted(results['summary']['unique_hours'])
    
    data = []
    for recorder in recorders:
        row = []
        for hour in hours:
            count = results['recorders'][recorder]['files_by_hour'].get(hour, 0)
            row.append(count)
        data.append(row)
    
    plt.imshow(data, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='Number of recordings')
    plt.xlabel('Hour of day')
    plt.ylabel('Recorder')
    plt.title('Recording Times Distribution by Recorder')
    
    plt.xticks(range(len(hours)), [f'{h:02d}:00' for h in hours])
    plt.yticks(range(len(recorders)), recorders)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recording_times_heatmap.png'))
    plt.close()
    
    # Plot 2: Bar chart of unique start minutes
    plt.figure(figsize=(12, 6))
    minutes = sorted(results['summary']['unique_minutes'])
    minute_counts = defaultdict(int)
    
    for recorder_data in results['recorders'].values():
        for minute, count in recorder_data['files_by_minute'].items():
            minute_counts[minute] += count
    
    plt.bar(minutes, [minute_counts[m] for m in minutes])
    plt.xlabel('Minute')
    plt.ylabel('Number of recordings')
    plt.title('Distribution of Recording Start Minutes')
    plt.xticks(minutes, [f'{m:02d}' for m in minutes], rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recording_start_minutes.png'))
    plt.close()

def main():
    """Main function to run the analysis"""
    data_dir = '/vol/research/datasets/audio/AI4S_SoH/VITALISE data light/Cnn14_DecisionLevelAtt_light'
    print(f"Analyzing recording times in {data_dir}")
    
    results = analyze_recording_times(data_dir)
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total files analyzed: {results['summary']['total_files']}")
    print(f"Total recorders: {results['summary']['total_recorders']}")
    print(f"Unique recording hours: {sorted(results['summary']['unique_hours'])}")
    print(f"Unique recording minutes: {sorted(results['summary']['unique_minutes'])}")
    
    print("\nDetailed results have been saved to analysis_results/recording_times/")

if __name__ == '__main__':
    main()