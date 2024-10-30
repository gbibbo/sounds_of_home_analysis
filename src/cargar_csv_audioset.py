# src/cargar_csv_audioset.py

import pandas as pd
import os

def get_class_params(class_id):
    """
    Gets the parameters for a specific AudioSet class.
    
    Args:
        class_id (str): Class identifier (label)
        
    Returns:
        dict: Dictionary with class parameters or None if not found
              Parameters include:
              - quality_estimate: Quality estimation
              - total_videos: Total number of videos
              - evaluation_videos: Number of evaluation videos
              - balanced_train_videos: Number of balanced training videos
              - unbalanced_train_videos: Number of unbalanced training videos
              - evaluation_hours: Evaluation hours
              - balanced_train_hours: Balanced training hours
              - unbalanced_train_hours: Unbalanced training hours
    """
    try:
        # Build path to CSV file
        csv_path = os.path.join('metadata', 'audioset_data.csv')
        
        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"Error: File {csv_path} not found")
            return None
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Search for class by ID (label)
        class_data = df[df['label'] == class_id]
        
        # Check if class was found
        if class_data.empty:
            print(f"Class '{class_id}' not found")
            return None
        
        # Convert row to dictionary and return parameters
        params = class_data.iloc[0].to_dict()
        
        return params
        
    except Exception as e:
        print(f"Error getting class parameters: {e}")
        return None

if __name__ == "__main__":
    # Example with existing class
    class_id = "Music"
    params = get_class_params(class_id)
    
    if params:
        print(f"\nParameters for class '{class_id}':")
        for key, value in params.items():
            print(f"{key}: {value}")
            
    # Example with non-existent class
    class_id = "NonExistentClass"
    params = get_class_params(class_id)
    if not params:
        print(f"\nNo parameters found for class '{class_id}'")