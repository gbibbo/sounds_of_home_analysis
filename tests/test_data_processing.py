import unittest
import os
import sys
from pathlib import Path

# Obtener la ruta absoluta al directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Adjust the path to import modules from src
sys.path.append(str(PROJECT_ROOT))

from src.data_processing.process_data import load_and_process_data
from src.data_processing.load_data import (
    load_class_labels, 
    load_ontology, 
    build_mappings, 
    build_parent_child_mappings
)
import src.config as config
from src.data_processing.utils import get_ancestors

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        # Save original configurations
        self.original_predictions_root_dir = config.PREDICTIONS_ROOT_DIR
        self.original_custom_categories = config.CUSTOM_CATEGORIES.copy()
        self.original_class_labels_csv_path = config.CLASS_LABELS_CSV_PATH
        self.original_ontology_path = config.ONTOLOGY_PATH

        # Set up configurations for testing
        config.PREDICTIONS_ROOT_DIR = str(PROJECT_ROOT / 'tests')
        config.CLASS_LABELS_CSV_PATH = str(PROJECT_ROOT / 'metadata' / 'class_labels_indices.csv')
        config.ONTOLOGY_PATH = str(PROJECT_ROOT / 'metadata' / 'ontology.json')
        config.CUSTOM_CATEGORIES = {
            'Noise': [],
            'Channel, environment and background': []
        }

        # Load ontology and mappings
        self.class_label_to_id, self.class_id_to_label = load_class_labels(config.CLASS_LABELS_CSV_PATH)
        ontology = load_ontology(config.ONTOLOGY_PATH)
        self.id_to_class, self.name_to_class_id = build_mappings(ontology)
        self.parent_to_children, self.child_to_parents = build_parent_child_mappings(ontology)

    def tearDown(self):
        # Restore original configurations
        config.PREDICTIONS_ROOT_DIR = self.original_predictions_root_dir
        config.CUSTOM_CATEGORIES = self.original_custom_categories
        config.CLASS_LABELS_CSV_PATH = self.original_class_labels_csv_path
        config.ONTOLOGY_PATH = self.original_ontology_path

    def get_all_descendants(self, class_id, parent_to_children, descendants=None):
        if descendants is None:
            descendants = set()
        children = parent_to_children.get(class_id, [])
        for child_id in children:
            descendants.add(child_id)
            self.get_all_descendants(child_id, parent_to_children, descendants)
        return descendants

    def test_threshold_0(self):
        # Threshold 0.0
        config.CONFIDENCE_THRESHOLD = 0.0
        config.SELECTED_RECORDERS = ['01', '02']
        config.SELECTED_CLASSES = ['Noise', 'Channel, environment and background',
                                'Environmental noise', 'Static', 'Mains hum',
                                'Inside, small room', 'Pink noise', 'Throbbing',
                                'Hubbub, speech noise, speech babble']
        config.SELECTED_DAYS = ['20231116']
        config.SELECTED_HOURS = ['2023-11-16 07:00', '2023-11-16 08:00']

        result = load_and_process_data()
        self.assertIsNotNone(result, "No data processed. Check selected parameters and thresholds.")
        data_counts, class_label_to_id, id_to_class, parent_to_children, name_to_class_id = result

        # Get the IDs of the main classes
        noise_id = name_to_class_id['Noise']
        channel_id = name_to_class_id['Channel, environment and background']

        # Initialize the counters
        total_noise_count = 0
        total_channel_env_bg_count = 0

        # Sum the counts of each hour
        for hour, counts in data_counts.items():
            for class_id, count in counts.items():
                # Get ancestors of the class
                ancestors = get_ancestors(class_id, self.child_to_parents)
                # If 'Noise' is among the ancestors or the class itself
                if noise_id in ancestors or class_id == noise_id:
                    total_noise_count += count
                # If 'Channel, environment and background' is among the ancestors or the class itself
                if channel_id in ancestors or class_id == channel_id:
                    total_channel_env_bg_count += count

        # Expected counts based on ontology
        expected_total_noise_events = 240  # Adjusted since 'Inside, small room' is not under 'Noise'
        expected_total_channel_events = 280

        print(f"\nFinal counts:")
        print(f"Total noise count: {total_noise_count}")
        print(f"Total channel/env/bg count: {total_channel_env_bg_count}")

        self.assertEqual(total_noise_count, expected_total_noise_events,
                        f"Total 'Noise' count should be {expected_total_noise_events}, got {total_noise_count}")
        self.assertEqual(total_channel_env_bg_count, expected_total_channel_events,
                        f"Total 'Channel, environment and background' count should be {expected_total_channel_events}")


    def test_threshold_05(self):
        # Threshold 0.5
        config.CONFIDENCE_THRESHOLD = 0.5
        config.SELECTED_RECORDERS = ['01', '02']
        config.SELECTED_CLASSES = ['Noise', 'Channel, environment and background',
                                   'Environmental noise', 'Static', 'Mains hum',
                                   'Inside, small room', 'Pink noise', 'Throbbing',
                                   'Hubbub, speech noise, speech babble']
        config.SELECTED_DAYS = ['20231116']
        config.SELECTED_HOURS = ['2023-11-16 07:00', '2023-11-16 08:00']

        result = load_and_process_data()
        self.assertIsNotNone(result, "No data processed. Check selected parameters and thresholds.")
        data_counts, class_label_to_id, id_to_class, parent_to_children, name_to_class_id = result

        # Rest of your test code remains the same, but adjust expected counts if necessary
        # ...

    def test_threshold_1(self):
        # Threshold 1.0
        config.CONFIDENCE_THRESHOLD = 1.0
        config.SELECTED_RECORDERS = ['01', '02']
        config.SELECTED_CLASSES = ['Noise', 'Channel, environment and background',
                                   'Environmental noise', 'Static', 'Mains hum',
                                   'Inside, small room', 'Pink noise', 'Throbbing',
                                   'Hubbub, speech noise, speech babble']
        config.SELECTED_DAYS = ['20231116']
        config.SELECTED_HOURS = ['2023-11-16 07:00', '2023-11-16 08:00']

        result = load_and_process_data()
        self.assertIsNotNone(result, "No data processed. Check selected parameters and thresholds.")
        data_counts, class_label_to_id, id_to_class, parent_to_children, name_to_class_id = result

        # Verify counts are zero for all classes
        for class_name in config.SELECTED_CLASSES:
            class_id = name_to_class_id[class_name]
            class_count = 0
            for hour_counts in data_counts.values():
                class_count += hour_counts.get(class_id, 0)
            self.assertEqual(class_count, 0, f"Total '{class_name}' count should be 0")

if __name__ == '__main__':
    unittest.main()
