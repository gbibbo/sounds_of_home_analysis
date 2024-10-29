# src/config.py

import os

# Root directory containing the recorder folders
#PREDICTIONS_ROOT_DIR = 'assets/sample_data_light'
PREDICTIONS_ROOT_DIR = '/vol/research/datasets/audio/AI4S_SoH/VITALISE data light/Cnn14_DecisionLevelAtt_light'

# Paths to the ontology and class labels files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY_PATH = os.path.join(BASE_DIR, '../metadata/ontology.json')
CLASS_LABELS_CSV_PATH = os.path.join(BASE_DIR, '../metadata/class_labels_indices.csv')

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.5  # Default value
CONFIDENCE_THRESHOLD_STR = '0.5'  # Default string value

# User-configurable parameters
SELECTED_RECORDERS = []  # Will be populated dynamically based on available directories
SELECTED_DAYS = []  # Will be dynamically populated based on available data
SELECTED_HOURS = []  # Will be dynamically populated based on available data
SELECTED_CLASSES = []  # No classes preselected

# Custom categories provided by the user
CUSTOM_CATEGORIES = {
    'Human sounds': [
        'Human voice',
        'Whistling',
        'Respiratory sounds',
        'Human locomotion',
        'Digestive',
        'Hands',
        'Heart sounds, heartbeat',
        'Otoacoustic emission',
        'Human group actions'
    ],
    'Source-ambiguous sounds': [
        'Generic impact sounds',
        'Surface contact',
        'Deformable shell',
        'Onomatopoeia',
        'Silence',
        'Other sourceless'
    ],
    'Animal': [
        'Animal',
        'Domestic animals, pets',
        'Livestock, farm animals, working animals',
        'Wild animals'
    ],
    'Sounds of things': [
        'Vehicle',
        'Engine',
        'Domestic sounds, home sounds',
        'Bell',
        'Alarm',
        'Mechanisms',
        'Tools',
        'Explosion',
        'Wood',
        'Glass',
        'Liquid',
        'Miscellaneous sources',
        'Specific impact sounds'
    ],
    'Music': [
        'Music',
        'Musical instrument',
        'Music genre',
        'Musical concepts',
        'Music role',
        'Music mood'
    ],
    'Natural sounds': [
        'Wind',
        'Thunderstorm',
        'Water',
        'Fire'
    ],
    'Channel, environment and background': [
        'Acoustic environment',
        'Noise',
        'Sound reproduction'
    ]
}