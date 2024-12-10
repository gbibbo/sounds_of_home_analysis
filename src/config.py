# src/config.py

import os

# Root directory containing the recorder folders
PREDICTIONS_ROOT_DIR = 'assets/sample_data_light'
#PREDICTIONS_ROOT_DIR = 'assets/dataset'
#PREDICTIONS_ROOT_DIR = '/vol/research/datasets/audio/AI4S_SoH/VITALISE data light/Cnn14_DecisionLevelAtt_light'

# Paths to the ontology and class labels files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY_PATH = os.path.join(BASE_DIR, '../metadata/ontology.json')
CLASS_LABELS_CSV_PATH = os.path.join(BASE_DIR, '../metadata/class_labels_indices.csv')

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.35  # Default value
CONFIDENCE_THRESHOLD_STR = '0.35'  # Default string value

# Default confidence threshold for labels with 100% quality estimation
DEFAULT_CONFIDENCE_THRESHOLD = 0.35  # You can change this value as needed
USE_LABEL_QUALITY_THRESHOLDS = True  # Set to True to use variable thresholds

# Path to the label quality estimates file
LABEL_QUALITY_ESTIMATES_PATH = os.path.join(BASE_DIR, '../metadata/audioset_data.csv')

# User-configurable parameters
SELECTED_RECORDERS = []  # Will be populated dynamically based on available directories
SELECTED_DAYS = []  # Will be dynamically populated based on available data
SELECTED_HOURS = []  # Will be dynamically populated based on available data
SELECTED_CLASSES = []  # No classes preselected

# Constants for speech muting
MUTE_LABELS = [
    "Speech",
    "Singing",
    "Male singing",
    "Female singing",
    "Child singing",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Conversation",
    "Narration, monologue",
    "Music"
]
MUTE_THRESHOLD = 0.2

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

# Custom categories for interface
CUSTOM_CATEGORIES_INTERFACE = {
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
        {
            'Domestic sounds, home sounds': [
                'Door',
                'Cupboard open or close',
                'Drawer open or close',
                'Dishes, pots, and pans',
                'Cutlery, silverware',
                'Chopping (food)',
                'Frying (food)',
                'Microwave oven',
                'Blender',
                'Kettle whistle',
                'Water tap, faucet',
                'Sink (filling or washing)',
                'Bathtub (filling or washing)',
                'Hair dryer',
                'Toilet flush',
                'Toothbrush',
                'Vacuum cleaner',
                'Zipper (clothing)',
                'Velcro, hook and loop fastener',
                'Keys jangling',
                'Coin (dropping)',
                'Packing tape, duct tape',
                'Scissors',
                'Electric shaver, electric razor',
                'Shuffling cards',
                'Typing',
                'Writing'
            ]
        },
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