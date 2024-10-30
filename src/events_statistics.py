
import os
import json
import csv
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

import os

# Root directory containing the recorder folders
#PREDICTIONS_ROOT_DIR = 'assets/sample_data'
PREDICTIONS_ROOT_DIR = '/vol/research/datasets/audio/AI4S_SoH/VITALISE data light/Cnn14_DecisionLevelAtt_light'

# Paths to the ontology and class labels files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY_PATH = os.path.join(BASE_DIR, '../metadata/ontology.json')
CLASS_LABELS_CSV_PATH = os.path.join(BASE_DIR, '../metadata/class_labels_indices.csv')

# Archivo JSON de salida
OUTPUT_JSON_PATH = 'events_statistics_results.json'

# Configuración de umbrales
DEFAULT_CONFIDENCE_THRESHOLD = 0.2  # Valor de umbral predeterminado

# Umbrales dinámicos basados en la calidad de las etiquetas
USE_LABEL_QUALITY_THRESHOLDS = True  # Establece a True para ajustar los umbrales basados en la calidad de las etiquetas

# Generar gráficos
GENERATE_GRAPHS = True  # Establece a False si no deseas generar gráficos

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

# Label quality estimates
LABEL_QUALITY_ESTIMATES = {
    'Music': 100,
    'Speech': 100,
    'Vehicle': 100,
    'Musical instrument': 100,
    'Plucked string instrument': 100,
    'Singing': 100,
    'Car': 100,
    'Animal': 100,
    'Outside, rural or natural': 100,
    'Violin, fiddle': 100,
    'Bird': 100,
    'Drum': 100,
    'Engine': 100,
    'Narration, monologue': 100,
    'Drum kit': 100,
    'Acoustic guitar': 100,
    'Dog': 100,
    'Child speech, kid speaking': 100,
    'Bass drum': 100,
    'Rail transport': 100,
    'Motor vehicle (road)': 100,
    'Water': 100,
    'Female speech, woman speaking': 100,
    'Siren': 100,
    'Railroad car, train wagon': 100,
    'Tools': 100,
    'Silence': 100,
    'Snare drum': 100,
    'Wind': 100,
    'Bird vocalization, bird call, bird song': 100,
    'Fowl': 100,
    'Wind instrument, woodwind instrument': 100,
    'Emergency vehicle': 100,
    'Laughter': 100,
    'Chirp, tweet': 100,
    'Rapping': 100,
    'Cheering': 100,
    'Gunshot, gunfire': 100,
    'Radio': 100,
    'Cat': 100,
    'Hi-hat': 100,
    'Helicopter': 100,
    'Fireworks': 100,
    'Stream': 100,
    'Bark': 100,
    'Baby cry, infant cry': 100,
    'Snoring': 100,
    'Train horn': 100,
    'Double bass': 100,
    'Explosion': 100,
    'Crowing, cock-a-doodle-doo': 100,
    'Bleat': 100,
    'Computer keyboard': 100,
    'Civil defense siren': 100,
    'Bee, wasp, etc.': 100,
    'Bell': 100,
    'Chainsaw': 100,
    'Oink': 100,
    'Tick': 100,
    'Tabla': 100,
    'Liquid': 100,
    'Traffic noise, roadway noise': 100,
    'Beep, bleep': 100,
    'Frying (food)': 100,
    'Whack, thwack': 100,
    'Sink (filling or washing)': 100,
    'Burping, eructation': 100,
    'Fart': 100,
    'Sneeze': 100,
    'Aircraft engine': 100,
    'Arrow': 100,
    'Giggle': 100,
    'Hiccup': 100,
    'Cough': 100,
    'Cricket': 100,
    'Sawing': 100,
    'Tambourine': 100,
    'Pump (liquid)': 100,
    'Squeak': 100,
    'Male speech, man speaking': 90,
    'Keyboard (musical)': 90,
    'Pigeon, dove': 90,
    'Motorboat, speedboat': 90,
    'Female singing': 90,
    'Brass instrument': 90,
    'Motorcycle': 90,
    'Choir': 90,
    'Race car, auto racing': 90,
    'Chicken, rooster': 90,
    'Idling': 90,
    'Sampler': 90,
    'Ukulele': 90,
    'Synthesizer': 90,
    'Cymbal': 90,
    'Spray': 90,
    'Accordion': 90,
    'Scratching (performance technique)': 90,
    'Child singing': 90,
    'Cluck': 90,
    'Water tap, faucet': 90,
    'Applause': 90,
    'Toilet flush': 90,
    'Whistling': 90,
    'Vacuum cleaner': 90,
    'Meow': 90,
    'Chatter': 90,
    'Whoop': 90,
    'Sewing machine': 90,
    'Bagpipes': 90,
    'Subway, metro, underground': 90,
    'Walk, footsteps': 90,
    'Whispering': 90,
    'Crying, sobbing': 90,
    'Thunder': 90,
    'Didgeridoo': 90,
    'Church bell': 90,
    'Ringtone': 90,
    'Buzzer': 90,
    'Splash, splatter': 90,
    'Fire alarm': 90,
    'Chime': 90,
    'Babbling': 90,
    'Glass': 90,
    'Chewing, mastication': 90,
    'Microwave oven': 90,
    'Air horn, truck horn': 90,
    'Growling': 90,
    'Telephone bell ringing': 90,
    'Moo': 90,
    'Change ringing (campanology)': 90,
    'Hands': 90,
    'Camera': 90,
    'Pour': 90,
    'Croak': 90,
    'Pant': 90,
    'Finger snapping': 90,
    'Gargling': 90,
    'Inside, small room': 89,
    'Outside, urban or manmade': 89,
    'Truck': 89,
    'Bowed string instrument': 89,
    'Medium engine (mid frequency)': 89,
    'Marimba, xylophone': 89,
    'Aircraft': 89,
    'Cello': 89,
    'Flute': 89,
    'Glockenspiel': 89,
    'Power tool': 89,
    'Fixed-wing aircraft, airplane': 89,
    'Waves, surf': 89,
    'Duck': 89,
    'Clarinet': 89,
    'Goat': 89,
    'Honk': 89,
    'Skidding': 89,
    'Hammond organ': 89,
    'Electronic organ': 89,
    'Thunderstorm': 89,
    'Steelpan': 89,
    'Slap, smack': 89,
    'Battle cry': 89,
    'Percussion': 88,
    'Trombone': 88,
    'Banjo': 88,
    'Mandolin': 86,
    'Guitar': 80,
    'Strum': 80,
    'Boat, Water vehicle': 80,
    'Accelerating, revving, vroom': 80,
    'Electric guitar': 80,
    'Orchestra': 80,
    'Wind noise (microphone)': 80,
    'Effects unit': 80,
    'Livestock, farm animals, working animals': 80,
    'Police car (siren)': 80,
    'Rain': 80,
    'Printer': 80,
    'Drum machine': 80,
    'Fire engine, fire truck (siren)': 80,
    'Insect': 80,
    'Skateboard': 80,
    'Coo': 80,
    'Conversation': 80,
    'Typing': 80,
    'Harp': 80,
    'Thump, thud': 80,
    'Mechanisms': 80,
    'Canidae, dogs, wolves': 80,
    'Chuckle, chortle': 80,
    'Rub': 80,
    'Boom': 80,
    'Hubbub, speech noise, speech babble': 80,
    'Telephone': 80,
    'Blender': 80,
    'Whimper': 80,
    'Screaming': 80,
    'Wild animals': 80,
    'Pig': 80,
    'Artillery fire': 80,
    'Electric shaver, electric razor': 80,
    'Baby laughter': 80,
    'Crow': 80,
    'Howl': 80,
    'Breathing': 80,
    'Cattle, bovinae': 80,
    'Roaring cats (lions, tigers)': 80,
    'Clapping': 80,
    'Alarm': 80,
    'Chink, clink': 80,
    'Ding': 80,
    'Toot': 80,
    'Clock': 80,
    'Children shouting': 80,
    'Fill (with liquid)': 80,
    'Purr': 80,
    'Rumble': 80,
    'Boing': 80,
    'Breaking': 80,
    'Light engine (high frequency)': 80,
    'Cash register': 80,
    'Bicycle bell': 80,
    'Inside, large room or hall': 78,
    'Domestic animals, pets': 78,
    'Bass guitar': 78,
    'Electric piano': 78,
    'Trumpet': 78,
    'Horse': 78,
    'Mallet percussion': 78,
    'Organ': 78,
    'Bicycle': 78,
    'Rain on surface': 78,
    'Quack': 78,
    'Drill': 78,
    'Machine gun': 78,
    'Lawn mower': 78,
    'Smash, crash': 78,
    'Trickle, dribble': 78,
    'Frog': 78,
    'Writing': 78,
    'Steam whistle': 78,
    'Groan': 78,
    'Hammer': 78,
    'Doorbell': 78,
    'Shofar': 78,
    'Cowbell': 78,
    'Wail, moan': 78,
    'Bouncing': 78,
    'Distortion': 75,
    'Vibraphone': 75,
    'Air brake': 75,
    'Field recording': 75,
    'Piano': 70,
    'Male singing': 70,
    'Bus': 70,
    'Wood': 70,
    'Tap': 70,
    'Ocean': 70,
    'Door': 70,
    'Vibration': 70,
    'Television': 70,
    'Harmonica': 70,
    'Basketball bounce': 70,
    'Clickety-clack': 70,
    'Dishes, pots, and pans': 70,
    'Crumpling, crinkling': 70,
    'Sitar': 70,
    'Tire squeal': 70,
    'Fly, housefly': 70,
    'Sizzle': 70,
    'Slosh': 70,
    'Engine starting': 70,
    'Mechanical fan': 70,
    'Stir': 70,
    'Children playing': 70,
    'Ping': 70,
    'Owl': 70,
    'Alarm clock': 70,
    'Car alarm': 70,
    'Telephone dialing, DTMF': 70,
    'Sine wave': 70,
    'Thunk': 70,
    'Coin (dropping)': 70,
    'Crunch': 70,
    'Zipper (clothing)': 70,
    'Mosquito': 70,
    'Shuffling cards': 70,
    'Pulleys': 70,
    'Toothbrush': 70,
    'Crowd': 67,
    'Saxophone': 67,
    'Rowboat, canoe, kayak': 67,
    'Steam': 67,
    'Ambulance (siren)': 67,
    'Goose': 67,
    'Crackle': 67,
    'Fire': 67,
    'Turkey': 67,
    'Heart sounds, heartbeat': 67,
    'Singing bowl': 67,
    'Reverberation': 67,
    'Clicking': 67,
    'Jet engine': 67,
    'Rodents, rats, mice': 67,
    'Typewriter': 67,
    'Caw': 67,
    'Knock': 67,
    'Ice cream truck, ice cream van': 67,
    'Stomach rumble': 67,
    'French horn': 63,
    'Roar': 63,
    'Theremin': 63,
    'Pulse': 63,
    'Train': 60,
    'Run': 60,
    'Vehicle horn, car horn, honking': 60,
    'Clip-clop': 60,
    'Sheep': 60,
    'Whoosh, swoosh, swish': 60,
    'Timpani': 60,
    'Throbbing': 60,
    'Firecracker': 60,
    'Belly laugh': 60,
    'Train whistle': 60,
    'Whistle': 60,
    'Whip': 60,
    'Gush': 60,
    'Biting': 60,
    'Scissors': 60,
    'Clang': 60,
    'Single-lens reflex camera': 57,
    'Chorus effect': 57,
    'Inside, public space': 56,
    'Steel guitar, slide guitar': 56,
    'Waterfall': 56,
    'Hum': 56,
    'Raindrop': 56,
    'Propeller, airscrew': 56,
    'Filing (rasp)': 56,
    'Reversing beeps': 56,
    'Shatter': 56,
    'Sanding': 56,
    'Wheeze': 56,
    'Hoot': 56,
    'Bow-wow': 50,
    'Car passing by': 50,
    'Tick-tock': 50,
    'Hiss': 50,
    'Snicker': 50,
    'Whimper (dog)': 50,
    'Shout': 50,
    'Echo': 50,
    'Rattle': 50,
    'Sliding door': 50,
    'Gobble': 50,
    'Plop': 50,
    'Yell': 50,
    'Drip': 50,
    'Neigh, whinny': 50,
    'Bellow': 50,
    'Keys jangling': 50,
    'Ding-dong': 50,
    'Buzz': 50,
    'Scratch': 50,
    'Rattle (instrument)': 50,
    'Hair dryer': 50,
    'Dial tone': 50,
    'Tearing': 50,
    'Bang': 50,
    'Noise': 50,
    'Bird flight, flapping wings': 50,
    'Grunt': 50,
    'Jackhammer': 50,
    'Drawer open or close': 50,
    'Whir': 50,
    'Tuning fork': 50,
    'Squawk': 50,
    'Jingle bell': 44,
    'Smoke detector, smoke alarm': 44,
    'Train wheels squealing': 44,
    'Caterwaul': 44,
    'Mouse': 44,
    'Crack': 44,
    'Whale vocalization': 44,
    'Squeal': 44,
    'Zither': 43,
    'Rimshot': 40,
    'Drum roll': 40,
    'Burst, pop': 40,
    'Wood block': 40,
    'Harpsichord': 40,
    'White noise': 40,
    'Bathtub (filling or washing)': 40,
    'Snake': 40,
    'Environmental noise': 40,
    'String section': 40,
    'Cacophony': 40,
    'Maraca': 40,
    'Snort': 40,
    'Yodeling': 40,
    'Electric toothbrush': 40,
    'Cupboard open or close': 40,
    'Sound effect': 38,
    'Tapping (guitar technique)': 38,
    'Ship': 38,
    'Sniff': 38,
    'Pink noise': 33,
    'Tubular bells': 33,
    'Gong': 33,
    'Flap': 33,
    'Throat clearing': 33,
    'Sigh': 33,
    'Busy signal': 33,
    'Zing': 33,
    'Sidetone': 33,
    'Crushing': 33,
    'Yip': 30,
    'Gurgling': 30,
    'Jingle, tinkle': 30,
    'Boiling': 30,
    'Mains hum': 30,
    'Humming': 30,
    'Sonar': 30,
    'Gasp': 30,
    'Power windows, electric windows': 30,
    'Splinter': 30,
    'Heart murmur': 29,
    'Air conditioning': 29,
    'Pizzicato': 25,
    'Ratchet, pawl': 22,
    'Chirp tone': 22,
    'Heavy engine (low frequency)': 20,
    'Rustling leaves': 20,
    'Speech synthesizer': 20,
    'Rustle': 20,
    'Clatter': 20,
    'Slam': 20,
    'Eruption': 20,
    'Cap gun': 20,
    'Synthetic singing': 20,
    'Shuffle': 20,
    'Wind chime': 20,
    'Chop': 20,
    'Scrape': 20,
    'Squish': 20,
    'Foghorn': 20,
    'Dental drill, dentist\'s drill': 20,
    'Harmonic': 17,
    'Static': 13,
    'Sailboat, sailing ship': 11,
    'Cutlery, silverware': 11,
    'Gears': 11,
    'Chopping (food)': 11,
    'Creak': 11,
    'Fusillade': 10,
    'Roll': 0,
    'Electronic tuner': 0,
    'Patter': 0,
    'Electronic music': 50,
    'Dubstep': 50,
    'Techno': 50,
    'Rock and roll': 50,
    'Pop music': 50,
    'Rock music': 50,
    'Hip hop music': 50,
    'Classical music': 50,
    'Soundtrack music': 50,
    'House music': 50,
    'Heavy metal': 50,
    'Exciting music': 50,
    'Country': 50,
    'Electronica': 50,
    'Rhythm and blues': 50,
    'Background music': 50,
    'Dance music': 50,
    'Jazz': 50,
    'Mantra': 50,
    'Blues': 50,
    'Trance music': 50,
    'Electronic dance music': 50,
    'Theme music': 50,
    'Gospel music': 50,
    'Music of Latin America': 50,
    'Disco': 50,
    'Tender music': 50,
    'Punk rock': 50,
    'Funk': 50,
    'Music of Asia': 50,
    'Drum and bass': 50,
    'Vocal music': 50,
    'Progressive rock': 50,
    'Music for children': 50,
    'Video game music': 50,
    'Lullaby': 50,
    'Reggae': 50,
    'New-age music': 50,
    'Christian music': 50,
    'Independent music': 50,
    'Soul music': 50,
    'Music of Africa': 50,
    'Ambient music': 50,
    'Bluegrass': 50,
    'Afrobeat': 50,
    'Salsa music': 50,
    'Music of Bollywood': 50,
    'Beatboxing': 50,
    'Flamenco': 50,
    'Psychedelic rock': 50,
    'Opera': 50,
    'Folk music': 50,
    'Christmas music': 50,
    'Middle Eastern music': 50,
    'Grunge': 50,
    'Song': 50,
    'A capella': 50,
    'Sad music': 50,
    'Traditional music': 50,
    'Scary music': 50,
    'Ska': 50,
    'Chant': 50,
    'Carnatic music': 50,
    'Swing music': 50,
    'Happy music': 50,
    'Jingle (music)': 50,
    'Funny music': 50,
    'Angry music': 50,
    'Wedding music': 50,
    'Engine knocking': 50
}

def load_ontology(ontology_path):
    """Carga la ontología desde un archivo JSON."""
    with open(ontology_path, 'r') as f:
        ontology = json.load(f)
    return ontology

def load_class_labels(csv_path):
    """Carga las etiquetas de clase y sus IDs correspondientes desde un archivo CSV."""
    class_label_to_id = {}
    class_id_to_label = {}
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            display_name = row['display_name'].strip('"').strip()
            class_id = row['mid']
            class_label_to_id[display_name.lower()] = class_id
            class_id_to_label[class_id] = display_name
    return class_label_to_id, class_id_to_label

def build_mappings(ontology):
    """
    Construye los mapeos necesarios desde la ontología.
    Retorna:
        id_to_class: Mapeo de ID de clase a detalles de la clase.
        name_to_class_id: Mapeo de nombre de clase a ID de clase.
    """
    id_to_class = {item['id']: item for item in ontology}
    name_to_class_id = {item['name']: item['id'] for item in ontology}
    return id_to_class, name_to_class_id

def build_parent_child_mappings(ontology):
    """
    Construye mapeos de padre a hijos y de hijo a padres.
    Retorna:
        parent_to_children: Mapeo de ID de padre a lista de IDs de hijos.
        child_to_parents: Mapeo de ID de hijo a lista de IDs de padres.
    """
    parent_to_children = defaultdict(list)
    child_to_parents = defaultdict(list)
    for class_item in ontology:
        parent_id = class_item['id']
        for child_id in class_item.get('child_ids', []):
            parent_to_children[parent_id].append(child_id)
            child_to_parents[child_id].append(parent_id)
    return parent_to_children, child_to_parents

def get_ancestors(class_id, child_to_parents, memo=None):
    """
    Obtiene recursivamente todos los ancestros de un ID de clase dado.
    Usa memoización para optimizar búsquedas repetidas.
    """
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

def map_classes_to_categories(ontology, custom_categories, id_to_class, name_to_class_id, parent_to_children, child_to_parents):
    """
    Mapea cada clase en la ontología a su categoría y subcategoría correspondiente.
    Retorna:
        class_id_to_category_info: Mapeo de ID de clase a su categoría y subcategoría.
    """
    class_id_to_category_info = {}
    
    # Mapea categorías principales y subcategorías a sus IDs
    category_name_to_id = {}
    subcategory_id_to_category = {}
    for category_name, subcategory_names in custom_categories.items():
        category_id = name_to_class_id.get(category_name)
        if category_id:
            category_name_to_id[category_name] = category_id
            # Mapea IDs de subcategorías al nombre de la categoría
            for sub_name in subcategory_names:
                sub_id = name_to_class_id.get(sub_name)
                if sub_id:
                    subcategory_id_to_category[sub_id] = (category_name, sub_name)
                else:
                    print(f"Advertencia: Subcategoría '{sub_name}' no encontrada en la ontología.")
        else:
            print(f"Advertencia: Categoría '{category_name}' no encontrada en la ontología.")
    
    # Para cada clase, determina su categoría y subcategoría
    for class_item in ontology:
        class_id = class_item['id']
        class_name = class_item['name']
        
        # Inicializa categoría y subcategoría
        category = None
        subcategory = None
        
        # Obtiene todos los ancestros de la clase
        ancestors = get_ancestors(class_id, child_to_parents)
        
        # Verifica si la clase en sí es una subcategoría
        if class_id in subcategory_id_to_category:
            category, subcategory = subcategory_id_to_category[class_id]
        else:
            # Busca subcategorías coincidentes entre los ancestros
            matching_subcategories = ancestors & set(subcategory_id_to_category.keys())
            if matching_subcategories:
                # Elige la subcategoría más específica (más profunda en la jerarquía)
                closest_sub_id = max(matching_subcategories, key=lambda x: len(get_ancestors(x, child_to_parents)))
                category, subcategory = subcategory_id_to_category[closest_sub_id]
            else:
                # Verifica si la clase en sí es una categoría principal
                if class_id in category_name_to_id.values():
                    category = class_name
                    subcategory = class_name
                else:
                    # Busca categorías principales coincidentes entre los ancestros
                    matching_categories = ancestors & set(category_name_to_id.values())
                    if matching_categories:
                        category_id = next(iter(matching_categories))
                        category = id_to_class[category_id]['name']
                        subcategory = class_name  # Usa la clase misma como subcategoría
                    else:
                        # Si no coincide con ninguna categoría, marca como 'Uncategorized'
                        category = 'Uncategorized'
                        subcategory = class_name
        
        # Asigna al mapeo
        class_id_to_category_info[class_id] = {
            'category': category,
            'subcategory': subcategory
        }
    
    return class_id_to_category_info

def process_predictions(predictions_root_dir):
    """
    Recorre archivos JSON de predicciones y recolecta todas las predicciones.
    Retorna:
        all_predictions: Lista de diccionarios de predicciones.
    """
    all_predictions = []
    
    # Recorre todos los subdirectorios en el directorio raíz
    for subdir_name in os.listdir(predictions_root_dir):
        subdir_path = os.path.join(predictions_root_dir, subdir_name)
        if os.path.isdir(subdir_path):
            # Recorre todos los archivos .json en el subdirectorio
            for filename in os.listdir(subdir_path):
                if filename.endswith('.json') or filename.endswith('_light.json'):
                    filepath = os.path.join(subdir_path, filename)
                    with open(filepath, 'r') as f:
                        try:
                            data = json.load(f)
                            for frame in data:
                                # Verifica si el frame tiene predicciones
                                if 'predictions' in frame:
                                    for prediction in frame['predictions']:
                                        all_predictions.append(prediction)
                                else:
                                    print(f"Advertencia: Frame en {filepath} no contiene 'predictions'.")
                        except json.JSONDecodeError:
                            print(f"Error: Falló al parsear el archivo JSON {filepath}.")
    return all_predictions

def compute_class_thresholds(class_label_to_id, class_id_to_label):
    """
    Computa umbrales específicos de clase basados en estimaciones de calidad de las etiquetas.
    Retorna:
        class_thresholds: Diccionario que mapea etiquetas de clase a umbrales.
    """
    class_thresholds = {}
    
    if USE_LABEL_QUALITY_THRESHOLDS:
        # Ajusta umbrales basados en estimaciones de calidad de etiquetas
        for class_label_lower, class_id in class_label_to_id.items():
            class_label = class_id_to_label[class_id]
            label_quality = LABEL_QUALITY_ESTIMATES.get(class_label, 50)  # Valor por defecto si no se encuentra
            # Mapea calidad de etiqueta a umbral
            if label_quality >= 100:
                threshold = 0.2
            elif label_quality >= 90:
                threshold = 0.2
            elif label_quality >= 80:
                threshold = 0.2
            elif label_quality >= 70:
                threshold = 0.2
            elif label_quality >= 60:
                threshold = 0.2
            else:
                threshold = 0.2
            class_thresholds[class_label] = threshold
    else:
        # Usa el umbral de confianza predeterminado para todas las clases
        for class_label_lower, class_id in class_label_to_id.items():
            class_label = class_id_to_label[class_id]
            class_thresholds[class_label] = DEFAULT_CONFIDENCE_THRESHOLD
    
    return class_thresholds

def filter_predictions(all_predictions, class_thresholds):
    """
    Filtra predicciones basadas en el método de umbral seleccionado.
    Retorna:
        filtered_predictions: Lista de diccionarios de predicciones.
    """
    filtered_predictions = []
    for pred in all_predictions:
        class_label = (pred.get('class_label') or pred.get('class', '')).strip()
        probability = pred.get('probability') or pred.get('prob', 0)
        threshold = class_thresholds.get(class_label, DEFAULT_CONFIDENCE_THRESHOLD)
        
        if probability >= threshold:
            filtered_predictions.append(pred)
    return filtered_predictions

def map_predictions_to_categories(predictions, class_label_to_id, class_id_to_category_info):
    """
    Mapea cada predicción a su categoría y subcategoría correspondiente.
    Retorna:
        predicted_categories: Lista de diccionarios con claves 'category' y 'subcategory'.
        unmapped_classes: Conjunto de etiquetas de clase que no se pudieron mapear.
    """
    predicted_categories = []
    unmapped_classes = set()
    
    for pred in predictions:
        class_label = pred.get('class_label') or pred.get('class', '').strip()
        class_label_lower = class_label.lower()
        class_id = class_label_to_id.get(class_label_lower)
        if class_id:
            category_info = class_id_to_category_info.get(class_id)
            if category_info:
                predicted_categories.append(category_info)
                pred['mapped_category'] = category_info['category']
                pred['mapped_subcategory'] = category_info['subcategory'] or category_info['category']
            else:
                # Asigna a 'Uncategorized' si no se encuentra la categoría
                predicted_categories.append({'category': 'Uncategorized', 'subcategory': class_label})
                pred['mapped_category'] = 'Uncategorized'
                pred['mapped_subcategory'] = class_label
                unmapped_classes.add(class_label)
        else:
            # Asigna a 'Uncategorized' si no se encuentra la etiqueta de clase
            predicted_categories.append({'category': 'Uncategorized', 'subcategory': class_label})
            pred['mapped_category'] = 'Uncategorized'
            pred['mapped_subcategory'] = class_label
            unmapped_classes.add(class_label)
    
    return predicted_categories, unmapped_classes

def generate_statistics(predicted_categories):
    """
    Genera conteos para categorías y subcategorías.
    Retorna:
        category_counts: Objeto Counter para categorías.
        subcategory_counts: defaultdict de objetos Counter para subcategorías dentro de cada categoría.
    """
    category_counts = Counter()
    subcategory_counts = defaultdict(Counter)
    
    for entry in predicted_categories:
        category = entry['category']
        subcategory = entry['subcategory'] if entry['subcategory'] else entry['category']
        category_counts[category] += 1
        subcategory_counts[category][subcategory] += 1
    
    return category_counts, subcategory_counts

def save_analysis_results(category_counts, subcategory_counts, output_json_path):
    """
    Guarda los resultados del análisis en un archivo JSON.
    """
    analysis_results = {
        'category_counts': dict(category_counts),
        'subcategory_counts': {cat: dict(sub_counts) for cat, sub_counts in subcategory_counts.items()}
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(analysis_results, f, indent=4)

# Añade las funciones de graficación aquí
def plot_main_categories(category_counts):
    """
    Genera un gráfico de barras para las categorías principales mostrando porcentajes.
    """
    total_count = sum(category_counts.values())
    sorted_categories = category_counts.most_common()
    categories = [cat for cat, _ in sorted_categories]
    counts = [count for _, count in sorted_categories]
    percentages = [count / total_count * 100 for count in counts]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, percentages, color='skyblue', edgecolor='black')
    plt.xlabel('Categories')
    plt.ylabel('Percentage of Samples (%)')
    plt.title('Sound Event Statistics by Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Añade etiquetas de porcentaje encima de cada barra
    for bar, percent in zip(bars, percentages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(percentages)*0.01, f'{percent:.1f}%', ha='center', va='bottom')
    
    plt.show()

def plot_subcategories(subcategory_counts):
    """
    Genera gráficos de barras para cada categoría principal mostrando porcentajes de subcategorías.
    """
    for category, sub_counts in subcategory_counts.items():
        if category == 'Uncategorized':
            continue  # Omite la categoría 'Uncategorized' si no deseas graficarla
        
        total_sub_count = sum(sub_counts.values())
        sorted_subcategories = sub_counts.most_common()
        subcategories = [sub for sub, _ in sorted_subcategories]
        counts = [count for _, count in sorted_subcategories]
        percentages = [count / total_sub_count * 100 for count in counts]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(subcategories, percentages, color='lightgreen', edgecolor='black')
        plt.xlabel('Subcategories')
        plt.ylabel('Percentage of Samples (%)')
        plt.title(f"Subcategory Breakdown in '{category}'")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Añade etiquetas de porcentaje encima de cada barra
        for bar, percent in zip(bars, percentages):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + max(percentages)*0.01, f'{percent:.1f}%', ha='center', va='bottom')
        
        plt.show()

def main():
    # Paso 1: Carga la ontología y las etiquetas de clase
    ontology = load_ontology(ONTOLOGY_PATH)
    class_label_to_id, class_id_to_label = load_class_labels(CLASS_LABELS_CSV_PATH)
    
    # Paso 2: Construye los mapeos necesarios
    id_to_class, name_to_class_id = build_mappings(ontology)
    parent_to_children, child_to_parents = build_parent_child_mappings(ontology)
    
    # Paso 3: Mapea clases a categorías y subcategorías
    class_id_to_category_info = map_classes_to_categories(
        ontology, CUSTOM_CATEGORIES, id_to_class, name_to_class_id,
        parent_to_children, child_to_parents
    )
    
    # Paso 4: Procesa las predicciones
    all_predictions = process_predictions(PREDICTIONS_ROOT_DIR)
    
    # Paso 5: Computa los umbrales específicos de clase
    class_thresholds = compute_class_thresholds(class_label_to_id, class_id_to_label)
    
    # Paso 6: Filtra las predicciones usando el método de umbral seleccionado
    filtered_predictions = filter_predictions(all_predictions, class_thresholds)
    
    # Paso 7: Mapea predicciones a categorías
    predicted_categories, unmapped_classes = map_predictions_to_categories(
        filtered_predictions, class_label_to_id, class_id_to_category_info
    )
    
    # Paso 8: Genera estadísticas
    category_counts, subcategory_counts = generate_statistics(predicted_categories)
    
    # Paso 9: Imprime estadísticas
    print("\nCategory Statistics:")
    for category, count in category_counts.items():
        print(f"{category}: {count}")
    
    print("\nSubcategory Statistics:")
    for category, sub_counts in subcategory_counts.items():
        print(f"\nSubcategories in '{category}':")
        for subcategory, count in sub_counts.items():
            print(f"  {subcategory}: {count}")
    
    # Paso 10: Guarda los resultados del análisis en JSON
    save_analysis_results(category_counts, subcategory_counts, OUTPUT_JSON_PATH)
    print(f"\nAnalysis results saved to {OUTPUT_JSON_PATH}")
    
    # Paso 11: Imprime clases no mapeadas si las hay
    if unmapped_classes:
        print("\nUnmapped Classes:")
        for cls in sorted(unmapped_classes):
            print(f"- '{cls}'")
    
    # Paso 12: Imprime el método de umbral utilizado
    if USE_LABEL_QUALITY_THRESHOLDS:
        print("\nUsing label quality-based thresholds:")
        for class_label, threshold in class_thresholds.items():
            print(f"- '{class_label}': Threshold = {threshold}")
    else:
        print(f"\nUsing fixed confidence threshold: {DEFAULT_CONFIDENCE_THRESHOLD}")
    
    # Paso 13: Genera gráficos si está habilitado
    if GENERATE_GRAPHS:
        plot_main_categories(category_counts)
        plot_subcategories(subcategory_counts)

if __name__ == "__main__":
    main()