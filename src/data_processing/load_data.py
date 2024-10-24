# src/data_processing/load_data.py

import json
import csv
import src.config as config 
from collections import defaultdict
# Define the nested dict factory function at the module level

def get_leaf_nodes(class_id, parent_to_children):
    """Recursively find all leaf nodes (nodes without children) under a given class ID."""
    children = parent_to_children.get(class_id, [])
    if not children:
        return [class_id]
    leaves = []
    for child in children:
        leaves.extend(get_leaf_nodes(child, parent_to_children))
    return leaves

def nested_dict():
    return dict(int)

# Function to load the ontology
def load_ontology(ontology_path):
    with open(ontology_path, 'r') as f:
        ontology = json.load(f)
    return ontology

# Function to load class labels from CSV
def load_class_labels(csv_path):
    """Load class labels and their IDs from a CSV file."""
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

# Function to build mappings and hierarchy
def build_mappings(ontology):
    id_to_class = {item['id']: item for item in ontology}
    name_to_class_id = {item['name']: item['id'] for item in ontology}
    return id_to_class, name_to_class_id

def build_parent_child_mappings(ontology):
    parent_to_children_dd = defaultdict(list)
    child_to_parents_dd = defaultdict(list)
    for class_item in ontology:
        parent_id = class_item['id']
        for child_id in class_item.get('child_ids', []):
            parent_to_children_dd[parent_id].append(child_id)
            child_to_parents_dd[child_id].append(parent_id)
    parent_to_children = dict(parent_to_children_dd)
    child_to_parents = dict(child_to_parents_dd)
    return parent_to_children, child_to_parents

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

def map_classes_to_categories(ontology, custom_categories, id_to_class, name_to_class_id, parent_to_children, child_to_parents):
    class_id_to_category_info = {}
    
    # Map main categories and subcategories to their IDs
    category_name_to_id = {}
    subcategory_id_to_category = {}
    for category_name, subcategory_names in custom_categories.items():
        category_id = name_to_class_id.get(category_name)
        if category_id:
            category_name_to_id[category_name] = category_id
            # Map subcategory IDs to category name
            for sub_name in subcategory_names:
                sub_id = name_to_class_id.get(sub_name)
                if sub_id:
                    subcategory_id_to_category[sub_id] = (category_name, sub_name)
                else:
                    print(f"Warning: Subcategory '{sub_name}' not found in ontology.")
        else:
            print(f"Warning: Category '{category_name}' not found in ontology.")
    
    # For each class, determine its category and subcategory
    for class_item in ontology:
        class_id = class_item['id']
        class_name = class_item['name']
        
        # Initialize category and subcategory
        category = None
        subcategory = None
        
        # Get all ancestors of the class
        ancestors = get_ancestors(class_id, child_to_parents)
        
        # Check if the class itself is a subcategory
        if class_id in subcategory_id_to_category:
            category, subcategory = subcategory_id_to_category[class_id]
        else:
            # Look for matching subcategories among ancestors
            matching_subcategories = ancestors & set(subcategory_id_to_category.keys())
            if matching_subcategories:
                # Choose the closest subcategory
                closest_sub_id = max(matching_subcategories, key=lambda x: len(get_ancestors(x, child_to_parents)))
                category, subcategory = subcategory_id_to_category[closest_sub_id]
            else:
                # Check if the class itself is a main category
                if class_id in category_name_to_id.values():
                    category = class_name
                    subcategory = class_name
                else:
                    # Look for matching main categories among ancestors
                    matching_categories = ancestors & set(category_name_to_id.values())
                    if matching_categories:
                        category_id = next(iter(matching_categories))
                        category = id_to_class[category_id]['name']
                        subcategory = class_name  # Use the class itself as subcategory
                    else:
                        # If no match, mark as 'Uncategorized'
                        category = 'Uncategorized'
                        subcategory = class_name
        
        # Assign to mapping
        class_id_to_category_info[class_id] = {
            'category': category,
            'subcategory': subcategory
        }
    
    return class_id_to_category_info