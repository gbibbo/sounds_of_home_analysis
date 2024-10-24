# src/data_processing/utils.py

import src.config as config 

def get_class_id(class_name, class_label_to_id, name_to_class_id):
    # Intentar obtener el ID directamente
    class_id = class_label_to_id.get(class_name.lower())
    if class_id:
        return class_id
    
    # Si no se encuentra, buscar en name_to_class_id
    class_id = name_to_class_id.get(class_name)
    if class_id:
        return class_id
    
    # Si aún no se encuentra, buscar de forma más flexible
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