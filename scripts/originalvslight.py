# scripts/originalvslight.py

import os
import json
from datetime import datetime
import numpy as np

# Rutas a los directorios
LIGHT_DIR = 'assets/sample_data_light'
ORIGINAL_DIR = 'assets/sample_data'

def analyze_json_file(file_path):
    """
    Analiza un archivo JSON y retorna estadísticas sobre sus frames y predicciones
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    total_frames = 0
    frame_indices = []
    predictions_per_frame = []
    unique_classes = set()
    frame_times = []

    # Analizar cada frame
    for item in data:
        if isinstance(item, dict) and 'predictions' in item:
            total_frames += 1
            frame_indices.append(item.get('frame_index'))
            frame_times.append(float(item.get('time', 0)))
            
            predictions = item.get('predictions', [])
            predictions_per_frame.append(len(predictions))
            
            # Recolectar clases únicas y sus probabilidades
            for pred in predictions:
                class_name = pred.get('class_label') or pred.get('class')
                unique_classes.add(class_name)

    class_probabilities = {}  # Para almacenar probabilidades por clase
    frames_with_class = {}    # Para contar en cuántos frames aparece cada clase
    
    for item in data:
        if isinstance(item, dict) and 'predictions' in item:
            frame_predictions = item.get('predictions', [])
            
            for pred in frame_predictions:
                class_name = pred.get('class_label') or pred.get('class')
                prob = float(pred.get('probability') or pred.get('prob', 0))
                
                if class_name not in class_probabilities:
                    class_probabilities[class_name] = []
                    frames_with_class[class_name] = 0
                    
                class_probabilities[class_name].append(prob)
                frames_with_class[class_name] += 1

    # Calcular estadísticas por clase
    class_stats = {}
    for class_name in class_probabilities:
        probs = class_probabilities[class_name]
        class_stats[class_name] = {
            'mean_prob': np.mean(probs),
            'frame_count': frames_with_class[class_name],
            'frame_percentage': (frames_with_class[class_name] / total_frames) * 100
        }

    return {
        'total_frames': total_frames,
        'frame_indices': frame_indices,
        'frame_index_diffs': np.diff(frame_indices) if frame_indices else [],
        'predictions_per_frame': predictions_per_frame,
        'unique_classes': len(unique_classes),
        'time_diffs': np.diff(frame_times) if frame_times else [],
        'class_stats': class_stats
    }

def compare_files(original_path, light_path):
    """
    Compara un archivo original con su versión light
    """
    print(f"\nComparando archivos:")
    print(f"Original: {os.path.basename(original_path)}")
    print(f"Light: {os.path.basename(light_path)}")
    
    original_stats = analyze_json_file(original_path)
    light_stats = analyze_json_file(light_path)
    
    print("\nEstadísticas básicas:")
    print(f"Total frames - Original: {original_stats['total_frames']}, Light: {light_stats['total_frames']}")
    print(f"Clases únicas - Original: {original_stats['unique_classes']}, Light: {light_stats['unique_classes']}")
    
    print("\nÍndices de frames:")
    print("Original - primeros 5:", original_stats['frame_indices'][:5])
    print("Light - primeros 5:", light_stats['frame_indices'][:5])
    
    if len(original_stats['frame_index_diffs']) > 0:
        print("\nDiferencias entre índices consecutivos:")
        print("Original - primeras 5:", original_stats['frame_index_diffs'][:5])
        print("Light - primeras 5:", light_stats['frame_index_diffs'][:5])
    
    print("\nTiempos entre frames consecutivos:")
    print("Original - primeros 5:", original_stats['time_diffs'][:5])
    print("Light - primeros 5:", light_stats['time_diffs'][:5])
    
    print("\nPredicciones por frame:")
    print("Original - media:", np.mean(original_stats['predictions_per_frame']))
    print("Light - media:", np.mean(light_stats['predictions_per_frame']))
    
    # Análisis de predicciones
    print("\nAnálisis detallado de clases:")
    orig_class_stats = original_stats['class_stats']
    light_class_stats = light_stats['class_stats']
    
    # Encontrar clases únicas en cada versión
    original_classes = set(orig_class_stats.keys())
    light_classes = set(light_class_stats.keys())
    
    only_in_original = original_classes - light_classes
    only_in_light = light_classes - original_classes
    common_classes = original_classes & light_classes
    
    print(f"\nDistribución de clases:")
    print(f"Total clases en original: {len(original_classes)}")
    print(f"Total clases en light: {len(light_classes)}")
    print(f"Clases solo en original: {len(only_in_original)}")
    print(f"Clases solo en light: {len(only_in_light)}")
    print(f"Clases comunes: {len(common_classes)}")
    
    # Analizar algunas discrepancias específicas
    if len(only_in_original) > 0:
        print("\nEjemplos de clases solo en original (top 5):")
        for class_name in sorted(only_in_original)[:5]:
            stats = orig_class_stats[class_name]
            print(f"- {class_name}: aparece en {stats['frame_count']} frames ({stats['frame_percentage']:.2f}%), "
                  f"prob media = {stats['mean_prob']:.3f}")
    
    if len(only_in_light) > 0:
        print("\nEjemplos de clases solo en light (top 5):")
        for class_name in sorted(only_in_light)[:5]:
            stats = light_class_stats[class_name]
            print(f"- {class_name}: aparece en {stats['frame_count']} frames ({stats['frame_percentage']:.2f}%), "
                  f"prob media = {stats['mean_prob']:.3f}")
    
    # Analizar clases comunes con mayor diferencia en la proporción de frames
    print("\nAnálisis de clases comunes (top 5 con mayores diferencias en proporción):")
    ratios = []
    for class_name in common_classes:
        orig_ratio = orig_class_stats[class_name]['frame_percentage']
        light_ratio = light_class_stats[class_name]['frame_percentage']
        diff = abs(orig_ratio - light_ratio)
        ratios.append((diff, class_name))
    
    for diff, class_name in sorted(ratios, reverse=True)[:5]:
        orig = orig_class_stats[class_name]
        light = light_class_stats[class_name]
        print(f"\nClase: {class_name}")
        print(f"Original: {orig['frame_count']} frames ({orig['frame_percentage']:.2f}%), prob = {orig['mean_prob']:.3f}")
        print(f"Light: {light['frame_count']} frames ({light['frame_percentage']:.2f}%), prob = {light['mean_prob']:.3f}")
        print(f"Diferencia en porcentaje: {diff:.2f}%")

    return original_stats, light_stats

def main():
    # Obtener lista de grabadores
    recorders = [d for d in os.listdir(LIGHT_DIR) if os.path.isdir(os.path.join(LIGHT_DIR, d))]
    
    print(f"Grabadores encontrados: {recorders}")
    
    for recorder in recorders:
        light_recorder_dir = os.path.join(LIGHT_DIR, recorder)
        original_recorder_dir = os.path.join(ORIGINAL_DIR, recorder)
        
        # Obtener archivos light
        light_files = [f for f in os.listdir(light_recorder_dir) if f.endswith('_light.json')]
        
        for light_file in light_files:
            # Encontrar el archivo original correspondiente
            original_file = light_file.replace('_light.json', '.json')
            light_path = os.path.join(light_recorder_dir, light_file)
            original_path = os.path.join(original_recorder_dir, original_file)
            
            if os.path.exists(original_path):
                compare_files(original_path, light_path)
            else:
                print(f"Archivo original no encontrado para {light_file}")

if __name__ == "__main__":
    main()