# scripts/data_preparation_granger.py

import os
import json
import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import src.config as config
from src.data_processing.utils import extract_datetime_from_filename
from src.data_processing.load_data import load_class_labels
from src.data_processing.process_data import compute_class_thresholds

def load_prepared_data(aggregation_level, normalize=False):
    """
    Carga los datos de los archivos intermedios y agrega según el nivel especificado.

    Args:
        aggregation_level (str): 'all', 'by_day', 'by_recorder'
        normalize (bool): Determina si se normalizan los datos cuando hay grabadores sin muestras.

    Returns:
        pd.DataFrame: Datos agregados
    """
    import pandas as pd
    from src.data_processing.load_data import load_class_labels

    # Cargar todas las clases (527)
    class_label_to_id, class_id_to_label = load_class_labels(config.CLASS_LABELS_CSV_PATH)
    all_class_ids = set(class_id_to_label.keys())

    data_frames = []
    recorders_dir = os.path.join('analysis_results', 'data_preparation_granger_results')
    recorder_folders = [f for f in os.listdir(recorders_dir) if os.path.isdir(os.path.join(recorders_dir, f))]
    total_recorders = len(recorder_folders)

    if total_recorders == 0:
        print("No se encontraron datos en los archivos intermedios.")
        return pd.DataFrame()

    for recorder_folder in recorder_folders:
        recorder_path = os.path.join(recorders_dir, recorder_folder)
        minute_counts_path = os.path.join(recorder_path, 'minute_counts.json')
        if os.path.exists(minute_counts_path):
            with open(minute_counts_path, 'r') as f:
                minute_counts = json.load(f)

            df = pd.DataFrame.from_dict(minute_counts, orient='index')
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            df['recorder'] = recorder_folder
            data_frames.append(df)
        else:
            print(f"No se encontró 'minute_counts.json' en {recorder_path}")

    if not data_frames:
        print("No se pudieron cargar datos de los grabadores.")
        return pd.DataFrame()

    # Concatenate data frames
    full_df = pd.concat(data_frames, axis=0, sort=True)

    # Convert class IDs to columns, fill missing classes with zeros
    for class_id in all_class_ids:
        if class_id not in full_df.columns:
            full_df[class_id] = 0

    if aggregation_level == 'all':
        # Aggregate over all recorders and days
        grouped = full_df.groupby(full_df.index).sum()
    elif aggregation_level == 'by_day':
        # Keep days independent
        full_df['date'] = full_df.index.date
        grouped = full_df.groupby(['date', full_df.index.time]).sum()
    elif aggregation_level == 'by_recorder':
        # Keep recorders and days independent
        grouped = full_df
    else:
        raise ValueError("Invalid aggregation level")

    if normalize:
        # Normalization process
        # For each time bin, adjust counts based on the number of active recorders
        grouped = grouped.reset_index()
        if 'recorder' in grouped.columns:
            # When aggregation_level is 'by_recorder', we need to group further
            time_bins = grouped['index'].unique()
            normalized_data = []

            for time_bin in time_bins:
                time_bin_data = grouped[grouped['index'] == time_bin]
                num_active_recorders = time_bin_data['recorder'].nunique()
                norm_factor = total_recorders / num_active_recorders if num_active_recorders > 0 else 1

                # Apply normalization to each class column
                class_columns = [col for col in time_bin_data.columns if col.startswith('/m/')]
                time_bin_data[class_columns] = time_bin_data[class_columns] * norm_factor

                normalized_data.append(time_bin_data)

            grouped = pd.concat(normalized_data, axis=0)
            grouped.set_index('index', inplace=True)
        else:
            # For other aggregation levels
            time_bins = grouped.index.unique()
            normalized_data = []

            for time_bin in time_bins:
                time_bin_data = grouped.loc[time_bin]
                num_active_recorders = len(recorder_folders)
                norm_factor = total_recorders / num_active_recorders if num_active_recorders > 0 else 1

                # Apply normalization
                class_columns = [col for col in grouped.columns if col.startswith('/m/')]
                time_bin_data[class_columns] = time_bin_data[class_columns] * norm_factor

                normalized_data.append(time_bin_data)

            grouped = pd.DataFrame(normalized_data, index=time_bins)

    else:
        # Fill missing values with zero (assuming no events occurred)
        grouped = grouped.fillna(0)

    # Optionally, convert class IDs to class names
    # class_id_to_label = {v: k for k, v in class_label_to_id.items()}
    # grouped.rename(columns=class_id_to_label, inplace=True)

    return grouped


def process_json_file(file_path, class_thresholds, class_label_to_id):
    """
    Procesa un archivo JSON de predicciones y devuelve los conteos por minuto.
    """
    minute_counts = defaultdict(lambda: defaultdict(int))
    try:
        filename = os.path.basename(file_path).replace('_light.json', '').replace('.json', '')
        file_datetime = datetime.datetime.strptime(filename, '%Y%m%d_%H%M%S')
        with open(file_path, 'r') as f:
            data = json.load(f)

        is_light = '_light.json' in file_path
        increment = 32 if is_light else 1

        for frame in data:
            if isinstance(frame, dict) and ('predictions' in frame):
                frame_time = file_datetime + datetime.timedelta(seconds=float(frame.get('time', 0)))
                minute_key = frame_time.strftime('%Y-%m-%dT%H:%M:00')  # Mantiene fecha y hora

                predictions = frame.get('predictions', [])
                for pred in predictions:
                    class_label = pred.get('class_label') or pred.get('class')
                    score = pred.get('probability') or pred.get('prob')

                    if class_label:
                        class_label_normalized = class_label.strip().lower()
                        threshold = class_thresholds.get(class_label_normalized, config.CONFIDENCE_THRESHOLD)
                        if score >= threshold:
                            class_id = class_label_to_id.get(class_label.lower())
                            if class_id:
                                minute_counts[minute_key][class_id] += increment

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        raise

    return minute_counts

def process_recorder(recorder_id, class_thresholds, class_label_to_id):
    """
    Procesa los archivos de predicciones para un grabador específico.
    Guarda los conteos por minuto en la carpeta del grabador.
    """
    recorder_dir = os.path.join(config.PREDICTIONS_ROOT_DIR, recorder_id)
    output_dir = os.path.join('analysis_results', 'data_preparation_granger_results', recorder_id)
    os.makedirs(output_dir, exist_ok=True)

    minute_counts = defaultdict(lambda: defaultdict(int))

    for filename in os.listdir(recorder_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(recorder_dir, filename)
            try:
                file_counts = process_json_file(file_path, class_thresholds, class_label_to_id)
                # Merge counts
                for minute, counts in file_counts.items():
                    for class_id, count in counts.items():
                        minute_counts[minute][class_id] += count
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Save minute_counts to JSON
    output_path = os.path.join(output_dir, 'minute_counts.json')
    with open(output_path, 'w') as f:
        json.dump(minute_counts, f)

def main():
    # Load class labels (all 527 classes)
    print("Cargando etiquetas de clase...")
    class_label_to_id, class_id_to_label = load_class_labels(config.CLASS_LABELS_CSV_PATH)

    # Calcular umbrales de clase si se utilizan umbrales variables
    if config.USE_LABEL_QUALITY_THRESHOLDS:
        class_thresholds = compute_class_thresholds(class_label_to_id, class_id_to_label)
    else:
        CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD
        class_thresholds = {label: CONFIDENCE_THRESHOLD for label in class_label_to_id.keys()}

    # Get list of recorders
    recorders = [rec for rec in os.listdir(config.PREDICTIONS_ROOT_DIR)
                 if os.path.isdir(os.path.join(config.PREDICTIONS_ROOT_DIR, rec))]

    if not recorders:
        print("No se encontraron grabadores en el directorio de predicciones.")
        return

    # Process each recorder
    print("Procesando grabadores...")
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_recorder, rec, class_thresholds, class_label_to_id): rec
                   for rec in recorders}
        for future in tqdm(as_completed(futures), total=len(futures)):
            rec = futures[future]
            try:
                future.result()
                print(f"Procesado el grabador {rec}")
            except Exception as e:
                print(f"Error al procesar el grabador {rec}: {e}")

if __name__ == "__main__":
    main()
