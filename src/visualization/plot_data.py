# src/visualization/plot_data.py

import matplotlib.pyplot as plt
import numpy as np
import src.config as config 
from src.data_processing.utils import get_all_subclasses, get_class_id

def plot_data(data_counts, class_label_to_id, id_to_class, parent_to_children, name_to_class_id):

    print("\nDEBUG - Plot Data Input:")
    print(f"Hours to process: {config.SELECTED_HOURS}")
    print(f"Classes to process: {config.SELECTED_CLASSES}")
    print("\nData counts structure:")
    for hour in data_counts:
        print(f"\nHour: {hour}")
        print(f"Number of different classes: {len(data_counts[hour])}")
        total = sum(data_counts[hour].values())
        print(f"Total events: {total}")
        if total > 0:
            print("Sample of class counts:")
            for class_id, count in list(data_counts[hour].items())[:5]:
                class_name = id_to_class[class_id]['name']
                print(f"  {class_name}: {count}")

    all_hours = sorted(config.SELECTED_HOURS)
    print(f"Hours to plot: {all_hours}")

    selected_classes = config.SELECTED_CLASSES
    print(f"Selected classes to plot: {selected_classes}")

    if not all_hours or not selected_classes:
        print("\nNo data to plot.")
        return

    counts_per_hour = {hour: {} for hour in all_hours}

    for hour in all_hours:
        total_events = sum(data_counts[hour].values()) if hour in data_counts else 0
        print(f"\nTotal events for {hour}: {total_events}")
        
        for class_name in selected_classes:
            class_id = get_class_id(class_name, class_label_to_id, name_to_class_id)
            if class_id:
                all_related_ids = get_all_subclasses(class_id, parent_to_children)
                class_count = sum(data_counts[hour].get(cid, 0) for cid in all_related_ids)
                percentage = (class_count / total_events * 100) if total_events > 0 else 0
                counts_per_hour[hour][class_name] = percentage
                print(f"  {class_name}: count = {class_count}, percentage = {percentage:.2f}%")
                
                print(f"    Detailed breakdown for {class_name}:")
                for cid in all_related_ids:
                    count = data_counts[hour].get(cid, 0)
                    if count > 0:
                        print(f"      {id_to_class[cid]['name']}: {count}")
            else:
                counts_per_hour[hour][class_name] = 0
                print(f"  {class_name}: count = 0, percentage = 0.00% (Class ID not found)")


    # Generate the graph
    fig, ax = plt.subplots(figsize=(14, 8))

    num_hours = len(all_hours)
    num_classes = len(selected_classes)
    bar_width = 0.8 / num_classes
    index = np.arange(num_hours)

    for i, class_name in enumerate(selected_classes):
        class_percentages = [counts_per_hour[hour].get(class_name, 0) for hour in all_hours]
        print(f"Plotting {class_name}: {class_percentages}")
        positions = index + i * bar_width
        ax.bar(positions, class_percentages, bar_width, label=class_name)

    ax.set_xlabel('Hour')
    ax.set_ylabel('Percentage of Events (%)')
    ax.set_title(f'Sound Events per Hour (threshold: {config.CONFIDENCE_THRESHOLD_STR})')
    ax.set_xticks(index + bar_width * (num_classes - 1) / 2)
    ax.set_xticklabels([hour.split(' ')[1] for hour in all_hours], rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()