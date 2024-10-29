# src/gui/tkinter_interface.py

import os
import datetime
import traceback
import tkinter as tk
from tkinter import ttk, messagebox
from src.data_processing.process_data import load_and_process_data
from src.visualization.plot_data import plot_data
import src.config as config

# Tkinter UI for selecting variables and triggering analysis
def run_tkinter_interface():
    # Determine the available recorders based on directories
    available_recorders = [rec for rec in os.listdir(config.PREDICTIONS_ROOT_DIR)
                           if os.path.isdir(os.path.join(config.PREDICTIONS_ROOT_DIR, rec))]

    root = tk.Tk()
    root.title("Audio Analysis Tool")
    root.geometry("1200x600")  # Adjust the window size for better visibility

    # Section for user options
    options_frame = ttk.LabelFrame(root, text="Select Parameters")
    options_frame.grid(row=0, column=0, padx=10, pady=20, sticky="nsew")

    # Confidence threshold selection
    ttk.Label(options_frame, text="Confidence Threshold:").grid(row=3, column=0, sticky="w")
    threshold_var = tk.DoubleVar(value=config.CONFIDENCE_THRESHOLD)
    threshold_entry = ttk.Entry(options_frame, textvariable=threshold_var)
    threshold_entry.grid(row=3, column=1, sticky="w", padx=(0, 20))

    # Configure grid to expand
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    options_frame.columnconfigure(0, weight=1)

    # Recorders selection
    ttk.Label(options_frame, text="Recorders:").grid(row=0, column=0, sticky="w")
    recorders_var = {rec: tk.BooleanVar(value=True) for rec in available_recorders}
    for idx, (rec, var) in enumerate(recorders_var.items()):
        ttk.Checkbutton(options_frame, text=rec, variable=var).grid(row=0, column=idx + 1, sticky="w")

    # Classes selection with hierarchical display
    class_vars = {}
    row_offset = 1

    # Create a frame for classes with a scrollbar
    classes_frame = ttk.Frame(options_frame)
    classes_frame.grid(row=row_offset, column=0, columnspan=10, sticky="nsew")
    options_frame.rowconfigure(row_offset, weight=1)
    row_offset += 1

    canvas = tk.Canvas(classes_frame)
    scrollbar = ttk.Scrollbar(classes_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    # Bind the scrollbar to the canvas
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Populate the class selection checkboxes
    col = 0
    row = 0
    for class_name, sub_classes in config.CUSTOM_CATEGORIES.items():
        # Main class checkbox
        class_var = tk.BooleanVar(value=False)
        class_vars[class_name] = class_var
        ttk.Checkbutton(scrollable_frame, text=class_name, variable=class_var).grid(row=row, column=col, sticky="w")
        row += 1

        # Subclass checkboxes
        for sub_class_name in sub_classes:
            sub_class_var = tk.BooleanVar(value=False)
            class_vars[sub_class_name] = sub_class_var
            ttk.Checkbutton(scrollable_frame, text="   " + sub_class_name, variable=sub_class_var).grid(row=row, column=col, sticky="w")
            row += 1

        if row > 20:
            row = 0
            col += 1

    row_offset += 1

    # Days selection
    row_offset += 1
    ttk.Label(options_frame, text="Days:").grid(row=row_offset, column=0, sticky="w")

    # Obtain available days based on the files present
    available_days_dict = get_available_days(config.PREDICTIONS_ROOT_DIR, available_recorders)
    if not available_days_dict:
        messagebox.showinfo("No Data", "No data available for selection.")
        root.destroy()
        return
    # Display the days in the desired format
    day_vars = {display_day: tk.BooleanVar(value=True) for display_day in sorted(available_days_dict.keys())}
    for idx, (display_day, var) in enumerate(day_vars.items()):
        ttk.Checkbutton(options_frame, text=display_day, variable=var).grid(row=row_offset, column=idx + 1, sticky="w")
    row_offset += 1

    # Hours selection
    row_offset += 1
    ttk.Label(options_frame, text="Hours:").grid(row=row_offset, column=0, sticky="w")
    available_hours_dict = get_available_hours(config.PREDICTIONS_ROOT_DIR, available_recorders)
    # Display the hours in the desired format
    hour_vars = {display_hour: tk.BooleanVar(value=True) for display_hour in sorted(available_hours_dict.keys())}
    for idx, (display_hour, var) in enumerate(hour_vars.items()):
        ttk.Checkbutton(options_frame, text=display_hour, variable=var).grid(row=row_offset, column=idx + 1, sticky="w")
    row_offset += 1
    row_offset += 1

    # Button to run the analysis
    def run_analysis():
        # Get the threshold string
        threshold_str = threshold_entry.get()
        # Validate the input
        try:
            threshold_value = float(threshold_str)
            config.CONFIDENCE_THRESHOLD = threshold_value
            config.CONFIDENCE_THRESHOLD_STR = threshold_str
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number for the confidence threshold.")
            return

        print(f"\nSelected confidence threshold: {config.CONFIDENCE_THRESHOLD_STR}")

        print("\nSelected classes and subclasses:")
        for class_name, var in class_vars.items():
            if var.get():
                print(f"- {class_name}")
                if class_name in config.CUSTOM_CATEGORIES:
                    for subclass in config.CUSTOM_CATEGORIES[class_name]:
                        if class_vars[subclass].get():
                            print(f"  - {subclass}")

        # Ensure at least one element is selected in each category
        if not any(var.get() for var in recorders_var.values()):
            messagebox.showwarning("Selection Error", "Please select at least one recorder.")
            return
        if not any(var.get() for var in day_vars.values()):
            messagebox.showwarning("Selection Error", "Please select at least one day.")
            return
        if not any(var.get() for var in hour_vars.values()):
            messagebox.showwarning("Selection Error", "Please select at least one hour.")
            return
        if not any(var.get() for var in class_vars.values()):
            messagebox.showwarning("Selection Error", "Please select at least one class.")
            return

        # Update selected variables in config
        config.SELECTED_RECORDERS = [rec for rec, var in recorders_var.items() if var.get()]
        config.SELECTED_CLASSES = [cls for cls, var in class_vars.items() if var.get()]

        # Map selected display days back to processing formats
        selected_display_days = [display_day for display_day, var in day_vars.items() if var.get()]
        config.SELECTED_DAYS = [available_days_dict[display_day] for display_day in selected_display_days]

        # Map selected display hours back to processing formats
        selected_display_hours = [display_hour for display_hour, var in hour_vars.items() if var.get()]
        config.SELECTED_HOURS = [available_hours_dict[display_hour] for display_hour in selected_display_hours]

        # Execute data processing
        try:
            data_counts, class_label_to_id, id_to_class, parent_to_children, name_to_class_id = load_and_process_data()
            if data_counts:
                plot_data(data_counts, class_label_to_id, id_to_class, parent_to_children, name_to_class_id)
            else:
                messagebox.showinfo("No Data", "No data available to generate the plot.")
        except Exception as e:
            # Print the full traceback to the console
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred during data processing:\n{e}")

    ttk.Button(root, text="Run Analysis", command=run_analysis).grid(row=row_offset, column=0, padx=10, pady=20)

    root.mainloop()

def extract_datetime_from_filename(filename):
    # Remove file extension
    filename_no_ext, ext = os.path.splitext(filename)
    # Remove '_light' termination if present
    if filename_no_ext.endswith('_light'):
        filename_no_ext = filename_no_ext[:-6]
    # Extract datetime
    try:
        file_datetime = datetime.datetime.strptime(filename_no_ext, '%Y%m%d_%H%M%S')
    except ValueError as ve:
        print(f"Error parsing date from file {filename}: {ve}")
        return None
    return file_datetime


def get_available_days(predictions_root_dir, recorders):
    available_days = {}
    for recorder in recorders:
        recorder_dir = os.path.join(predictions_root_dir, recorder)
        days_set = set()
        for filename in os.listdir(recorder_dir):
            if filename.endswith('.json'):
                file_datetime = extract_datetime_from_filename(filename)
                if file_datetime:
                    date_str = file_datetime.strftime('%Y%m%d')
                    days_set.add(date_str)
        available_days[recorder] = sorted(days_set)
    return available_days

def get_available_hours(predictions_root_dir, recorders):
    available_hours = {}
    for recorder in recorders:
        recorder_dir = os.path.join(predictions_root_dir, recorder)
        hours_set = set()
        for filename in os.listdir(recorder_dir):
            if filename.endswith('.json'):
                file_datetime = extract_datetime_from_filename(filename)
                if file_datetime:
                    hour_str = file_datetime.strftime('%H')
                    hours_set.add(hour_str)
        available_hours[recorder] = sorted(hours_set)
    return available_hours

