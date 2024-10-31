# src/gui/tkinter_interface.py

import os
import tkinter as tk
from tkinter import ttk, messagebox
import src.config as config
from scripts.plot_results import plot_results  # Ensure this function exists and is correctly imported
import json

def run_tkinter_interface():
    # List of all 14 recorders
    all_recorders = [f"{i:02d}" for i in range(1, 15)]
    KITCHEN_RECORDERS = ['02', '04', '05', '07', '10', '11', '13']
    LIVING_RECORDERS = ['01', '03', '06', '08', '09', '12', '14']

    root = tk.Tk()
    root.title("Audio Analysis Tool")
    root.geometry("800x700")  # Adjust window size as needed

    # Style configuration
    style = ttk.Style(root)
    style.configure('TButton', background='white', foreground='black')
    style.map('TButton',
              background=[('active', 'white')],
              foreground=[('active', 'black')])

    style.configure('Selected.TButton', background='blue', foreground='white')

    # Section for user options
    options_frame = ttk.LabelFrame(root, text="Select Parameters")
    options_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Recorders selection
    recorder_frame = ttk.Frame(options_frame)
    recorder_frame.pack(fill="x", pady=10)

    ttk.Label(recorder_frame, text="Recorders:").grid(row=0, column=0, sticky="w")

    # Kitchen and Living room options
    kitchen_var = tk.BooleanVar()
    living_var = tk.BooleanVar()

    recorder_vars = {rec: tk.BooleanVar(value=False) for rec in all_recorders}
    recorder_checkboxes = {}

    def update_recorder_selection():
        # Handle Kitchen selection
        if kitchen_var.get():
            for rec in KITCHEN_RECORDERS:
                recorder_vars[rec].set(True)
                recorder_checkboxes[rec]['state'] = 'disabled'
        else:
            for rec in KITCHEN_RECORDERS:
                recorder_vars[rec].set(False)
                recorder_checkboxes[rec]['state'] = 'normal'

        # Handle Living Room selection
        if living_var.get():
            for rec in LIVING_RECORDERS:
                recorder_vars[rec].set(True)
                recorder_checkboxes[rec]['state'] = 'disabled'
        else:
            for rec in LIVING_RECORDERS:
                recorder_vars[rec].set(False)
                recorder_checkboxes[rec]['state'] = 'normal'

    # Kitchen and Living Room checkbuttons
    ttk.Checkbutton(recorder_frame, text="Kitchen", variable=kitchen_var, command=update_recorder_selection).grid(row=1, column=0, sticky="w")
    ttk.Checkbutton(recorder_frame, text="Living Room", variable=living_var, command=update_recorder_selection).grid(row=2, column=0, sticky="w")

    # Individual recorder checkbuttons
    for idx, rec in enumerate(all_recorders):
        var = recorder_vars[rec]
        cb = ttk.Checkbutton(recorder_frame, text=rec, variable=var)
        row = idx // 7 + 1
        col = idx % 7 + 1
        cb.grid(row=row, column=col, sticky="w")
        recorder_checkboxes[rec] = cb

    # Threshold selection
    threshold_frame = ttk.Frame(options_frame)
    threshold_frame.pack(fill="x", pady=10)
    ttk.Label(threshold_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky="w")

    # Threshold options
    threshold_options = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 'variable']
    threshold_var = tk.StringVar(value='')  # No threshold selected by default

    def select_threshold(value):
        threshold_var.set(str(value))
        # Update button styles
        for btn in threshold_buttons:
            if btn['text'] == str(value):
                btn.configure(style='Selected.TButton')
            else:
                btn.configure(style='TButton')

    threshold_buttons = []
    for idx, option in enumerate(threshold_options):
        btn = ttk.Button(
            threshold_frame,
            text=str(option),
            command=lambda opt=option: select_threshold(opt),
            style='TButton'
        )
        row = idx // 6 + 1
        col = idx % 6 + 1
        btn.grid(row=row, column=col, padx=5, pady=5, sticky="w")
        threshold_buttons.append(btn)

    # Classes selection
    class_vars = {}
    class_frame = ttk.Frame(options_frame)
    class_frame.pack(fill="both", expand=True, pady=10)

    ttk.Label(class_frame, text="Classes:").grid(row=0, column=0, sticky="w")

    # Create a frame for classes with a scrollbar
    classes_canvas = tk.Canvas(class_frame)
    classes_scrollbar = ttk.Scrollbar(class_frame, orient="vertical", command=classes_canvas.yview)
    classes_scrollable_frame = ttk.Frame(classes_canvas)

    # Bind the scrollbar to the canvas
    classes_scrollable_frame.bind(
        "<Configure>",
        lambda e: classes_canvas.configure(
            scrollregion=classes_canvas.bbox("all")
        )
    )

    classes_canvas.create_window((0, 0), window=classes_scrollable_frame, anchor="nw")
    classes_canvas.configure(yscrollcommand=classes_scrollbar.set)

    classes_canvas.grid(row=1, column=0, sticky="nsew")
    classes_scrollbar.grid(row=1, column=1, sticky="ns")

    # Configure grid weights
    class_frame.columnconfigure(0, weight=1)
    class_frame.rowconfigure(1, weight=1)

    # Populate the class selection checkboxes in multiple columns
    col = 0
    row = 0
    max_rows = 20  # Adjust as needed
    for class_name, sub_classes in config.CUSTOM_CATEGORIES.items():
        # Main class checkbox
        class_var = tk.BooleanVar(value=False)
        class_vars[class_name] = class_var
        ttk.Checkbutton(classes_scrollable_frame, text=class_name, variable=class_var).grid(row=row, column=col, sticky="w")
        row += 1

        # Subclass checkboxes
        for sub_class_name in sub_classes:
            sub_class_var = tk.BooleanVar(value=False)
            class_vars[sub_class_name] = sub_class_var
            ttk.Checkbutton(classes_scrollable_frame, text="   " + sub_class_name, variable=sub_class_var).grid(row=row, column=col, sticky="w")
            row += 1

        if row >= max_rows:
            row = 0
            col += 1

    # Button to generate the plot
    def generate_plot():
        selected_threshold = threshold_var.get()
        if selected_threshold == '':
            messagebox.showwarning("Selection Error", "Please select a confidence threshold.")
            return

        # Ensure at least one recorder is selected
        if not any(var.get() for var in recorder_vars.values()):
            messagebox.showwarning("Selection Error", "Please select at least one recorder.")
            return

        # Ensure at least one class is selected
        if not any(var.get() for var in class_vars.values()):
            messagebox.showwarning("Selection Error", "Please select at least one class.")
            return

        # Update selected variables in config
        config.SELECTED_RECORDERS = [rec for rec, var in recorder_vars.items() if var.get()]
        config.SELECTED_CLASSES = [cls for cls, var in class_vars.items() if var.get()]

        # Determine if recorders are from Kitchen or Living Room
        if kitchen_var.get() and not living_var.get():
            recorder_info = 'Kitchen recorders'
        elif living_var.get() and not kitchen_var.get():
            recorder_info = 'Living Room recorders'
        elif kitchen_var.get() and living_var.get():
            recorder_info = 'Kitchen and Living Room recorders'
        else:
            # List the selected recorders
            recorder_info = 'Selected recorders: ' + ', '.join(config.SELECTED_RECORDERS)

        # Load data from the precomputed JSON files
        threshold_str = selected_threshold
        if threshold_str == 'variable':
            input_file = os.path.join('analysis_results/batch_analysis_results', f'analysis_results_threshold_variable.json')
        else:
            input_file = os.path.join('analysis_results/batch_analysis_results', f'analysis_results_threshold_{threshold_str}.json')

        if not os.path.exists(input_file):
            messagebox.showerror("File Not Found", f"The analysis results file '{input_file}' does not exist.")
            return

        with open(input_file, 'r') as f:
            data = json.load(f)

        data_counts = data['data_counts']

        # Call the plotting function from plot_results.py
        try:
            plot_results(data_counts, config.SELECTED_RECORDERS, config.SELECTED_CLASSES, threshold_str, recorder_info)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while generating the plot:\n{e}")

    ttk.Button(root, text="Plot", command=generate_plot).pack(pady=20)

    root.mainloop()
