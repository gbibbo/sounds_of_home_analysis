# src/gui/tkinter_interface.py

import os
import tkinter as tk
from tkinter import ttk, messagebox
import src.config as config
from scripts.plot_results import plot_results
import json
from scripts.advanced_analysis import run_advanced_analysis

def run_tkinter_interface():
    """
    Main function to run the audio analysis graphical interface.
    Provides options for selecting recorders, thresholds, and sound classes.
    """
    # Initialize recorder configurations
    all_recorders = [f"{i:02d}" for i in range(1, 15)]
    KITCHEN_RECORDERS = ['02', '04', '05', '07', '10', '11', '13']
    LIVING_RECORDERS = ['01', '03', '06', '08', '09', '12', '14']

    # Create main window
    root = tk.Tk()
    root.title("Audio Analysis Tool")
    root.geometry("1100x877")

    # Configure styles
    style = ttk.Style(root)
    style.configure('TButton', background='white', foreground='black')
    style.map('TButton',
              background=[('active', 'white')],
              foreground=[('active', 'black')])
    style.configure('Selected.TButton', background='blue', foreground='white')
    style.configure('Muted.TButton', padding=10, width=10, anchor='center')
    style.map('Muted.TButton',
          background=[('selected', '#4a6cd4'), ('!selected', '#e1e1e1')],
          foreground=[('selected', 'white'), ('!selected', 'black')])

    # Create main options frame
    options_frame = ttk.LabelFrame(root, text="Select Parameters")
    options_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Recorder selection section
    recorder_frame = ttk.Frame(options_frame)
    recorder_frame.pack(fill="x", pady=10)
    ttk.Label(recorder_frame, text="Recorders:").grid(row=0, column=0, sticky="w")

    # Initialize variables for recorder selection
    kitchen_var = tk.BooleanVar()
    living_var = tk.BooleanVar()
    recorder_vars = {rec: tk.BooleanVar(value=False) for rec in all_recorders}
    recorder_checkboxes = {}

    # Days selection variables
    all_days_var = tk.BooleanVar(value=True)  # "All Days" selected by default
    day_vars = []
    num_days = 7  # Number of days in the study

    def update_recorder_selection():
        """Update recorder checkboxes based on Kitchen/Living Room selection"""
        for rec in KITCHEN_RECORDERS:
            recorder_vars[rec].set(kitchen_var.get())
            recorder_checkboxes[rec]['state'] = 'disabled' if kitchen_var.get() else 'normal'
        for rec in LIVING_RECORDERS:
            recorder_vars[rec].set(living_var.get())
            recorder_checkboxes[rec]['state'] = 'disabled' if living_var.get() else 'normal'

    def update_day_selection():
        """Update day checkboxes based on 'All Days' selection"""
        if all_days_var.get():
            # Disable individual day checkboxes and deselect them
            for var, cb in zip(day_vars, day_checkboxes):
                var.set(False)
                cb['state'] = 'disabled'
        else:
            # Enable individual day checkboxes
            for cb in day_checkboxes:
                cb['state'] = 'normal'

    def select_day(index):
        """Ensure only one day is selected at a time"""
        # Deselect other days
        for i, var in enumerate(day_vars):
            if i != index:
                var.set(False)
    # Add Kitchen and Living Room checkbuttons
    ttk.Checkbutton(recorder_frame, text="Kitchen", variable=kitchen_var, 
                    command=update_recorder_selection).grid(row=1, column=0, sticky="w")
    ttk.Checkbutton(recorder_frame, text="Living Room", variable=living_var, 
                    command=update_recorder_selection).grid(row=2, column=0, sticky="w")

    # Add individual recorder checkbuttons
    for idx, rec in enumerate(all_recorders):
        var = recorder_vars[rec]
        cb = ttk.Checkbutton(recorder_frame, text=rec, variable=var)
        row = idx // 7 + 1
        col = idx % 7 + 1
        cb.grid(row=row, column=col, sticky="w")
        recorder_checkboxes[rec] = cb

    # Add muted speech toggle button in recorder frame
    muted_speech_var = tk.BooleanVar(value=False)
    muted_speech_btn = ttk.Checkbutton(
        recorder_frame,
        text="Muted\nSpeech",
        style='Muted.TButton',
        variable=muted_speech_var,
        command=lambda: style.configure(
            'Muted.TButton',
            background='#4a6cd4' if muted_speech_var.get() else '#e1e1e1'
        )
    )
    muted_speech_btn.grid(row=1, column=8, rowspan=2, padx=(500,0), sticky="nsew")

    # Days selection section
    days_frame = ttk.Frame(options_frame)
    days_frame.pack(fill="x", pady=10)
    ttk.Label(days_frame, text="Days:").grid(row=0, column=0, sticky="w")

    # "All Days" checkbox
    ttk.Checkbutton(
        days_frame,
        text="All Days",
        variable=all_days_var,
        command=update_day_selection
    ).grid(row=1, column=0, sticky="w")

    day_checkboxes = []

    for idx in range(num_days):
        var = tk.BooleanVar(value=False)
        day_vars.append(var)
        cb = ttk.Checkbutton(
            days_frame,
            text=f"Day {idx+1}",
            variable=var,
            command=lambda idx=idx: select_day(idx)
        )
        cb.grid(row=1, column=idx+1, sticky="w")
        day_checkboxes.append(cb)

    # Initialize day selection state
    update_day_selection()

    # Threshold selection section
    threshold_frame = ttk.Frame(options_frame)
    threshold_frame.pack(fill="x", pady=10)
    ttk.Label(threshold_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky="w")

    threshold_options = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 'variable']
    threshold_var = tk.StringVar(value='')

    def select_threshold(value):
        """Handle threshold button selection and update button styles"""
        threshold_var.set(str(value))
        for btn in threshold_buttons:
            btn.configure(style='Selected.TButton' if btn['text'] == str(value) else 'TButton')

    # Add threshold selection buttons
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

    # Normalization option
    normalize_frame = ttk.Frame(options_frame)
    normalize_frame.pack(fill="x", pady=10)
    normalize_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        normalize_frame,
        text="Normalize by active recorders",
        variable=normalize_var
    ).pack(side="left", padx=5)

    # Class selection section with improved layout
    class_vars = {}
    class_frame = ttk.Frame(options_frame)
    class_frame.pack(fill="both", expand=True, pady=10)
    ttk.Label(class_frame, text="Classes:").grid(row=0, column=0, sticky="w")

    # Create scrollable frame with horizontal scroll support
    classes_canvas = tk.Canvas(class_frame)
    v_scrollbar = ttk.Scrollbar(class_frame, orient="vertical", command=classes_canvas.yview)
    h_scrollbar = ttk.Scrollbar(class_frame, orient="horizontal", command=classes_canvas.xview)
    classes_scrollable_frame = ttk.Frame(classes_canvas)

    classes_scrollable_frame.bind(
        "<Configure>",
        lambda e: classes_canvas.configure(scrollregion=classes_canvas.bbox("all"))
    )

    classes_canvas.create_window((0, 0), window=classes_scrollable_frame, anchor="nw")
    classes_canvas.configure(yscrollcommand=v_scrollbar.set,
                           xscrollcommand=h_scrollbar.set)

    # Configure grid weights and scrollbars
    class_frame.columnconfigure(0, weight=1)
    class_frame.rowconfigure(1, weight=1)
    
    classes_canvas.grid(row=1, column=0, sticky="nsew")
    v_scrollbar.grid(row=1, column=1, sticky="ns")
    h_scrollbar.grid(row=2, column=0, sticky="ew")

    def add_class_checkbuttons(parent_frame, items, indent_level=0, row=0, col=0):
        """
        Recursively add checkbuttons for sound classes with proper hierarchy.
        
        Args:
            parent_frame: Frame to add checkbuttons to
            items: List of classes or nested dictionaries
            indent_level: Current indentation level
            row: Current row in the grid
            col: Current column in the grid
        
        Returns:
            tuple: (next_row, current_col) for positioning subsequent items
        """
        current_row = row
        current_col = col
        max_rows = 26  # Maximum rows before starting a new column
        
        for item in items:
            if isinstance(item, dict):
                # Handle nested categories
                for category_name, subcategories in item.items():
                    # Add parent category checkbox
                    category_var = tk.BooleanVar(value=False)
                    class_vars[category_name] = category_var
                    
                    ttk.Checkbutton(parent_frame, 
                                  text="   " * indent_level + category_name,
                                  variable=category_var).grid(row=current_row, 
                                                            column=current_col, 
                                                            sticky="w",
                                                            padx=(5 + indent_level * 20, 10))
                    current_row += 1
                    
                    # Add subcategories
                    next_row, next_col = add_class_checkbuttons(
                        parent_frame, 
                        subcategories, 
                        indent_level + 1, 
                        current_row, 
                        current_col
                    )
                    
                    current_row = next_row
                    current_col = next_col

                    # Check if we need to start a new column
                    if current_row >= max_rows:
                        current_row = 0
                        current_col += 1
            else:
                # Add regular class checkbox
                class_var = tk.BooleanVar(value=False)
                class_vars[item] = class_var
                ttk.Checkbutton(parent_frame, 
                              text="   " * indent_level + item,
                              variable=class_var).grid(row=current_row, 
                                                     column=current_col, 
                                                     sticky="w",
                                                     padx=(5 + indent_level * 20, 10))
                current_row += 1
                
                # Start a new column if we exceed max_rows
                if current_row >= max_rows:
                    current_row = 0
                    current_col += 1
        
        return current_row, current_col

    # Populate class selection area with improved layout
    current_row = 0
    current_col = 0

    # Add main categories and their subcategories
    for category_name, subcategories in config.CUSTOM_CATEGORIES_INTERFACE.items():
        category_var = tk.BooleanVar(value=False)
        class_vars[category_name] = category_var
        
        ttk.Checkbutton(classes_scrollable_frame, 
                       text=category_name,
                       variable=category_var).grid(row=current_row, 
                                                 column=current_col, 
                                                 sticky="w",
                                                 padx=5)
        current_row += 1
        
        # Add subcategories
        next_row, next_col = add_class_checkbuttons(
            classes_scrollable_frame, 
            subcategories, 
            1, 
            current_row, 
            current_col
        )
        
        current_row = next_row
        current_col = next_col
        
        # Add spacing between main categories
        current_row += 1
        
        # Start a new column if needed
        if current_row >= 15:
            current_row = 0
            current_col += 1

    def generate_plot(run_analysis=True):
        """Generate plot and optionally run advanced analysis based on selected parameters"""
        # Validate selections
        if not threshold_var.get():
            messagebox.showwarning("Selection Error", "Please select a confidence threshold.")
            return
        if not any(var.get() for var in recorder_vars.values()):
            messagebox.showwarning("Selection Error", "Please select at least one recorder.")
            return
        if not any(var.get() for var in class_vars.values()):
            messagebox.showwarning("Selection Error", "Please select at least one class.")
            return
        
        # Validate day selection
        if not all_days_var.get() and not any(var.get() for var in day_vars):
            messagebox.showwarning("Selection Error", "Please select at least one day.")
            return

        # Determine selected day(s)
        if all_days_var.get():
            results_subdir = 'total'
            day_info = 'All Days'
        else:
            selected_day_index = None
            for i, var in enumerate(day_vars):
                if var.get():
                    selected_day_index = i
                    break
            if selected_day_index is None:
                messagebox.showwarning("Selection Error", "Please select at least one day.")
                return
            day_str = f'{selected_day_index + 1:02d}'  # Format day as '01', '02', etc.
            results_subdir = day_str
            day_info = f'Day {selected_day_index + 1}'

        # Update configurations
        config.SELECTED_RECORDERS = [rec for rec, var in recorder_vars.items() if var.get()]
        config.SELECTED_CLASSES = [cls for cls, var in class_vars.items() if var.get()]

        # Determine recorder info string
        if kitchen_var.get() and not living_var.get():
            recorder_info = 'Kitchen recorders'
        elif living_var.get() and not kitchen_var.get():
            recorder_info = 'Living Room recorders'
        elif kitchen_var.get() and living_var.get():
            recorder_info = 'Kitchen and Living Room recorders'
        else:
            recorder_info = 'Selected recorders: ' + ', '.join(config.SELECTED_RECORDERS)

        # Append day_info to recorder_info
        recorder_info += f', {day_info}'

        # Load and process data
        threshold_str = threshold_var.get()
        results_dir = 'analysis_results/batch_analysis_results_MUTED' if muted_speech_var.get() else 'analysis_results/batch_analysis_results'
        input_file = os.path.join(results_dir, results_subdir, f'analysis_results_threshold_{threshold_str}.json')

        if not os.path.exists(input_file):
            messagebox.showerror("File Not Found", 
                            f"The analysis results file '{input_file}' does not exist.")
            return

        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            data_counts = data['data_counts']
            
            # Execute plot_results
            plot_results(data_counts, config.SELECTED_RECORDERS, 
                    config.SELECTED_CLASSES, threshold_str, recorder_info,
                    normalize_var.get())

            # Execute run_advanced_analysis only if run_analysis is True
            if run_analysis:
                run_advanced_analysis(data_counts, config.SELECTED_CLASSES, 
                                threshold_str, recorder_info,
                                normalize_var.get())
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while generating the analysis:\n{e}")

    # Add plot button at the bottom
    ttk.Button(root, text="Plot", command=lambda: generate_plot(run_analysis=False)).pack(side="left", padx=20, pady=20)

    # Add plot and analyze button at the bottom
    ttk.Button(root, text="Plot and Analyze", command=lambda: generate_plot(run_analysis=True)).pack(side="right", padx=20, pady=20)

    # Start the main event loop
    root.mainloop()