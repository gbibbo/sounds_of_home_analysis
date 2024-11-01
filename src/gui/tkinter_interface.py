# src/gui/tkinter_interface.py

import os
import tkinter as tk
from tkinter import ttk, messagebox
import src.config as config
from scripts.plot_results import plot_results
import json

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
    root.geometry("1100x820")

    # Configure styles
    style = ttk.Style(root)
    style.configure('TButton', background='white', foreground='black')
    style.map('TButton',
              background=[('active', 'white')],
              foreground=[('active', 'black')])
    style.configure('Selected.TButton', background='blue', foreground='white')

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

    def update_recorder_selection():
        """Update recorder checkboxes based on Kitchen/Living Room selection"""
        for rec in KITCHEN_RECORDERS:
            recorder_vars[rec].set(kitchen_var.get())
            recorder_checkboxes[rec]['state'] = 'disabled' if kitchen_var.get() else 'normal'
        for rec in LIVING_RECORDERS:
            recorder_vars[rec].set(living_var.get())
            recorder_checkboxes[rec]['state'] = 'disabled' if living_var.get() else 'normal'

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
    for category_name, subcategories in config.CUSTOM_CATEGORIES.items():
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

    def generate_plot():
        """Generate plot based on selected parameters"""
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

        # Load and process data
        threshold_str = threshold_var.get()
        input_file = os.path.join('analysis_results/batch_analysis_results', 
                                f'analysis_results_threshold_{threshold_str}.json')

        if not os.path.exists(input_file):
            messagebox.showerror("File Not Found", 
                               f"The analysis results file '{input_file}' does not exist.")
            return

        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
            data_counts = data['data_counts']
            plot_results(data_counts, config.SELECTED_RECORDERS, 
                       config.SELECTED_CLASSES, threshold_str, recorder_info,
                       normalize_var.get())
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while generating the plot:\n{e}")

    # Add plot button at the bottom
    ttk.Button(root, text="Plot", command=generate_plot).pack(pady=20)

    # Start the main event loop
    root.mainloop()