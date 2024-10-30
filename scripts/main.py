# scripts/main.py

import sys
import os
import argparse
import multiprocessing

# Modify the search path to include the root directory of the project
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.tkinter_interface import run_tkinter_interface

if __name__ == '__main__':
    # Configuring argument analysis
    parser = argparse.ArgumentParser(description="Run the audio analysis tool.")
    parser.add_argument('--gui', action='store_true', help="Run the program with a graphical user interface")
    # Parse arguments
    args = parser.parse_args()

    # Execute the main function with the parsed arguments
    if args.gui:
        # If the --gui argument is passed, it executes the graphical interface
        multiprocessing.set_start_method('spawn')
        run_tkinter_interface()