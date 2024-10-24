# src/main.py

import sys
import os
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.tkinter_interface import run_tkinter_interface

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    run_tkinter_interface()
