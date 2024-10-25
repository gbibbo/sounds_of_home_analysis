import sys
import os
import argparse
import multiprocessing

# Modificar la ruta de búsqueda para incluir el directorio raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.tkinter_interface import run_tkinter_interface

if __name__ == '__main__':
    # Configurar el análisis de argumentos
    parser = argparse.ArgumentParser(description="Run the audio analysis tool.")
    parser.add_argument('--gui', action='store_true', help="Run the program with a graphical user interface")
    
    # Parsear los argumentos
    args = parser.parse_args()

    # Ejecutar la función principal con los argumentos parseados
    if args.gui:
        # Si se pasa el argumento --gui, ejecuta la interfaz gráfica
        multiprocessing.set_start_method('spawn')
        run_tkinter_interface()