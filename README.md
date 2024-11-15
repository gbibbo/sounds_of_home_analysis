# Sounds of Home Analysis

This repository provides tools for analyzing and visualizing sound events detected by recorders from the [Sounds of Home Dataset](https://www.cvssp.org/data/ai4s/sounds_of_home/). It offers a user-friendly interface that adheres to the [AudioSet ontology](https://research.google.com/audioset/ontology/index.html), enabling users to explore, categorize, and analyze acoustic data in a structured manner.

## Visualization Preview

Here's a dynamic preview of the application interface:

![SOH Visualization](assets/images/SOH_Visualization%20(06%20to%2023hs).gif)

## Accessing the Application on Your Local Machine

To view and interact with the application locally:

1. **Start the Local Server**:
   - After downloading the repository and installing any necessary dependencies as per the installation instructions, navigate to the root directory of the project via a command terminal.
   - You will need to start a local server. If you have Python installed, you can quickly start a server with the following command:
     ```bash
     # For Python 3.x
     python -m http.server
     ```
     - This command will start a local server on port 8000.

2. **Access the Application**:
   - Open your web browser and go to [http://localhost:8000](http://localhost:8000).
   - Navigate to the specific HTML file that launches the application interface, for example:
     ```
     http://localhost:8000/path_to_your_html_file.html
     ```
   - Replace `path_to_your_html_file.html` with the actual path to the HTML file within your project directory that serves the application's interface.

## Interface Preview

Here's a preview of the application interface:

![Application Interface](assets/images/interface.png)

The interface allows you to select different parameters and visualize the results like this:

![Example Plot](assets/images/plot.png)

## Features

- **User Interface (GUI) with Tkinter**: Allows selection of parameters such as recorders, sound classes, days, and hours to customize the analysis.
- **Batch Analysis Script**: Perform analyses over multiple confidence thresholds without opening the GUI.
- **Events Statistics Generator**: Analyze and generate comprehensive statistics about sound events.
- **Quality-Based Threshold System**: Adjust confidence thresholds based on AudioSet label quality estimates.
- **Multiprocessing Support**: Utilizes multiple CPU cores to process large datasets quickly.
- **Customizable Visualization**: Generates graphs showing the distribution of sound events by hour and category.
- **AudioSet Ontology Compatibility**: Respects the hierarchy and categories defined in the ontology.
- **Comprehensive Testing Suite**: Includes tests for data processing and analysis functions.

## Requirements

- Python 3.6 or higher
- Python libraries:
  - `tkinter`
  - `matplotlib`
  - `numpy`
  - `pandas`
  - `tqdm`
  - `multiprocessing`
- Access to the "Sounds of Home" experiment dataset

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/gbibbo/sounds_of_home_analysis.git
cd sounds_of_home_analysis
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

**Note**: Make sure your Python environment is properly configured. Using a virtual environment is recommended.

## Dataset

This project is designed to run with the dataset that can be downloaded from:

[Sounds of Home Dataset](https://www.cvssp.org/data/ai4s/sounds_of_home/)

Download the dataset and ensure the prediction files (JSON files) are located in the appropriate directory within the project, as specified in the configuration.

## Project Structure

```plaintext
.
├── analysis_results
│   ├── batch_analysis_results
│   │   └── analysis_results_threshold_*.json
│   └── events_statistics_results
│       ├── events_statistics_results.json
│       ├── main_categories.png
│       └── subcategories_*.png
├── assets
│   └── images
│       ├── interface.png
│       └── plot.png
├── metadata
│   ├── class_labels_indices.csv
│   └── ontology.json
├── README.md
├── requirements.txt
├── scripts
│   ├── batch_analysis.py
│   ├── events_statistics.py
│   ├── main.py
│   └── plot_results.py
├── setup.py
├── src
│   ├── config.py
│   ├── data_processing
│   │   ├── load_data.py
│   │   ├── process_data.py
│   │   └── utils.py
│   ├── gui
│   │   └── tkinter_interface.py
│   └── visualization
│       └── plot_data.py
└── tests
    └── test_data_processing.py
```

**Note**: The directories and files excluded by `.gitignore` (such as sample data and analysis results) are not shown in the project structure.

## Usage

### Configure the Predictions Directory

In the `src/config.py` file, set the path to the directory containing the JSON prediction files you downloaded:

```python
PREDICTIONS_ROOT_DIR = 'path/to/predictions'
```

### Run the Graphical Interface

Execute the `main.py` file to start the graphical interface:

```bash
python scripts/main.py --gui
```

### Select Parameters in the Interface

- **Confidence Threshold**: Set the minimum confidence threshold for considering a valid prediction.
- **Recorders**: Select the recorders whose data you want to analyze.
- **Classes and Subclasses**: Select the sound categories of interest from the predefined CUSTOM_CATEGORIES.
- **Days and Hours**: Select specific days and hours for analysis.

### Run the Analysis

Click the "Run Analysis" button to process the data and generate the visualization.

### View Results

A graph will be displayed representing the percentage of sound events per hour, according to the selected parameters.

## Run Batch Analysis without GUI

A new script `batch_analysis.py` has been added to perform batch analyses over all data without opening the GUI. This is particularly useful for processing large datasets efficiently.

### Steps to Run Batch Analysis

1. **Ensure the Dataset is Configured**: Make sure that `PREDICTIONS_ROOT_DIR` in `scripts/config.py` points to the directory containing your dataset.

2. **Run the Batch Analysis Script**:

```bash
python scripts/batch_analysis.py
```

### What to Expect

- The script will perform analysis for multiple confidence thresholds: 0.0, 0.1, 0.2, 0.3, 0.4, and 0.5.
- It uses multiprocessing to speed up data processing on large datasets.
- The analysis results will be saved in the `analysis_results` directory, with each file named to include the threshold value (e.g., `analysis_results_threshold_0.2.json`).
- Inside each JSON file, the threshold value and the analysis data (`data_counts`) are recorded.

## Generate Plots from Analysis Results

A new script `plot_results.py` has been added to generate plots using the results from the batch analysis.

### Steps to Generate Plots

1. **Ensure Analysis Results Exist**: Run `batch_analysis.py` to generate the analysis results if you haven't already.

2. **Specify Classes and Threshold in `plot_results.py`**:

At the beginning of `scripts/plot_results.py`, you can specify the classes you want to plot and the threshold value:

```python
# Classes to plot
classes_to_plot = [
    'Channel, environment and background',
    'Acoustic environment',
    'Noise',
    'Sound reproduction'
]

# Threshold to use
threshold = 0.2
```

3. **Run the Plotting Script**:

```bash
python scripts/plot_results.py
```

### What to Expect

- The script will read the analysis results from the specified JSON file in the `analysis_results` directory.
- It will generate a bar chart showing the event counts for the specified classes across different hours.
- The plot will be displayed using matplotlib.

### Customizing the Plot

- **Change Classes**: Modify the `classes_to_plot` list to include the classes you're interested in.
- **Change Threshold**: Set the `threshold` variable to match the threshold of the analysis results you want to use.
- **Ensure Data Availability**: Make sure that the analysis results for the specified threshold exist.

## Customization

### Custom Categories Configuration

You can modify or add categories in the `src/config.py` file, where the `CUSTOM_CATEGORIES` dictionary is defined to adapt the analysis to your needs. This affects both the GUI and the batch analysis scripts.

## events_statistics.py

The `events_statistics.py` script processes and analyzes sound event predictions based on the AudioSet ontology. It generates statistical information about the occurrence of different sound categories and their hierarchical relationships in the analyzed audio recordings.

This script is particularly useful for:

- Understanding the distribution of sound events in your dataset.
- Analyzing the hierarchical relationships of sound classes.
- Generating visualizations for better interpretation of sound event data.

![Events Statistics](analysis_results/events_statistics_results/main_categories.png)

### Features

- **Processes multiple JSON prediction files** to aggregate sound event data.
- **Handles hierarchical relationships** using the AudioSet ontology.
- **Applies adjustable confidence thresholds**, including label quality-based thresholds.
- **Generates statistics** at category and subcategory levels.
- **Creates visualizations** like bar charts for easier interpretation.

### Configuration

Adjust settings in `src/config.py` if necessary:

- `PREDICTIONS_ROOT_DIR`: Path to the directory containing prediction JSON files.
- `DEFAULT_CONFIDENCE_THRESHOLD`: Confidence threshold for filtering predictions.
- `USE_LABEL_QUALITY_THRESHOLDS`: Set to `True`  to adjust thresholds based on label quality.
- `GENERATE_GRAPHS`: Set to `False` to generate visualization graphs.
- `CUSTOM_CATEGORIES`: Dictionary defining custom categories and subcategories.

### Usage

To run the `events_statistics.py` script, navigate to the project root directory and execute:

```bash
python scripts/events_statistics.py
```

### Output

- **Statistical results** saved in `events_statistics_results.json`.

- **Visualization plots** saved in `assets/images/`:

- - `main_categories.png`: Bar chart showing percentages of main categories.
- - `subcategories_<category_name>.png`: Bar charts for subcategories within each main category.
- **Console output**: Detailed statistics and summaries printed to the console.

#### Notes

- **Data Preparation**: Ensure prediction JSON files are correctly formatted and located in `PREDICTIONS_ROOT_DIR`.
- **Ontology and Labels**: The script uses `ontology.json` and `class_labels_indices.csv` from the `metadata/` directory.
- **Customization**: Modify `CUSTOM_CATEGORIES` in `config.py` to suit your analysis needs.
- **Visualization**: Generated plots can be used for presentations or further analysis.

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.

2. Create your feature branch:

```bash
git checkout -b feature/new-feature
```

3. Commit your changes:

```bash
git commit -m 'Add new feature'
```

4. Push to the branch:

```bash
git push origin feature/new-feature
```

5. Open a Pull Request on GitHub.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or support, you can reach me through:

- GitHub: gbibbo
- Email: g.bibbo@surrey.ac.uk
