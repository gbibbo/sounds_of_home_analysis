# scripts/extract_csv.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def safe_convert_number(value_str, to_int=True):
    """
    Safely converts a string to a number, handling commas and decimal points.
    
    Args:
        value_str (str): String to convert
        to_int (bool): If True, converts to integer, otherwise to float
        
    Returns:
        int or float: Converted number, or 0 if conversion fails
    """
    try:
        # Remove commas and convert to float first
        clean_value = float(value_str.replace(',', ''))
        # If we want an integer and the float is close to a whole number, convert to int
        if to_int and abs(clean_value - round(clean_value)) < 0.01:
            return int(round(clean_value))
        # If we want an integer but have a true decimal, keep as int
        elif to_int:
            return int(clean_value)
        # If we want a float, return as is
        return clean_value
    except (ValueError, AttributeError):
        return 0

def get_audioset_data():
    """
    Downloads and processes AudioSet data from the official website.
    
    Returns:
        pandas.DataFrame: DataFrame containing AudioSet class parameters
    """
    # AudioSet URL
    url = "https://research.google.com/audioset/dataset/index.html"
    
    # Make request with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Error downloading page after {max_retries} attempts: {e}")
                return None
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2)

    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the table
    table = soup.find('table', id='dataset-index')
    if not table:
        print("Could not find data table on page")
        return None
    
    # Initialize list for data
    data = []
    
    # Process each row
    for row in table.find_all('tr', class_='ab'):
        try:
            # Get all cells
            cells = row.find_all('td')
            if len(cells) < 9:  # Ensure enough cells
                continue
            
            # Get label
            label = cells[0].find('a').text.strip()
            
            # Get quality estimate
            quality = float(cells[1]['data-quality']) if cells[1].has_attr('data-quality') else 0.0
            
            # Get videos and hours - using safe conversion and correcting the order
            total_videos = safe_convert_number(cells[2].text.strip(), to_int=True)
            eval_hours = safe_convert_number(cells[3].text.strip(), to_int=False)
            eval_videos = safe_convert_number(cells[4].text.strip(), to_int=True)
            balanced_train_hours = safe_convert_number(cells[5].text.strip(), to_int=False)
            balanced_train_videos = safe_convert_number(cells[6].text.strip(), to_int=True)
            # Corrected order for unbalanced data
            unbalanced_train_hours = safe_convert_number(cells[7].text.strip(), to_int=False)
            unbalanced_train_videos = safe_convert_number(cells[8].text.strip(), to_int=True)
            
            data.append({
                'label': label,
                'quality_estimate': quality,
                'total_videos': total_videos,
                'evaluation_videos': eval_videos,
                'balanced_train_videos': balanced_train_videos,
                'unbalanced_train_videos': unbalanced_train_videos,
                'evaluation_hours': eval_hours,
                'balanced_train_hours': balanced_train_hours,
                'unbalanced_train_hours': unbalanced_train_hours
            })
            
        except Exception as e:
            print(f"Error processing row for {label if 'label' in locals() else 'unknown'}: {str(e)}")
            continue
    
    # Check if we got any data
    if not data:
        print("No data was processed successfully")
        return None
        
    return pd.DataFrame(data)

def main():
    print("Downloading and processing AudioSet data...")
    df = get_audioset_data()
    
    if df is not None and not df.empty:
        # Save to CSV
        output_file = 'metadata/audioset_data.csv'
        df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        print(f"Total classes processed: {len(df)}")
        
        # Show some statistics
        print("\nFirst 5 classes:")
        print(df.head())
        print("\nTotal videos statistics:")
        print(df['total_videos'].describe())
        
        return df
    else:
        print("Could not obtain data or DataFrame is empty")
        return None

if __name__ == "__main__":
    main()