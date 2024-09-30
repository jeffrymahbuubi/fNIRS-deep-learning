import pandas as pd
import glob
import os
from tqdm import tqdm

def check_unique_values_in_csv_files(directory):
    # Dictionary to hold the filename and unique values from the second row
    result = {}

    # Use glob to find all .csv files in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    # Use tqdm to create a progress bar
    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        try:
            # Read the CSV file using pandas
            df = pd.read_csv(csv_file, header=None)
            print(df)

            # Check if there is at least two rows (including header)
            if len(df) > 1:
                # Get the second row (index 1)
                second_row = df.iloc[1, :]
                print(second_row)

                # Extract unique values
                unique_values = set(second_row)
                result[os.path.basename(csv_file)] = unique_values
                print(unique_values)
            else:
                result[os.path.basename(csv_file)] = "No second row available"
        except Exception as e:
            result[os.path.basename(csv_file)] = f"Error reading file: {e}"

    return result


# Example usage
directory_path = 'D:\AUNUUN JEFFRY MAHBUUBI\PROJECT AND RESEARCH\PROJECTS\\36. FNIRS-Anxiety\CODE\\3. fNIRS\data\\raw\HRF_csv'  # Specify your directory containing the CSV files
unique_values = check_unique_values_in_csv_files(directory_path)

# Print or process the result as needed
for filename, values in unique_values.items():
    print(f"File: {filename}")
    print(f"Unique values in the 2nd row: {values}")
    print("-" * 50)
