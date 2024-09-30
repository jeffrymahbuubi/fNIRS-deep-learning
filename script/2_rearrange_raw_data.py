import pandas as pd
import glob
from pathlib import Path
import os

BASE_DIR = Path("D:\AUNUUN JEFFRY MAHBUUBI\PROJECT AND RESEARCH\PROJECTS\\36. FNIRS-Anxiety\CODE\\3. fNIRS\data\\raw\HRF_csv")

def main(directory):
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    excel_files = glob.glob(os.path.join(directory, "*.xlsx"))

    # Load the Excel file
    df_excel = pd.read_excel(excel_files[0])

    # Extract subject IDs, tasks, and records from the Excel file
    subject_ids = df_excel.iloc[:, 0]
    tasks = df_excel.iloc[:, 1]
    records = df_excel.iloc[:, -1]

    # Dictionary to store subject data
    subject_data = {}

    for idx, subject_id in enumerate(subject_ids):
        task = tasks[idx]
        record = records[idx]

        # Create or update the subject entry
        if subject_id not in subject_data:
            # Classify based on the second character of the subject ID
            classification = 'healthy' if subject_id[1] == 'H' else 'non-healthy'
            subject_data[subject_id] = {'data': {}, 'class': classification}

        # Collect all files (HbO, HbR, HbT) that match the record
        matched_files = []
        for csv_file in csv_files:
            filename = "_".join(os.path.basename(csv_file).split("_")[:2])
            if record == filename:
                matched_files.append(csv_file)

        # If we find all three (HbO, HbR, HbT) for the task, store them in a list under the task
        if matched_files:
            subject_data[subject_id]['data'][task] = matched_files

    return pd.DataFrame.from_dict(subject_data, orient='index')

if __name__ == "__main__":
    df = main(BASE_DIR)
    print("Test")
