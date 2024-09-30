import os
import numpy as np
import pandas as pd
from pathlib import Path

def _aggregate_task_data(path_extracted_data, data_type='HbO'):
    """
    Aggregates the file paths of tasks (VF, 1backWM, SS, GNG) for the specified data type(s) 
    into a nested structure containing class -> subject -> task -> [HbO, HbR, HbT].

    Args:
        path_extracted_data (str or Path): Path to the directory containing extracted data.
        data_type (str): Type of data to filter for ('HbO', 'HbR', 'HbT', or 'all'). 
                         Defaults to 'HbO'. If 'all', all data types will be included.

    Returns:
        dict: A dictionary containing the aggregated file paths organized by class, subject, and task.
    """
    path_extracted_data = Path(path_extracted_data)

    # Find unique class labels in the extracted group
    class_labels = [folder for folder in os.listdir(path_extracted_data) if os.path.isdir(path_extracted_data / folder)]

    # Create a dictionary to store the data for each class
    class_data = {}

    # Define the available data types
    available_data_types = ['HbO', 'HbR', 'HbT']

    # Determine which data types to collect based on input
    data_types_to_collect = available_data_types if data_type == 'all' else [data_type]

    # Load the data for each class
    for class_label in class_labels:
        class_folder = path_extracted_data / class_label

        subject_folders = [f for f in os.listdir(class_folder) if os.path.isdir(class_folder / f)]

        subject_data = {}

        for subject_folder in subject_folders:
            subject_folder_path = class_folder / subject_folder
            task_folders = [f for f in os.listdir(subject_folder_path) if os.path.isdir(subject_folder_path / f)]

            task_data = {}

            for task_folder in task_folders:
                task_folder_path = subject_folder_path / task_folder
                task_files = os.listdir(task_folder_path)

                # Filter task files for the data types specified in data_types_to_collect
                collected_files = []
                for dt in data_types_to_collect:
                    if data_files := [
                        str(task_folder_path / f)
                        for f in task_files
                        if dt in f
                    ]:
                        collected_files.append(data_files[0])  # Assuming one file per type

                if collected_files:
                    task_data[task_folder] = collected_files

            if task_data:
                subject_data[subject_folder] = task_data

        if subject_data:
            class_data[class_label] = subject_data

    return class_data

def _load_and_concatenate(subject_data):
        HbO_df = pd.read_csv(subject_data[0], header=None)  # Load HbO
        HbR_df = pd.read_csv(subject_data[1], header=None)  # Load HbR
        HbT_df = pd.read_csv(subject_data[2], header=None)  # Load HbT

        # Extract channel data (excluding the first two rows: 'time' and 'time-marker')
        HbO_channels = HbO_df.iloc[2:, :].values  # Shape (23, no_of_samples)
        HbR_channels = HbR_df.iloc[2:, :].values  # Shape (23, no_of_samples)
        HbT_channels = HbT_df.iloc[2:, :].values  # Shape (23, no_of_samples)

        # Concatenate along the channel axis (23+23+23 = 69 channels)
        concatenated_data = np.concatenate((HbO_channels, HbR_channels, HbT_channels), axis=0)  # Shape (69, no_of_samples)
        
        # Convert to DataFrame for consistency
        return pd.DataFrame(concatenated_data)


def load_concatenated_fNIRS_data(data_folder_path, task_type):
    """
    Loads and concatenates HbO, HbR, and HbT data for each subject and task type (e.g., 'VF', 'GNG').
    
    Args:
        data_folder_path (str or Path): Path to the folder containing the data.
        task_type (str): Task type to load ('VF', 'GNG', '1backWM', 'SS').
        
    Returns:
        dict: A dictionary containing the concatenated DataFrames for each subject and task.
              The structure is: { 'healthy': { 'subject_name': { 'task_type': DataFrame } }, 
                                  'non-healthy': { 'subject_name': { 'task_type': DataFrame } } }
    """
    # Aggregate task data using the existing function
    data_dict = _aggregate_task_data(data_folder_path, data_type='all')
    
    # Initialize an empty dictionary to store the concatenated data
    concatenated_data_dict = {'healthy': {}, 'non-healthy': {}}

    # Function to load and concatenate the data for a given subject and task

    # Process healthy data
    for subject, tasks in data_dict['healthy'].items():
        subject_tasks = {}
        if task_type in tasks:
            subject_data = tasks[task_type]  # Get the list of file paths (HbO, HbR, HbT)
            subject_tasks[task_type] = _load_and_concatenate(subject_data)
        concatenated_data_dict['healthy'][subject] = subject_tasks

    # Process non-healthy data
    for subject, tasks in data_dict['non-healthy'].items():
        subject_tasks = {}
        if task_type in tasks:
            subject_data = tasks[task_type]  # Get the list of file paths (HbO, HbR, HbT)
            subject_tasks[task_type] = _load_and_concatenate(subject_data)
        concatenated_data_dict['non-healthy'][subject] = subject_tasks

    return concatenated_data_dict

def create_data_and_labels(fNIRS_data_dict: dict) -> tuple:
    """
    Converts the fNIRS_data_dict into cnn_data and labels arrays for deep learning.
    
    Args:
        fNIRS_data_dict (dict): The dictionary containing the fNIRS data.
    
    Returns:
        tuple: A tuple containing:
            - cnn_data (numpy.ndarray): Array of shape (num_samples, channels, time).
            - labels (numpy.ndarray): Array of shape (num_samples,), with 0 for healthy and 1 for non-healthy.
    """
    if not isinstance(fNIRS_data_dict, dict):
        raise ValueError("Input must be a dictionary")

    cnn_data = []
    labels = []

    # Process healthy data
    for _, tasks in fNIRS_data_dict.get('healthy', {}).items():
        for _, data_df in tasks.items():
            # Convert the DataFrame to a NumPy array and append to cnn_data
            cnn_data.append(data_df.values)
            # Append the corresponding label (0 for healthy)
            labels.append(0)
    
    # Process non-healthy data
    for _, tasks in fNIRS_data_dict.get('non-healthy', {}).items():
        for _, data_df in tasks.items():
            # Convert the DataFrame to a NumPy array and append to cnn_data
            cnn_data.append(data_df.values)
            # Append the corresponding label (1 for non-healthy)
            labels.append(1)

    # Convert the lists to NumPy arrays
    cnn_data = np.array(cnn_data)  # Shape: (num_samples, channels, time)
    labels = np.array(labels)  # Shape: (num_samples,)
    
    return cnn_data, labels

