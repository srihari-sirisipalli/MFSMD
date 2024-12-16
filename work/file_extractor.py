import os
import numpy as np
import pandas as pd
from src.feature_extraction.time_domain import TimeDomainFeatures


def list_channel_files(directory_path):
    """
    List all files in 'TIME SIGNAL' channels for each main folder.

    Args:
        directory_path (str): The path to the root directory.

    Returns:
        dict: A dictionary containing lists of files for each channel in 'TIME SIGNAL' of every main folder.
    """
    channel_files = {}

    try:
        print(f"Scanning directory: {directory_path}")

        # List all main folders in the directory
        main_folders = [entry for entry in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, entry))]
        print(f"Found {len(main_folders)} main folders.\n")

        for main_folder in main_folders:
            main_folder_path = os.path.join(directory_path, main_folder)

            # Locate 'TIME SIGNAL' folder
            time_signal_path = os.path.join(main_folder_path, "TIME SIGNAL")
            if os.path.exists(time_signal_path):
                # List all channels in 'TIME SIGNAL'
                channels = [entry for entry in os.listdir(time_signal_path) if os.path.isdir(os.path.join(time_signal_path, entry))]
                for channel in channels:
                    channel_path = os.path.join(time_signal_path, channel)
                    # List files in the channel
                    files = [os.path.join(channel_path, file) for file in os.listdir(channel_path) if os.path.isfile(os.path.join(channel_path, file))]
                    key = f"{main_folder} - TIME SIGNAL - {channel}"
                    channel_files[key] = files

        print("Directory scan complete.\n")
    except Exception as e:
        print(f"Error during directory scanning: {e}")

    return channel_files


def create_2d_array(channel_files):
    """
    Create a 2D array from the channel_files dictionary where each row contains files from the same index across all channels.

    Args:
        channel_files (dict): A dictionary where keys are channel names and values are lists of files.

    Returns:
        list: A 2D array where each row contains files from corresponding positions across all channels.
    """
    print("Creating 2D array from channel files...")

    # Determine the maximum number of files in any channel
    max_files = max(len(files) for files in channel_files.values())
    total_channels = len(channel_files.keys())

    # Create a 2D array
    rows = []
    for i in range(max_files):
        row = []
        for key in channel_files.keys():
            row.append(channel_files[key][i] if i < len(channel_files[key]) else None)
        rows.append(row)

    print(f"2D array created with shape: ({len(rows)}, {total_channels})\n")
    return rows


def parse_signal(file_path):
    """
    Parse a time-domain signal file to extract metadata, signal values, and time axis.

    Args:
        file_path (str): Path to the time-domain signal file.

    Returns:
        tuple: (signal_values, time_axis, metadata)
    """
    metadata = {}
    signal_values = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('%'):  # Metadata line
                    if ':' in line:
                        key, value = map(str.strip, line[1:].split(':', 1))
                        metadata[key] = value.lstrip('=').strip()
                else:  # Signal value line
                    try:
                        signal_values.append(float(line))
                    except ValueError:
                        pass

        # Handle metadata and defaults
        min_x = float(metadata.get('Min_X', 0))
        max_x = float(metadata.get('Max_X', 1))
        no_of_items = int(metadata.get('NoOfItems', len(signal_values)))

        if no_of_items != len(signal_values):
            raise ValueError(
                f"Mismatch in 'NoOfItems' ({no_of_items}) and number of signal values ({len(signal_values)})."
            )

        # Create time axis
        time_axis = np.linspace(min_x, max_x, len(signal_values))
        return np.array(signal_values), time_axis, metadata

    except Exception as e:
        raise RuntimeError(f"Error parsing signal file '{file_path}': {e}")


def extract_features_from_files(file_array, base_path):
    """
    Extract time-domain features from files listed in a 2D array.

    Args:
        file_array (list): 2D array of file names.
        base_path (str): Base directory containing the files.

    Returns:
        pd.DataFrame: DataFrame containing extracted features.
    """
    print("Starting feature extraction...\n")
    features_list = []
    total_files = sum(1 for row in file_array for file in row if file is not None)
    processed_files = 0

    for row_idx, row in enumerate(file_array):
        for col_idx, file_name in enumerate(row):
            if file_name is None:
                continue

            file_path = os.path.join(base_path, file_name)
            try:
                # Parse the signal
                signal, _, _ = parse_signal(file_path)

                # Extract features
                tdf = TimeDomainFeatures(signal)
                features = {
                    "file_name": file_name,
                    "channel": f"Row {row_idx + 1}, Col {col_idx + 1}",
                    "mean": tdf.mean(),
                    "max": tdf.maximum(),
                    "min": tdf.minimum(),
                    "variance": tdf.variance(),
                    "std_dev": tdf.standard_deviation(),
                    "skewness": tdf.skewness(),
                    "kurtosis": tdf.kurtosis_factor(),
                    "rms": tdf.root_mean_square(),
                    "crest_factor": tdf.crest_factor(),
                    "zero_crossings": tdf.zero_crossings(),
                }
                features_list.append(features)

                processed_files += 1
                if processed_files % 10 == 0:
                    print(f"  Processed {processed_files}/{total_files} files...")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    print(f"\nFeature extraction complete. Processed {processed_files}/{total_files} files.\n")
    return pd.DataFrame(features_list)


if __name__ == "__main__":
    folder_path = r"C:\Users\siris\Projects\Machine Fault Detection and Monitoring System\MFDMS\data\raw\MFFS UNBALANCE 1500 RPM"
    print("Starting pipeline...\n")

    # Step 1: Get the dictionary of channel files
    file_dict = list_channel_files(folder_path)

    # Step 2: Create the 2D array
    file_array = create_2d_array(file_dict)

    # Step 3: Extract features
    features_df = extract_features_from_files(file_array, folder_path)

    # Step 4: Save features to CSV
    features_df.to_csv("unbalance_time_domain_features.csv", index=False)
    print("Features saved to 'unbalance_time_domain_features.csv'.\n")
