import numpy as np
from src.feature_extraction.frequency_domain import FrequencyDomainFeatures
from src.feature_extraction.time_domain import TimeDomainFeatures
import src.feature_extraction.time_domain as time_domain
print(time_domain.TimeDomainFeatures)

def parse_file(filepath):
    """
    Parse the input file to extract metadata and signal values.

    Parameters:
        filepath (str): Path to the input file.

    Returns:
        tuple: (metadata dictionary, signal values list)
    """
    metadata = {}
    signal_values = []
    
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('%'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    value = value.strip().lstrip('=')
                    key = key.strip('% ').strip()
                    metadata[key] = value
            elif line:  # Non-empty line
                try:
                    signal_values.append(float(line))
                except ValueError:
                    pass  # Ignore invalid lines
    return metadata, signal_values

def calculate_sampling_rate(metadata):
    """
    Calculate the sampling rate from metadata.

    Parameters:
        metadata (dict): Parsed metadata.

    Returns:
        float: Calculated sampling rate.
    """
    try:
        no_of_items = int(metadata['NoOfItems'])
        min_x = float(metadata['Min_X'])
        max_x = float(metadata['Max_X'])
        duration = max_x - min_x
        sampling_rate = no_of_items / duration
        return sampling_rate
    except KeyError as e:
        raise ValueError(f"Missing metadata key: {e}")

def extract_features(signal_values, sampling_rate):
    """
    Extract time-domain and frequency-domain features.

    Parameters:
        signal_values (list or np.ndarray): Signal values.
        sampling_rate (float): Sampling rate of the signal.

    Returns:
        dict: Dictionary containing extracted features.
    """
    # Convert signal to numpy array
    signal = np.array(signal_values)

    # Time-Domain Features
    time_features = TimeDomainFeatures(signal)
    time_domain_results = {
        "Mean": time_features.mean(),
        "Maximum": time_features.maximum(),
        "Minimum": time_features.minimum(),
        "Variance": time_features.variance(),
        "Standard Deviation": time_features.standard_deviation(),
        "Skewness": time_features.skewness(),
        "Kurtosis": time_features.kurtosis_factor(),
        "Zero Crossings": time_features.zero_crossings(),
        "Peak-to-Peak": time_features.peak_to_peak(),
        "Root Mean Square (RMS)": time_features.root_mean_square(),
        "Absolute Average Value": time_features.absolute_average_value(),
        "Crest Factor": time_features.crest_factor(),
        "Hjorth Parameters": time_features.hjorth_parameters(),
        "Entropy Estimation": time_features.entropy_estimation(),
    }

    # Frequency-Domain Features
    freq_features = FrequencyDomainFeatures(signal, sampling_rate)
    frequency_domain_results = {
        "Mean Frequency": freq_features.mean_frequency(),
        "Mean Square Frequency": freq_features.mean_square_frequency(),
        "Root Mean Square Frequency": freq_features.root_mean_square_frequency(),
        "Median Frequency": freq_features.median_frequency(),
        "Variance of Frequency": freq_features.variance_of_frequency(),
        "Root Variance Frequency": freq_features.root_variance_frequency(),
        "Spectral Entropy": freq_features.spectral_entropy(),
        "Shannon Entropy": freq_features.shannon_entropy(),
        "Spectral Skewness": freq_features.spectral_skewness(),
        "Spectral Kurtosis": freq_features.spectral_kurtosis(),
        "Energy": freq_features.energy(),
        "Residual Energy": freq_features.residual_energy(),
        "Power Ratio of Max Defective": freq_features.power_ratio_max_defective(),
    }

    # Combine results
    all_features = {**time_domain_results, **frequency_domain_results}
    return all_features

if __name__ == "__main__":
    # Path to your input file
    file_path = r'C:\Users\siris\Projects\Machine Fault Detection and Monitoring System\MFDMS\data\raw\MFFS NORMAL 1500 RPM\MOTOR DRIVE END\TIME SIGNAL\CHANNEL 1\M-003.02  MOTOR DE_ 2 channel vib (10-12-2024 17_02_44) - Time Signal.txt'  # Replace with your file path

    # Parse the file
    metadata, signal_values = parse_file(file_path)
    
    # Calculate sampling rate
    try:
        sampling_rate = calculate_sampling_rate(metadata)
        print(f"Sampling Rate: {sampling_rate:.2f} Hz")
    except ValueError as e:
        print(f"Error calculating sampling rate: {e}")
        exit()

    # Extract features
    try:
        features = extract_features(signal_values, sampling_rate)
        print("\nExtracted Features:")
        for feature, value in features.items():
            print(f"{feature}: {value}")
    except Exception as e:
        print(f"Error extracting features: {e}")



