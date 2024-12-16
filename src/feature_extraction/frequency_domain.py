import numpy as np
from scipy.stats import kurtosis, skew, moment
from scipy.fftpack import fft, fftfreq
from src.utils.logger import setup_logger

logger = setup_logger("FrequencyDomainFeatures", log_dir='logs/frequency_domain', log_file="frequency_features.log")
logger.info("Logger initialized for FrequencyDomainFeatures")

class FrequencyDomainFeatures:
    def __init__(self, signal, sampling_rate):
        """
        Initialize the class with the time signal or FFT signal.

        Parameters:
            signal (numpy array): The time-domain signal.
            sampling_rate (float): Sampling rate of the time-domain signal.
        """
        if not isinstance(signal, (np.ndarray, list)):
            raise ValueError("Signal must be a list or numpy array.")
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be a positive number.")
        
        self.signal = np.asarray(signal)
        self.sampling_rate = sampling_rate
        self.freqs = None
        self.amplitudes = None

        if self._is_time_domain():
            logger.info("Time-domain signal detected. Performing FFT.")
            self._compute_fft()
        else:
            logger.info("Frequency-domain signal detected. Using provided data.")

    def _is_time_domain(self):
        """
        Check if the signal is in the time domain (assumes frequency-domain signals are complex).

        Returns:
            bool: True if the signal is time-domain, False otherwise.
        """
        return np.isreal(self.signal).all()

    def _compute_fft(self):
        """
        Compute the FFT of the time-domain signal.
        """
        fft_result = fft(self.signal)
        self.freqs = fftfreq(len(self.signal), d=1 / self.sampling_rate)[:len(self.signal) // 2]
        self.amplitudes = np.abs(fft_result)[:len(self.signal) // 2]

    def mean_frequency(self):
        logger.debug("Calculating mean frequency.")
        return np.sum(self.freqs * self.amplitudes) / np.sum(self.amplitudes)

    def mean_square_frequency(self):
        logger.debug("Calculating mean square frequency.")
        return np.sum(self.freqs ** 2 * self.amplitudes) / np.sum(self.amplitudes)

    def root_mean_square_frequency(self):
        logger.debug("Calculating root mean square frequency.")
        return np.sqrt(self.mean_square_frequency())

    def median_frequency(self):
        logger.debug("Calculating median frequency.")
        cumulative_amplitude = np.cumsum(self.amplitudes)
        half_total_amplitude = cumulative_amplitude[-1] / 2
        median_index = np.where(cumulative_amplitude >= half_total_amplitude)[0][0]
        return self.freqs[median_index]

    def variance_of_frequency(self):
        logger.debug("Calculating variance of frequency.")
        mean_freq = self.mean_frequency()
        return np.sum((self.freqs - mean_freq) ** 2 * self.amplitudes) / np.sum(self.amplitudes)

    def root_variance_frequency(self):
        logger.debug("Calculating root variance frequency.")
        return np.sqrt(self.variance_of_frequency())

    def spectral_entropy(self):
        logger.debug("Calculating spectral entropy.")
        power_spectrum = self.amplitudes ** 2
        normalized_power = power_spectrum / np.sum(power_spectrum)
        return -np.sum(normalized_power * np.log2(normalized_power + 1e-12))

    def spectral_skewness(self):
        logger.debug("Calculating spectral skewness.")
        return skew(self.amplitudes)

    def spectral_kurtosis(self):
        logger.debug("Calculating spectral kurtosis.")
        return kurtosis(self.amplitudes)

    def energy(self):
        logger.debug("Calculating energy.")
        return np.sum(self.amplitudes ** 2)

    def residual_energy(self):
        logger.debug("Calculating residual energy.")
        max_energy = np.max(self.amplitudes ** 2)
        return self.energy() - max_energy

    def all_features(self):
        """
        Compute and return all frequency domain features as a dictionary.

        Returns:
            dict: A dictionary containing all computed features.
        """
        logger.debug("Calculating all frequency domain features.")
        try:
            features = {
                "Mean Frequency (Hz)": self.mean_frequency(),
                "Mean Square Frequency (Hz^2)": self.mean_square_frequency(),
                "Root Mean Square Frequency (Hz)": self.root_mean_square_frequency(),
                "Median Frequency (Hz)": self.median_frequency(),
                "Variance of Frequency (Hz^2)": self.variance_of_frequency(),
                "Root Variance Frequency (Hz)": self.root_variance_frequency(),
                "Spectral Entropy": self.spectral_entropy(),
                "Spectral Skewness": self.spectral_skewness(),
                "Spectral Kurtosis": self.spectral_kurtosis(),
                "Energy": self.energy(),
                "Residual Energy": self.residual_energy(),
            }
            return features
        except Exception as e:
            logger.error(f"Error calculating frequency domain features: {e}")
            raise

if __name__ == "__main__":
    # Example signal
    sampling_rate = 1000  # Hz
    time = np.linspace(0, 1, sampling_rate, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * time) + 0.5 * np.sin(2 * np.pi * 120 * time)
    
    features = FrequencyDomainFeatures(signal, sampling_rate)
    try:
        all_features = features.all_features()
        print("\nFrequency Domain Features:")
        for feature, value in all_features.items():
            print(f"{feature}: {value}")
    except Exception as e:
        logger.error(f"Error during frequency domain feature computation: {e}")
