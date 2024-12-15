import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.stats import kurtosis, skew
from logging import Logger
from src.utils.logger import setup_logger  # Ensure the setup_logger function is available

# Initialize logger
logger: Logger = setup_logger("FrequencyDomainFeatures",log_dir="logs/frequency_domain", log_file="frequency_features.log")


class FrequencyDomainFeatures:
    def __init__(self, signal, sampling_rate):
        """
        Initialize the class with the time signal or FFT signal.

        Parameters:
            signal (numpy array): The signal (time domain or frequency domain).
            sampling_rate (float): Sampling rate of the time-domain signal.
        """
        if not isinstance(signal, (np.ndarray, list)):
            logger.error("Signal must be a numpy array or a list.")
            raise ValueError("Signal must be a numpy array or a list.")
        if sampling_rate <= 0:
            logger.error("Sampling rate must be a positive number.")
            raise ValueError("Sampling rate must be a positive number.")

        self.signal = np.asarray(signal)
        self.sampling_rate = sampling_rate
        self.freqs = None
        self.amplitudes = None

        # Check if input is time-domain or frequency-domain
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
        return np.sum(self.freqs * self.amplitudes) / np.sum(self.amplitudes)

    def mean_square_frequency(self):
        return np.sum(self.freqs ** 2 * self.amplitudes) / np.sum(self.amplitudes)

    def root_mean_square_frequency(self):
        return np.sqrt(self.mean_square_frequency())

    def median_frequency(self):
        cumulative_amplitude = np.cumsum(self.amplitudes)
        half_total_amplitude = cumulative_amplitude[-1] / 2
        median_index = np.where(cumulative_amplitude >= half_total_amplitude)[0][0]
        return self.freqs[median_index]

    def variance_of_frequency(self):
        mean_freq = self.mean_frequency()
        return np.sum((self.freqs - mean_freq) ** 2 * self.amplitudes) / np.sum(self.amplitudes)

    def root_variance_frequency(self):
        return np.sqrt(self.variance_of_frequency())

    def spectral_entropy(self):
        power_spectrum = self.amplitudes ** 2
        normalized_power = power_spectrum / np.sum(power_spectrum)
        return -np.sum(normalized_power * np.log2(normalized_power + 1e-12))

    def shannon_entropy(self):
        power_spectrum = self.amplitudes ** 2
        normalized_power = power_spectrum / np.sum(power_spectrum)
        return -np.sum(normalized_power * np.log(normalized_power + 1e-12))

    def spectral_skewness(self):
        return skew(self.amplitudes)

    def spectral_kurtosis(self):
        return kurtosis(self.amplitudes)

    def energy(self):
        return np.sum(self.amplitudes ** 2)

    def residual_energy(self):
        max_energy = np.max(self.amplitudes ** 2)
        return self.energy() - max_energy

    def power_ratio_max_defective(self):
        max_defective_power = np.max(self.amplitudes ** 2)
        mean_power = np.mean(self.amplitudes ** 2)
        return max_defective_power / mean_power


# Example usage
if __name__ == "__main__":
    # Example time-domain signal
    sampling_rate = 1000  # Hz
    time = np.linspace(0, 1, sampling_rate, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * time) + 0.5 * np.sin(2 * np.pi * 120 * time)

    try:
        logger.info("Initializing FrequencyDomainFeatures with example signal.")
        features = FrequencyDomainFeatures(signal, sampling_rate)

        logger.info("Computing frequency domain features.")
        print("\nFrequency Domain Features:")
        print(f"Mean Frequency (FC): {features.mean_frequency():.2f} Hz")
        print(f"Mean Square Frequency (MSF): {features.mean_square_frequency():.2f} Hz²")
        print(f"Root Mean Square Frequency (RMSF): {features.root_mean_square_frequency():.2f} Hz")
        print(f"Median Frequency: {features.median_frequency():.2f} Hz")
        print(f"Variance of Frequency: {features.variance_of_frequency():.2f} Hz²")
        print(f"Root Variance Frequency (RVF): {features.root_variance_frequency():.2f} Hz")
        print(f"Spectral Entropy: {features.spectral_entropy():.4f}")
        print(f"Shannon Entropy: {features.shannon_entropy():.4f}")
        print(f"Spectral Skewness: {features.spectral_skewness():.4f}")
        print(f"Spectral Kurtosis: {features.spectral_kurtosis():.4f}")
        print(f"Energy: {features.energy():.2f}")
        print(f"Residual Energy: {features.residual_energy():.2f}")
        print(f"Power Ratio of Max Defective Frequency to Mean (PMM): {features.power_ratio_max_defective():.2f}")
    except Exception as e:
        logger.error(f"Error during frequency domain feature computation: {e}")
