import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.stats import kurtosis, skew
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FrequencyDomainFeatures:
    def __init__(self, signal, sampling_rate):
        """
        Initialize the class with the time signal or FFT signal.

        Parameters:
            signal (numpy array): The signal (time domain or frequency domain).
            sampling_rate (float): Sampling rate of the time-domain signal.
        """
        self.signal = np.asarray(signal)
        self.sampling_rate = sampling_rate
        self.freqs = None
        self.amplitudes = None

        # Check if input is time-domain or frequency-domain
        if self._is_time_domain():
            logging.info("Time-domain signal detected. Performing FFT.")
            self._compute_fft()
        else:
            logging.info("Frequency-domain signal detected. Using provided data.")

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
        """
        Compute the mean frequency or frequency center (FC).

        Returns:
            float: Mean frequency.
        """
        return np.sum(self.freqs * self.amplitudes) / np.sum(self.amplitudes)

    def mean_square_frequency(self):
        """
        Compute the mean square frequency (MSF).

        Returns:
            float: Mean square frequency.
        """
        return np.sum(self.freqs ** 2 * self.amplitudes) / np.sum(self.amplitudes)

    def root_mean_square_frequency(self):
        """
        Compute the root mean square frequency (RMSF).

        Returns:
            float: RMS frequency.
        """
        return np.sqrt(self.mean_square_frequency())

    def median_frequency(self):
        """
        Compute the median frequency.

        Returns:
            float: Median frequency.
        """
        cumulative_amplitude = np.cumsum(self.amplitudes)
        half_total_amplitude = cumulative_amplitude[-1] / 2
        median_index = np.where(cumulative_amplitude >= half_total_amplitude)[0][0]
        return self.freqs[median_index]

    def variance_of_frequency(self):
        """
        Compute the variance of the frequency.

        Returns:
            float: Variance of the frequency.
        """
        mean_freq = self.mean_frequency()
        return np.sum((self.freqs - mean_freq) ** 2 * self.amplitudes) / np.sum(self.amplitudes)

    def root_variance_frequency(self):
        """
        Compute the root variance frequency (RVF).

        Returns:
            float: Root variance frequency.
        """
        return np.sqrt(self.variance_of_frequency())

    def spectral_entropy(self):
        """
        Compute the spectral entropy.

        Returns:
            float: Spectral entropy.
        """
        power_spectrum = self.amplitudes ** 2
        normalized_power = power_spectrum / np.sum(power_spectrum)
        return -np.sum(normalized_power * np.log2(normalized_power + 1e-12))

    def shannon_entropy(self):
        """
        Compute Shannon entropy.

        Returns:
            float: Shannon entropy.
        """
        power_spectrum = self.amplitudes ** 2
        normalized_power = power_spectrum / np.sum(power_spectrum)
        return -np.sum(normalized_power * np.log(normalized_power + 1e-12))

    def spectral_skewness(self):
        """
        Compute the spectral skewness.

        Returns:
            float: Spectral skewness.
        """
        return skew(self.amplitudes)

    def spectral_kurtosis(self):
        """
        Compute the spectral kurtosis.

        Returns:
            float: Spectral kurtosis.
        """
        return kurtosis(self.amplitudes)

    def energy(self):
        """
        Compute the energy of the signal.

        Returns:
            float: Energy.
        """
        return np.sum(self.amplitudes ** 2)

    def residual_energy(self):
        """
        Compute the residual energy (total energy minus max peak energy).

        Returns:
            float: Residual energy.
        """
        max_energy = np.max(self.amplitudes ** 2)
        return self.energy() - max_energy

    def power_ratio_max_defective(self):
        """
        Compute the power ratio of the max defective frequency to the mean.

        Returns:
            float: Power ratio.
        """
        max_defective_power = np.max(self.amplitudes ** 2)
        mean_power = np.mean(self.amplitudes ** 2)
        return max_defective_power / mean_power

# Example usage
if __name__ == "__main__":
    # Example time-domain signal
    sampling_rate = 1000  # Hz
    time = np.linspace(0, 1, sampling_rate, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * time) + 0.5 * np.sin(2 * np.pi * 120 * time)

    features = FrequencyDomainFeatures(signal, sampling_rate)

    print("Mean Frequency (FC):", features.mean_frequency())
    print("Mean Square Frequency (MSF):", features.mean_square_frequency())
    print("Root Mean Square Frequency (RMSF):", features.root_mean_square_frequency())
    print("Median Frequency:", features.median_frequency())
    print("Variance of Frequency:", features.variance_of_frequency())
    print("Root Variance Frequency (RVF):", features.root_variance_frequency())
    print("Spectral Entropy:", features.spectral_entropy())
    print("Shannon Entropy:", features.shannon_entropy())
    print("Spectral Skewness:", features.spectral_skewness())
    print("Spectral Kurtosis:", features.spectral_kurtosis())
    print("Energy:", features.energy())
    print("Residual Energy:", features.residual_energy())
    print("Power Ratio of Max Defective Frequency to Mean (PMM):", features.power_ratio_max_defective())
