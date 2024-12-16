import numpy as np
from scipy.stats import kurtosis, skew, moment
from scipy.signal import find_peaks
from skimage.morphology import dilation, erosion, opening, closing
from src.utils.logger import setup_logger

logger = setup_logger("TimeDomainFeatures", log_dir='logs/time_domain', log_file="time_domain_features.log")
logger.info("Logger initialized for TimeDomainFeatures")

class TimeDomainFeatures:
    def __init__(self, signal):
        """
        Initialize the class with the time signal.

        Parameters:
            signal (numpy array): The time-domain signal.
        """
        if not isinstance(signal, (np.ndarray, list)):
            raise ValueError("Signal must be a list or numpy array.")
        self.signal = np.asarray(signal)
        if self.signal.size == 0:
            raise ValueError("Signal cannot be empty.")
        logger.info("Initialized TimeDomainFeatures class.")

    def mean(self):
        logger.debug("Calculating mean.")
        return np.mean(self.signal)

    def maximum(self):
        logger.debug("Calculating maximum value.")
        return np.max(self.signal)

    def minimum(self):
        logger.debug("Calculating minimum value.")
        return np.min(self.signal)

    def peak_to_peak(self):
        logger.debug("Calculating peak-to-peak value.")
        return np.ptp(self.signal)

    def root_mean_square(self):
        logger.debug("Calculating root mean square.")
        return np.sqrt(np.mean(self.signal ** 2))

    def absolute_average_value(self):
        logger.debug("Calculating absolute average value.")
        return np.mean(np.abs(self.signal))

    def standard_deviation(self):
        logger.debug("Calculating standard deviation.")
        return np.std(self.signal)

    def variance(self):
        logger.debug("Calculating variance.")
        return np.var(self.signal)

    def skewness(self):
        logger.debug("Calculating skewness.")
        return skew(self.signal)

    def kurtosis_factor(self):
        logger.debug("Calculating kurtosis factor.")
        return kurtosis(self.signal)

    def zero_crossings(self):
        logger.debug("Calculating zero crossings.")
        zero_crossings = np.where(np.diff(np.sign(self.signal)))[0]
        return len(zero_crossings)

    def hjorth_parameters(self):
        """
        Compute the Hjorth parameters: activity, mobility, and complexity.

        Returns:
            tuple: (activity, mobility, complexity)
        """
        logger.debug("Calculating Hjorth parameters.")
        variance = np.var(self.signal)
        if variance == 0:
            logger.warning("Signal variance is zero; Hjorth parameters are undefined.")
            return 0, 0, 0
        diff_signal = np.diff(self.signal)
        diff_variance = np.var(diff_signal)
        activity = float(variance)
        mobility = float(np.sqrt(diff_variance / variance))
        complexity = float(np.sqrt(np.var(np.diff(diff_signal)) / diff_variance) / mobility)
        return [activity, mobility, complexity]

    def crest_factor(self):
        logger.debug("Calculating crest factor.")
        rms = self.root_mean_square()
        if rms == 0:
            logger.warning("RMS value is zero; crest factor is undefined.")
            return np.inf
        return self.peak_to_peak() / rms

    def entropy_estimation(self):
        """
        Estimate the entropy of the signal.

        Returns:
            float: Entropy of the signal.
        """
        logger.debug("Estimating entropy of the signal.")
        probability_distribution, _ = np.histogram(self.signal, bins=256, density=True)
        probability_distribution = probability_distribution[probability_distribution > 0]
        return -np.sum(probability_distribution * np.log2(probability_distribution))

    def all_features(self):
        """
        Compute and return all features as a dictionary.

        Returns:
            dict: A dictionary containing all computed features.
        """
        logger.debug("Calculating all features.")
        try:
            hjorth = self.hjorth_parameters()
            features = {
                "1st Hjorth parameter (activity)": hjorth[0],
                "2nd Hjorth parameter (mobility)": hjorth[1],
                "3rd Hjorth parameter (complexity)": hjorth[2],
                "Absolute Average Value": self.absolute_average_value(),
                "Crest Factor": self.crest_factor(),
                "Impulse Factor (IF)": self.peak_to_peak() / np.mean(np.abs(self.signal)),
                "Kurtosis Factor": self.kurtosis_factor(),
                "Maximum": self.maximum(),
                "Mean": self.mean(),
                "Minimum": self.minimum(),
                "Peak-to-Peak": self.peak_to_peak(),
                "Root Mean Square (RMS)": self.root_mean_square(),
                "Shape Factor (waveform factor)": self.root_mean_square() / self.absolute_average_value(),
                "Skewness": self.skewness(),
                "Standard Deviation": self.standard_deviation(),
                "Variance": self.variance(),
                "Zero Crossings": self.zero_crossings(),
                "Entropy Estimation": self.entropy_estimation(),
            }
            return features
        except Exception as e:
            logger.error(f"Error calculating all features: {e}")
            raise

if __name__ == "__main__":
    # Example signal: Noisy sine wave
    signal = np.sin(np.linspace(0, 10 * np.pi, 1000)) + 0.1 * np.random.randn(1000)
    features = TimeDomainFeatures(signal)

    try:
        all_features = features.all_features()
        print("\nAll Features:")
        for feature, value in all_features.items():
            print(f"{feature}: {value}")

    except Exception as e:
        logger.error(f"Error during feature computation: {e}")
