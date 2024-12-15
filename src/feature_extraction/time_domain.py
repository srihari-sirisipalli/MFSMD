import numpy as np
from scipy.stats import kurtosis, skew, moment
from scipy.signal import find_peaks
from skimage.morphology import dilation, erosion, opening, closing
from src.utils.logger import setup_logger

logger = setup_logger("TimeDomainFeatures",log_dir='logs/time_domain', log_file="time_domain_features.log")
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
        activity = variance
        mobility = np.sqrt(diff_variance / variance)
        complexity = np.sqrt(np.var(np.diff(diff_signal)) / diff_variance) / mobility
        return activity, mobility, complexity

    def crest_factor(self):
        logger.debug("Calculating crest factor.")
        rms = self.root_mean_square()
        if rms == 0:
            logger.warning("RMS value is zero; crest factor is undefined.")
            return np.inf
        return self.peak_to_peak() / rms

    def histogram_bounds(self, bins=10):
        logger.debug("Calculating histogram bounds.")
        hist, bin_edges = np.histogram(self.signal, bins=bins)
        lower = bin_edges[0]
        upper = bin_edges[-1]
        diff = upper - lower
        return lower, upper, diff

    def upper_lower_histogram(self, threshold):
        """
        Compute counts of values above and below a threshold.

        Parameters:
            threshold (float): The threshold value.

        Returns:
            tuple: (upper_count, lower_count)
        """
        logger.debug(f"Calculating upper and lower histogram for threshold {threshold}.")
        upper_count = np.sum(self.signal > threshold)
        lower_count = np.sum(self.signal <= threshold)
        return upper_count, lower_count

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

    def morphological_operations(self):
        """
        Perform mathematical morphology operations: dilation, erosion, opening, and closing.

        Returns:
            tuple: (dilated, eroded, opened, closed)
        """
        logger.debug("Performing mathematical morphology operations.")
        if len(self.signal.shape) != 1:
            logger.error("Signal must be 1D for morphological operations.")
            raise ValueError("Signal must be 1D for morphological operations.")
        signal_reshaped = self.signal.reshape(-1, 1)
        dilated = dilation(signal_reshaped).flatten()
        eroded = erosion(signal_reshaped).flatten()
        opened = opening(signal_reshaped).flatten()
        closed = closing(signal_reshaped).flatten()
        logger.info("Performed dilation, erosion, opening, and closing operations.")
        return dilated, eroded, opened, closed


if __name__ == "__main__":
    # Example signal: Noisy sine wave
    signal = np.sin(np.linspace(0, 10 * np.pi, 1000)) + 0.1 * np.random.randn(1000)
    features = TimeDomainFeatures(signal)

    try:
        # Statistical Features
        print("\nStatistical Features:")
        print(f"Mean: {features.mean()}")
        print(f"Maximum: {features.maximum()}")
        print(f"Minimum: {features.minimum()}")
        print(f"Variance: {features.variance()}")
        print(f"Standard Deviation: {features.standard_deviation()}")
        print(f"Skewness: {features.skewness()}")
        print(f"Kurtosis: {features.kurtosis_factor()}")
        print(f"Zero Crossings: {features.zero_crossings()}")

        # Signal Shape Descriptors
        print("\nSignal Shape Descriptors:")
        print(f"Peak-to-Peak: {features.peak_to_peak()}")
        print(f"Root Mean Square (RMS): {features.root_mean_square()}")
        print(f"Absolute Average Value: {features.absolute_average_value()}")
        print(f"Crest Factor: {features.crest_factor()}")

        # Advanced Features
        print("\nAdvanced Features:")
        print(f"Hjorth Parameters: {features.hjorth_parameters()}")
        print(f"Entropy Estimation: {features.entropy_estimation()}")

        # Histogram Features
        print("\nHistogram Features:")
        lower, upper, diff = features.histogram_bounds()
        print(f"Histogram Bounds: Lower={lower}, Upper={upper}, Difference={diff}")
        upper_count, lower_count = features.upper_lower_histogram(threshold=0)
        print(f"Values Above Threshold: {upper_count}, Below Threshold: {lower_count}")

        # Morphological Operations
        print("\nMorphological Operations (First 5 Values):")
        dilated, eroded, opened, closed = features.morphological_operations()
        print(f"Dilation: {dilated[:5]}")
        print(f"Erosion: {eroded[:5]}")
        print(f"Opening: {opened[:5]}")
        print(f"Closing: {closed[:5]}")

    except Exception as e:
        logger.error(f"Error during feature computation: {e}")
