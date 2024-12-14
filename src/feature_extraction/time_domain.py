import numpy as np
from scipy.stats import kurtosis, skew, moment
from scipy.signal import find_peaks
from skimage.morphology import dilation, erosion, opening, closing
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeDomainFeatures:
    def __init__(self, signal):
        """
        Initialize the class with the time signal.

        Parameters:
            signal (numpy array): The time-domain signal.
        """
        self.signal = np.asarray(signal)
        logging.info("Initialized TimeDomainFeatures class.")

    def mean(self):
        logging.debug("Calculating mean.")
        return np.mean(self.signal)

    def maximum(self):
        logging.debug("Calculating maximum value.")
        return np.max(self.signal)

    def minimum(self):
        logging.debug("Calculating minimum value.")
        return np.min(self.signal)

    def peak_value(self):
        logging.debug("Calculating peak value.")
        return np.max(np.abs(self.signal))

    def peak_to_peak(self):
        logging.debug("Calculating peak-to-peak value.")
        return np.ptp(self.signal)

    def root_mean_square(self):
        logging.debug("Calculating root mean square.")
        return np.sqrt(np.mean(self.signal ** 2))

    def absolute_average_value(self):
        logging.debug("Calculating absolute average value.")
        return np.mean(np.abs(self.signal))

    def standard_deviation(self):
        logging.debug("Calculating standard deviation.")
        return np.std(self.signal)

    def variance(self):
        logging.debug("Calculating variance.")
        return np.var(self.signal)

    def skewness(self):
        logging.debug("Calculating skewness.")
        return skew(self.signal)

    def skewness_index(self):
        logging.debug("Calculating skewness index.")
        return skew(self.signal) / (np.std(self.signal) ** 3)

    def kurtosis_factor(self):
        logging.debug("Calculating kurtosis factor.")
        return kurtosis(self.signal)

    def zero_crossings(self):
        logging.debug("Calculating zero crossings.")
        zero_crossings = np.where(np.diff(np.sign(self.signal)))[0]
        return len(zero_crossings)

    def hjorth_parameters(self):
        """
        Compute the 1st, 2nd, and 3rd Hjorth parameters.

        Returns:
            (tuple): Activity, mobility, and complexity.
        """
        logging.debug("Calculating Hjorth parameters.")
        variance = np.var(self.signal)
        diff_signal = np.diff(self.signal)
        diff_variance = np.var(diff_signal)
        activity = variance
        mobility = np.sqrt(diff_variance / variance)
        complexity = np.sqrt(np.var(np.diff(diff_signal)) / diff_variance) / mobility
        return activity, mobility, complexity

    def central_moments(self, order):
        """
        Compute the central moment of a given order.

        Parameters:
            order (int): The order of the central moment.

        Returns:
            float: The central moment.
        """
        logging.debug(f"Calculating central moment of order {order}.")
        return moment(self.signal, moment=order)

    def crest_factor(self):
        logging.debug("Calculating crest factor.")
        return self.peak_value() / self.root_mean_square()

    def clearance_factor(self):
        logging.debug("Calculating clearance factor.")
        max_value = self.peak_value()
        rms = self.root_mean_square()
        return max_value / (rms ** 2)

    def impulse_factor(self):
        logging.debug("Calculating impulse factor.")
        return self.peak_value() / self.absolute_average_value()

    def shape_factor(self):
        logging.debug("Calculating shape factor.")
        return self.root_mean_square() / self.absolute_average_value()

    def histogram_bounds(self, bins=10):
        logging.debug("Calculating histogram bounds.")
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
            (int, int): Counts of values above and below the threshold.
        """
        logging.debug(f"Calculating upper and lower histogram for threshold {threshold}.")
        upper_count = np.sum(self.signal > threshold)
        lower_count = np.sum(self.signal <= threshold)
        return upper_count, lower_count

    def autoregressive_coefficients(self, order=4):
        """
        Compute autoregressive (AR) coefficients using Yule-Walker equations.

        Parameters:
            order (int): The order of the AR model.

        Returns:
            numpy array: AR coefficients.
        """
        logging.debug("Calculating autoregressive coefficients.")
        from statsmodels.regression.linear_model import yule_walker

        rho, sigma = yule_walker(self.signal, order=order)
        return rho

    def entropy_estimation(self):
        """
        Estimate entropy of the signal.

        Returns:
            float: Entropy of the signal.
        """
        logging.debug("Estimating entropy of the signal.")
        probability_distribution, _ = np.histogram(self.signal, bins=256, density=True)
        probability_distribution = probability_distribution[probability_distribution > 0]
        return -np.sum(probability_distribution * np.log2(probability_distribution))

    def morphological_operations(self):
        """
        Perform mathematical morphology operations: dilation, erosion, opening, and closing.

        Returns:
            tuple: Dilation, erosion, opening, and closing of the signal.
        """
        logging.debug("Performing mathematical morphology operations.")
        signal_reshaped = self.signal.reshape(-1, 1)  # Reshape for compatibility with morphology functions
        dilated = dilation(signal_reshaped).flatten()
        eroded = erosion(signal_reshaped).flatten()
        opened = opening(signal_reshaped).flatten()
        closed = closing(signal_reshaped).flatten()
        logging.info("Performed dilation, erosion, opening, and closing operations.")
        return dilated, eroded, opened, closed

# Example usage
if __name__ == "__main__":
    # Example signal
    signal = np.sin(np.linspace(0, 10 * np.pi, 1000)) + 0.1 * np.random.randn(1000)

    features = TimeDomainFeatures(signal)

    print("Mean:", features.mean())
    print("Maximum:", features.maximum())
    print("Minimum:", features.minimum())
    print("Peak-to-peak:", features.peak_to_peak())
    print("RMS:", features.root_mean_square())
    print("Absolute Average Value:", features.absolute_average_value())
    print("Standard Deviation:", features.standard_deviation())
    print("Variance:", features.variance())
    print("Skewness:", features.skewness())
    print("Skewness Index:", features.skewness_index())
    print("Kurtosis:", features.kurtosis_factor())
    print("Zero Crossings:", features.zero_crossings())
    print("Hjorth Parameters:", features.hjorth_parameters())
    print("Crest Factor:", features.crest_factor())
    print("Impulse Factor:", features.impulse_factor())
    print("Shape Factor:", features.shape_factor())
    print("Histogram Bounds:", features.histogram_bounds())
    print("Upper and Lower Histogram:", features.upper_lower_histogram(threshold=0))
    print("Entropy Estimation:", features.entropy_estimation())

    dilated, eroded, opened, closed = features.morphological_operations()
    print("Dilation:", dilated[:5])  # Display first 5 values
    print("Erosion:", eroded[:5])  # Display first 5 values
    print("Opening:", opened[:5])
    print("Closing:", closed[:5])
