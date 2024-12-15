import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QWidget, QTableWidget, QTableWidgetItem, QTabWidget
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from src.feature_extraction.frequency_domain import FrequencyDomainFeatures
from src.feature_extraction.time_domain import TimeDomainFeatures
from src.utils.logger import setup_logger

# Set up the logger
logger = setup_logger("FeatureExtractorApp", log_dir="logs", debug=True)


class FeatureExtractorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Feature Extractor")
        self.setGeometry(100, 100, 1200, 900)

        # Main layout
        self.main_layout = QVBoxLayout()

        # UI Components
        self.label = QLabel("Select a text file to extract features:")
        self.label.setFont(QFont("Arial", 14))
        self.main_layout.addWidget(self.label)

        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        self.main_layout.addWidget(self.file_button)

        # Matplotlib Figure for plotting
        self.figure = Figure(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        self.main_layout.addWidget(self.canvas)

        # Tabs for features
        self.tabs = QTabWidget()
        self.time_tab = QWidget()
        self.freq_tab = QWidget()

        # Time-Domain Tab Layout
        self.time_layout = QVBoxLayout()
        self.time_table = QTableWidget()
        self.time_table.setColumnCount(2)
        self.time_table.setHorizontalHeaderLabels(["Feature", "Value"])
        self.time_layout.addWidget(self.time_table)
        self.time_tab.setLayout(self.time_layout)

        # Frequency-Domain Tab Layout
        self.freq_layout = QVBoxLayout()
        self.freq_table = QTableWidget()
        self.freq_table.setColumnCount(2)
        self.freq_table.setHorizontalHeaderLabels(["Feature", "Value"])
        self.freq_layout.addWidget(self.freq_table)
        self.freq_tab.setLayout(self.freq_layout)

        self.tabs.addTab(self.time_tab, "Time-Domain Features")
        self.tabs.addTab(self.freq_tab, "Frequency-Domain Features")

        self.main_layout.addWidget(self.tabs)

        # Main widget
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        logger.info("Feature Extractor Application Initialized")

    def select_file(self):
        # File dialog to select a text file
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Signal File", "", "Text Files (*.txt)")
        if file_path:
            try:
                logger.info(f"File selected: {file_path}")
                metadata, signal_values = self.parse_file(file_path)
                sampling_rate = self.calculate_sampling_rate(metadata)
                logger.info(f"Sampling rate calculated: {sampling_rate} Hz")

                # Plot time signal
                self.plot_time_signal(signal_values)

                # Extract features
                time_features = self.extract_time_features(signal_values)
                freq_features = self.extract_frequency_features(signal_values, sampling_rate)

                # Display features
                self.display_features(time_features, freq_features)
                logger.info("Features successfully extracted and displayed.")
            except Exception as e:
                self.label.setText(f"Error: {e}")
                logger.error(f"Error during feature extraction: {e}")

    @staticmethod
    def parse_file(filepath):
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
        logger.info(f"Parsed metadata: {metadata}")
        return metadata, signal_values

    @staticmethod
    def calculate_sampling_rate(metadata):
        try:
            no_of_items = int(metadata['NoOfItems'])
            min_x = float(metadata['Min_X'])
            max_x = float(metadata['Max_X'])
            duration = max_x - min_x
            return no_of_items / duration
        except KeyError as e:
            raise ValueError(f"Missing metadata key: {e}")

    def plot_time_signal(self, signal, title="Time Signal", xlabel="Time (samples)", ylabel="Amplitude"):
        """
        Plot the time-domain signal.

        Parameters:
            signal (list): The time-domain signal to plot.
        """
        logger.debug("Plotting time-domain signal.")
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(signal, color='b', alpha=0.8)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()

    @staticmethod
    def extract_time_features(signal_values):
        signal = np.array(signal_values)
        time_features = TimeDomainFeatures(signal)

        results = {
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

        logger.debug(f"Extracted time-domain features: {results}")
        return results

    @staticmethod
    def extract_frequency_features(signal_values, sampling_rate):
        signal = np.array(signal_values)
        freq_features = FrequencyDomainFeatures(signal, sampling_rate)

        results = {
            "Mean Frequency (FC)": freq_features.mean_frequency(),
            "Mean Square Frequency (MSF)": freq_features.mean_square_frequency(),
            "Root Mean Square Frequency (RMSF)": freq_features.root_mean_square_frequency(),
            "Median Frequency": freq_features.median_frequency(),
            "Variance of Frequency": freq_features.variance_of_frequency(),
            "Root Variance Frequency (RVF)": freq_features.root_variance_frequency(),
            "Spectral Entropy": freq_features.spectral_entropy(),
            "Shannon Entropy": freq_features.shannon_entropy(),
            "Spectral Skewness": freq_features.spectral_skewness(),
            "Spectral Kurtosis": freq_features.spectral_kurtosis(),
            "Energy": freq_features.energy(),
            "Residual Energy": freq_features.residual_energy(),
            "Power Ratio of Max Defective Frequency to Mean (PMM)": freq_features.power_ratio_max_defective(),
        }

        logger.debug(f"Extracted frequency-domain features: {results}")
        return results

    def display_features(self, time_features, freq_features):
        # Populate time-domain table
        self.time_table.setRowCount(len(time_features))
        for row, (feature, value) in enumerate(time_features.items()):
            self.time_table.setItem(row, 0, QTableWidgetItem(feature))
            self.time_table.setItem(row, 1, QTableWidgetItem(str(value)))

        # Populate frequency-domain table
        self.freq_table.setRowCount(len(freq_features))
        for row, (feature, value) in enumerate(freq_features.items()):
            self.freq_table.setItem(row, 0, QTableWidgetItem(feature))
            self.freq_table.setItem(row, 1, QTableWidgetItem(str(value)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FeatureExtractorApp()
    main_window.show()
    sys.exit(app.exec_())
