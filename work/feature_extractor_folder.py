import os
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QWidget, QTableWidget, QTableWidgetItem
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from src.feature_extraction.time_domain import TimeDomainFeatures
from src.utils.logger import setup_logger

# Set up the logger
logger = setup_logger("FeatureExtractorApp", log_dir="logs", debug=True)


class FeatureExtractorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Condition-Based Feature Extractor")
        self.setGeometry(100, 100, 1200, 800)

        # Main layout
        self.main_layout = QVBoxLayout()

        # UI Components
        self.label = QLabel("Select a folder corresponding to a condition (e.g., NORMAL or UNBALANCED):")
        self.label.setFont(QFont("Arial", 14))
        self.main_layout.addWidget(self.label)

        self.folder_button = QPushButton("Select Folder")
        self.folder_button.clicked.connect(self.select_folder)
        self.main_layout.addWidget(self.folder_button)

        self.result_label = QLabel("Results:")
        self.result_label.setFont(QFont("Arial", 12))
        self.main_layout.addWidget(self.result_label)

        # Table to display features
        self.feature_table = QTableWidget()
        self.feature_table.setColumnCount(2)
        self.feature_table.setHorizontalHeaderLabels(["Feature", "Value"])
        self.main_layout.addWidget(self.feature_table)

        # Main widget
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        logger.info("Feature Extractor Application Initialized")

    def select_folder(self):
        # Folder dialog to select the condition folder
        folder_path = QFileDialog.getExistingDirectory(self, "Select Condition Folder")
        if folder_path:
            try:
                logger.info(f"Folder selected: {folder_path}")
                self.extract_features_from_condition(folder_path)
            except Exception as e:
                self.result_label.setText(f"Error: {e}")
                logger.error(f"Error during feature extraction: {e}")

    def extract_features_from_condition(self, folder_path):
        """
        Extract features from the selected condition folder.
        """
        time_signal_path = os.path.join(folder_path, "TIME SIGNAL")
        if not os.path.exists(time_signal_path):
            raise ValueError("TIME SIGNAL folder not found in the selected condition folder.")

        # Organize files for each signal (16 files per signal)
        file_groups = self.organize_files_by_signal(time_signal_path)
        if not file_groups:
            raise ValueError("No valid file groups found for feature extraction.")

        # Extract features for the first group of 16 files
        logger.info(f"Processing {len(file_groups)} groups of signals.")
        first_group = file_groups[0]
        combined_signal = self.combine_signals(first_group)
        time_features = self.extract_time_features(combined_signal)

        # Display features
        self.display_features(time_features)

    @staticmethod
    def organize_files_by_signal(base_path):
        """
        Organize files into groups of 16 for each signal.
        """
        signal_files = []
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith(".txt"):
                    signal_files.append(os.path.join(root, file))

        # Group files into sets of 16 (4 measurement points × 2 channels × 2 components)
        signal_files.sort()
        file_groups = [signal_files[i:i + 16] for i in range(0, len(signal_files), 16)]
        logger.info(f"Found {len(file_groups)} signal groups.")
        return file_groups

    @staticmethod
    def combine_signals(file_group):
        """
        Combine the signals from 16 files into a single array.
        """
        combined_signal = []
        for file_path in file_group:
            with open(file_path, 'r') as f:
                signal = [float(line.strip()) for line in f if line.strip()]
                combined_signal.extend(signal)
        logger.info(f"Combined signal length: {len(combined_signal)}")
        return np.array(combined_signal)

    @staticmethod
    def extract_time_features(signal):
        """
        Extract time-domain features from the combined signal.
        """
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

    def display_features(self, features):
        """
        Display extracted features in the table.
        """
        self.feature_table.setRowCount(len(features))
        for row, (feature, value) in enumerate(features.items()):
            self.feature_table.setItem(row, 0, QTableWidgetItem(feature))
            self.feature_table.setItem(row, 1, QTableWidgetItem(str(value)))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FeatureExtractorApp()
    main_window.show()
    sys.exit(app.exec_())
