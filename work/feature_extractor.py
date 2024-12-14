import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog, QLabel, QTextEdit, QWidget)
from feature_extraction.time_domain import TimeDomainFeatures
from feature_extraction.frequence_domain import FrequencyDomainFeatures

class FeatureExtractionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Feature Extraction Tool")
        self.setGeometry(100, 100, 600, 400)

        self.initUI()

    def initUI(self):
        # Create main layout
        layout = QVBoxLayout()

        # Labels
        self.label = QLabel("Load a time-domain or frequency-domain signal file:")
        layout.addWidget(self.label)

        # Buttons
        self.load_button = QPushButton("Load Signal File")
        self.load_button.clicked.connect(self.load_file)
        layout.addWidget(self.load_button)

        self.extract_button = QPushButton("Extract Features")
        self.extract_button.clicked.connect(self.extract_features)
        self.extract_button.setEnabled(False)
        layout.addWidget(self.extract_button)

        # Output display
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        layout.addWidget(self.output_display)

        # Set central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Data attributes
        self.signal = None
        self.sampling_rate = None

    def load_file(self):
        """Load the signal file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Signal File", "", "Text Files (*.txt);;All Files (*)", options=options)

        if file_path:
            try:
                # Read the file
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                # Extract metadata and signal
                self.sampling_rate = None
                signal_values = []

                for line in lines:
                    if line.startswith('%'):
                        # Example: Extract sampling rate if present in metadata
                        if 'SamplingRate' in line:
                            self.sampling_rate = float(line.split(':')[1].strip())
                    else:
                        try:
                            signal_values.append(float(line.strip()))
                        except ValueError:
                            pass

                if not signal_values:
                    raise ValueError("Signal file is empty or incorrectly formatted.")

                self.signal = np.array(signal_values)

                # Set default sampling rate if missing
                if self.sampling_rate is None:
                    self.sampling_rate = 1000  # Default to 1000 Hz

                self.label.setText(f"File loaded: {file_path}\nSampling Rate: {self.sampling_rate} Hz")
                self.extract_button.setEnabled(True)
                self.output_display.append("Signal file loaded successfully.")

            except Exception as e:
                self.output_display.append(f"Error loading file: {str(e)}")

    def extract_features(self):
        """Extract features from the loaded signal."""
        if self.signal is None:
            self.output_display.append("No signal loaded. Please load a file first.")
            return

        self.output_display.append("\nExtracting features...")

        # Determine whether the signal is time-domain or frequency-domain
        if np.isreal(self.signal).all():
            self.output_display.append("Detected time-domain signal. Computing FFT and extracting features.")

            # Time-domain features
            time_features = TimeDomainFeatures(self.signal)

            # Extract and display time-domain features
            self.output_display.append("\nTime-Domain Features:")
            self.output_display.append(f"Mean: {time_features.mean()}")
            self.output_display.append(f"Maximum: {time_features.maximum()}")
            self.output_display.append(f"Minimum: {time_features.minimum()}")
            self.output_display.append(f"RMS: {time_features.root_mean_square()}")
            self.output_display.append(f"Variance: {time_features.variance()}")
            self.output_display.append(f"Skewness: {time_features.skewness()}")
            self.output_display.append(f"Kurtosis: {time_features.kurtosis_factor()}")

            # Frequency-domain features
            freq_features = FrequencyDomainFeatures(self.signal, self.sampling_rate)

        else:
            self.output_display.append("Detected frequency-domain signal. Extracting features directly.")
            freq_features = FrequencyDomainFeatures(self.signal, self.sampling_rate)

        # Extract and display frequency-domain features
        self.output_display.append("\nFrequency-Domain Features:")
        self.output_display.append(f"Mean Frequency: {freq_features.mean_frequency()}")
        self.output_display.append(f"Median Frequency: {freq_features.median_frequency()}")
        self.output_display.append(f"RMS Frequency: {freq_features.root_mean_square_frequency()}")
        self.output_display.append(f"Spectral Entropy: {freq_features.spectral_entropy()}")
        self.output_display.append(f"Energy: {freq_features.energy()}")
        self.output_display.append(f"Residual Energy: {freq_features.residual_energy()}")
        self.output_display.append(f"Spectral Skewness: {freq_features.spectral_skewness()}")
        self.output_display.append(f"Spectral Kurtosis: {freq_features.spectral_kurtosis()}")

        self.output_display.append("\nFeature extraction completed.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FeatureExtractionApp()
    window.show()
    sys.exit(app.exec_())
