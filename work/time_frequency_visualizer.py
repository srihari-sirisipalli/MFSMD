import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)
import matplotlib.pyplot as plt

import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from PyQt5.QtWidgets import QLineEdit, QLabel, QHBoxLayout

class TimeSignalVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time Signal Visualizer")

        logging.debug("Initializing the UI.")
        self.initUI()

    def initUI(self):
        # Initialize main UI components
        self.canvas = FigureCanvas(plt.figure())
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.load_button = QPushButton("Load Time Signal File")
        self.load_button.clicked.connect(self.load_file)

        self.fft_button = QPushButton("Compute FFT")
        self.fft_button.clicked.connect(self.compute_fft)
        self.fft_button.setEnabled(False)

        self.load_fft_button = QPushButton("Load FFT or Spectrum File")
        self.load_fft_button.clicked.connect(self.load_fft_file)

        # Range selection widgets
        self.range_widgets = self.create_range_widgets()

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.load_button)
        layout.addWidget(self.fft_button)
        layout.addWidget(self.load_fft_button)

        # Add range selection widgets to the layout
        layout.addLayout(self.range_widgets)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initialize data variables
        self.time_signal = None
        self.time_axis = None
        self.fft_frequencies = None
        self.fft_amplitudes = None
        self.spectrum_frequencies = None
        self.spectrum_amplitudes = None

    def create_range_widgets(self):
        """Create widgets for range selection."""
        layout = QVBoxLayout()

        # Time Signal Range
        time_signal_layout = QHBoxLayout()
        time_signal_layout.addWidget(QLabel("Time Signal X Range:"))
        self.time_signal_x_min = QLineEdit()
        self.time_signal_x_max = QLineEdit()
        time_signal_layout.addWidget(self.time_signal_x_min)
        time_signal_layout.addWidget(self.time_signal_x_max)
        time_signal_layout.addWidget(QPushButton("Apply", clicked=self.update_time_signal_range))

        layout.addLayout(time_signal_layout)

        # FFT Range
        fft_layout = QHBoxLayout()
        fft_layout.addWidget(QLabel("FFT X Range:"))
        self.fft_x_min = QLineEdit()
        self.fft_x_max = QLineEdit()
        fft_layout.addWidget(self.fft_x_min)
        fft_layout.addWidget(self.fft_x_max)
        fft_layout.addWidget(QPushButton("Apply", clicked=self.update_fft_range))

        layout.addLayout(fft_layout)

        # Spectrum Range
        spectrum_layout = QHBoxLayout()
        spectrum_layout.addWidget(QLabel("Spectrum X Range:"))
        self.spectrum_x_min = QLineEdit()
        self.spectrum_x_max = QLineEdit()
        spectrum_layout.addWidget(self.spectrum_x_min)
        spectrum_layout.addWidget(self.spectrum_x_max)
        spectrum_layout.addWidget(QPushButton("Apply", clicked=self.update_spectrum_range))

        layout.addLayout(spectrum_layout)

        return layout

    def update_time_signal_range(self):
        """Update the time signal plot range."""
        try:
            x_min = float(self.time_signal_x_min.text())
            x_max = float(self.time_signal_x_max.text())
            ax = self.canvas.figure.axes[0]
            ax.set_xlim(x_min, x_max)
            self.canvas.draw()
        except ValueError:
            logging.warning("Invalid input for time signal range.")

    def update_fft_range(self):
        """Update the FFT plot range."""
        try:
            x_min = float(self.fft_x_min.text())
            x_max = float(self.fft_x_max.text())
            ax = self.canvas.figure.axes[1]
            ax.set_xlim(x_min, x_max)
            self.canvas.draw()
        except ValueError:
            logging.warning("Invalid input for FFT range.")

    def update_spectrum_range(self):
        """Update the spectrum plot range."""
        try:
            x_min = float(self.spectrum_x_min.text())
            x_max = float(self.spectrum_x_max.text())
            ax = self.canvas.figure.axes[2]
            ax.set_xlim(x_min, x_max)
            self.canvas.draw()
        except ValueError:
            logging.warning("Invalid input for spectrum range.")

    def load_file(self):
        options = QFileDialog.Options()
        logging.info("Opening file dialog to select a time signal file.")
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Time Signal File", "", "Text Files (*.txt);;All Files (*)", options=options)

        if file_path:
            logging.info(f"File selected: {file_path}")
            try:
                self.time_signal, self.time_axis, metadata = self.parse_time_signal(file_path)
                logging.debug(f"Metadata extracted: {metadata}")
                self.plot_combined()
                self.fft_button.setEnabled(True)  # Enable FFT button
            except Exception as e:
                logging.error(f"Error while parsing the file: {e}")
        else:
            logging.warning("No file selected.")

    def load_fft_file(self):
        options = QFileDialog.Options()
        logging.info("Opening file dialog to select an FFT or Spectrum file.")
        file_path, _ = QFileDialog.getOpenFileName(self, "Open FFT or Spectrum File", "", "Text Files (*.txt);;All Files (*)", options=options)

        if file_path:
            logging.info(f"File selected: {file_path}")
            try:
                # Decide whether the file is FFT or Spectrum based on content or file name
                if "Spectrum" in file_path:  # Adjust condition as necessary
                    self.spectrum_frequencies, self.spectrum_amplitudes = self.parse_fft_file(file_path)
                else:
                    self.fft_frequencies, self.fft_amplitudes = self.parse_fft_file(file_path)
                self.adjust_fft_values()
                self.plot_combined()
            except Exception as e:
                logging.error(f"Error while parsing the FFT or Spectrum file: {e}")
        else:
            logging.warning("No file selected.")

    def parse_time_signal(self, file_path):
        logging.debug(f"Parsing time signal file: {file_path}")
        metadata = {}
        signal_values = []

        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('%'):
                    # Extract metadata from the header lines
                    if ':' in line:
                        key, value = line.split(':', 1)
                        value = value.strip().lstrip('=')  # Remove '=' if it exists
                        metadata[key.strip('% ').strip()] = value.strip()
                        logging.debug(f"Extracted metadata - {key.strip('% ').strip()}: {value.strip()}")
                else:
                    try:
                        # Parse signal values
                        signal_values.append(float(line.strip()))
                    except ValueError:
                        logging.warning(f"Non-numeric line ignored: {line.strip()}")

        # Create a time axis based on Min_X and Max_X
        try:
            min_x = float(metadata.get('Min_X', 0))
            max_x = float(metadata.get('Max_X', 1))
            no_of_items = int(metadata.get('NoOfItems', len(signal_values)))
        except ValueError as e:
            logging.error(f"Error parsing metadata for Min_X, Max_X, or NoOfItems: {e}")
            raise

        logging.debug(f"Creating time axis from Min_X={min_x}, Max_X={max_x}, NoOfItems={no_of_items}")
        time_axis = np.linspace(min_x, max_x, no_of_items)

        return np.array(signal_values), time_axis, metadata

    def parse_fft_file(self, file_path):
        logging.debug(f"Parsing FFT or Spectrum file: {file_path}")
        frequencies = []
        amplitudes = []
        metadata = {}

        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('%'):
                    # Extract metadata from the header lines
                    if ':' in line:
                        key, value = line.split(':', 1)
                        value = value.strip().lstrip('=')
                        metadata[key.strip('% ').strip()] = value.strip()
                        logging.debug(f"Extracted metadata - {key.strip('% ').strip()}: {value.strip()}")
                else:
                    try:
                        freq, amp = map(float, line.strip().split())
                        frequencies.append(freq)
                        amplitudes.append(amp)
                    except ValueError:
                        try:
                            # If only amplitude is present, assume it's single-column
                            amplitudes.append(float(line.strip()))
                        except ValueError:
                            logging.warning(f"Non-numeric or malformed line ignored: {line.strip()}")

        if not frequencies:
            # Generate frequencies if missing, using metadata
            try:
                min_x = float(metadata.get('Min_X', 0))
                max_x = float(metadata.get('Max_X', 1))
                no_of_items = int(metadata.get('NoOfItems', len(amplitudes)))
                frequencies = np.linspace(min_x, max_x, no_of_items)
                logging.debug(f"Generated frequencies from Min_X={min_x}, Max_X={max_x}, NoOfItems={no_of_items}")
            except Exception as e:
                logging.error(f"Error generating frequencies from metadata: {e}")
                raise

        logging.debug(f"Parsed {len(frequencies)} frequencies and {len(amplitudes)} amplitudes.")
        return np.array(frequencies), np.array(amplitudes)

    def adjust_fft_values(self):
        if self.fft_frequencies is not None and self.fft_amplitudes is not None:
            logging.info("Adjusting FFT values: setting amplitudes to 0 for frequencies between 1000 and 1200 Hz.")
            mask = (self.fft_frequencies >= 1000) & (self.fft_frequencies <= 1200)
            self.fft_amplitudes[mask] = 0
            logging.debug("FFT amplitudes adjusted.")


    def plot_combined(self):
        logging.info("Plotting combined data.")
        self.canvas.figure.clear()

        # Create subplots for Time Signal, FFT, and Spectrum
        axes = self.canvas.figure.subplots(3, 1, sharex=False)

        # Plot time signal
        if self.time_signal is not None and self.time_axis is not None:
            axes[0].plot(self.time_axis, self.time_signal, label="Time Signal")
            axes[0].set_xlabel("Time (Sec)")
            axes[0].set_ylabel("Amplitude")
            axes[0].set_title("Time Signal Visualization")
            axes[0].legend()

        # Plot FFT
        if self.fft_frequencies is not None and self.fft_amplitudes is not None:
            axes[1].plot(self.fft_frequencies, self.fft_amplitudes, label="FFT Spectrum")
            axes[1].set_xlabel("Frequency (Hz)")
            axes[1].set_ylabel("Amplitude")
            axes[1].set_title("FFT - Frequency Spectrum (Calculated from Time Signal)")
            axes[1].legend()

        # Plot Spectrum
        if self.spectrum_frequencies is not None and self.spectrum_amplitudes is not None:
            axes[2].plot(self.spectrum_frequencies, self.spectrum_amplitudes, label="Spectrum Signal")
            axes[2].set_xlabel("Frequency (Hz)")
            axes[2].set_ylabel("Amplitude")
            axes[2].set_title("Spectrum Signal (Loaded from the file)")
            axes[2].legend()

        self.canvas.figure.tight_layout()
        self.canvas.draw()
        logging.debug("Combined plot updated.")

    def compute_fft(self):
        if self.time_signal is None or self.time_axis is None:
            logging.error("No time signal loaded to compute FFT.")
            return

        logging.info("Computing FFT of the time signal.")
        signal_length = len(self.time_signal)
        sampling_interval = (self.time_axis[-1] - self.time_axis[0]) / (signal_length - 1)
        sampling_frequency = 1 / sampling_interval

        logging.debug(f"Signal length: {signal_length}, Sampling interval: {sampling_interval}, Sampling frequency: {sampling_frequency}")

        fft_result = np.fft.fft(self.time_signal)
        fft_frequencies = np.fft.fftfreq(signal_length, d=sampling_interval)

        self.fft_frequencies = fft_frequencies[:signal_length // 2]
        self.fft_amplitudes = np.abs(fft_result[:signal_length // 2])

        self.plot_combined()

if __name__ == '__main__':
    logging.info("Starting Time Signal Visualizer application.")
    app = QApplication(sys.argv)
    window = TimeSignalVisualizer()
    window.show()
    sys.exit(app.exec_())
