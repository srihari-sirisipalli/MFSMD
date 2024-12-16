import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)
import matplotlib.pyplot as plt

from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger(
    "TimeFrequencyVisualizer - Sample Normal Condition - MOTOR DRIER END , CHANNEL-1(Vertical)",
    log_dir="logs/time_frequency_visualizer",
    log_file="visualizer.log")

from PyQt5.QtWidgets import QLineEdit, QLabel, QHBoxLayout
class TimeSignalVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time Signal Visualizer")

        # Default file paths
        self.default_time_signal_file_1 = r"C:\Users\siris\Projects\Machine Fault Detection and Monitoring System\MFDMS\data\raw\MFFS NORMAL 1500 RPM\MOTOR DRIVE END\TIME SIGNAL\CHANNEL 1\M-003.02  MOTOR DE_ 2 channel vib (10-12-2024 17_02_44) - Time Signal.txt"
        self.default_time_signal_file_2 = r"C:\Users\siris\Projects\Machine Fault Detection and Monitoring System\MFDMS\data\raw\MFFS UNBALANCE 1500 RPM\MOTOR DRIVEN END\TIME SIGNAL\CHANNEL 1\M-006.02  MOTOR DE_ 2 channel vib (11-12-2024 14_10_25) - Time Signal.txt"

        logger.debug("Initializing the UI.")
        self.initUI()

    def initUI(self):
        # Initialize main UI components
        self.canvas = FigureCanvas(plt.figure())
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.load_signal1_button = QPushButton("Load Signal 1")
        self.load_signal1_button.clicked.connect(self.load_signal1)

        self.load_signal2_button = QPushButton("Load Signal 2")
        self.load_signal2_button.clicked.connect(self.load_signal2)

        # Range selection widgets
        self.range_widgets = self.create_range_widgets()

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.load_signal1_button)
        layout.addWidget(self.load_signal2_button)

        # Add range selection widgets to the layout
        layout.addLayout(self.range_widgets)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Initialize data variables
        self.time_signal_1 = None
        self.time_axis_1 = None
        self.time_signal_2 = None
        self.time_axis_2 = None

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
        return layout

    def load_signal1(self):
        self.time_signal_1, self.time_axis_1 = self.load_time_signal(self.default_time_signal_file_1)
        self.plot_combined()

    def load_signal2(self):
        self.time_signal_2, self.time_axis_2 = self.load_time_signal(self.default_time_signal_file_2)
        self.plot_combined()

    def load_time_signal(self, default_path):
        file_path = default_path  # Use default path
        options = QFileDialog.Options()
        logger.info("Opening file dialog to select a time signal file.")
        selected_file, _ = QFileDialog.getOpenFileName(self, "Open Time Signal File", file_path, "Text Files (*.txt);;All Files (*)", options=options)

        if selected_file:
            file_path = selected_file
        logger.info(f"File selected: {file_path}")

        try:
            time_signal, time_axis, metadata = self.parse_time_signal(file_path)
            logger.debug(f"Metadata extracted: {metadata}")
            return time_signal, time_axis
        except Exception as e:
            logger.error(f"Error while parsing the file: {e}")
            return None, None

    def parse_time_signal(self, file_path):
        logger.debug(f"Parsing time signal file: {file_path}")
        metadata = {}
        signal_values = []

        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('%'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        value = value.strip().lstrip('= ')
                        metadata[key.strip('% ').strip()] = value.strip()
                        logger.debug(f"Extracted metadata - {key.strip('% ').strip()}: {value.strip()}")
                else:
                    try:
                        signal_values.append(float(line.strip()))
                    except ValueError:
                        logger.warning(f"Non-numeric line ignored: {line.strip()}")

        try:
            min_x = float(metadata.get('Min_X', 0))
            max_x = float(metadata.get('Max_X', 1))
            no_of_items = int(metadata.get('NoOfItems', len(signal_values)))
        except ValueError as e:
            logger.error(f"Error parsing metadata for Min_X, Max_X, or NoOfItems: {e}")
            raise

        logger.debug(f"Creating time axis from Min_X={min_x}, Max_X={max_x}, NoOfItems={no_of_items}")
        time_axis = np.linspace(min_x, max_x, no_of_items)

        return np.array(signal_values), time_axis, metadata

    def plot_combined(self):
        logger.info("Plotting time signals in separate subplots.")
        self.canvas.figure.clear()

        # Create two subplots
        axes = self.canvas.figure.subplots(2, 1, sharex=True)

        # Plot Signal 1 in the top subplot
        if self.time_signal_1 is not None and self.time_axis_1 is not None:
            axes[0].plot(self.time_axis_1, self.time_signal_1, label="Time Signal 1")
            axes[0].set_xlabel("Time (Sec)")
            axes[0].set_ylabel("Amplitude")
            axes[0].set_title("Example - Normal Condition Time Signal")
            axes[0].legend()

        # Plot Signal 2 in the bottom subplot
        if self.time_signal_2 is not None and self.time_axis_2 is not None:
            axes[1].plot(self.time_axis_2, self.time_signal_2, label="Time Signal 2")
            axes[1].set_xlabel("Time (Sec)")
            axes[1].set_ylabel("Amplitude")
            axes[1].set_title("Example - Unbalance Condition Time Signal")
            axes[1].legend()

        # Adjust layout and draw
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        logger.debug("Separate subplots for time signals updated.")


    def update_time_signal_range(self):
        """Update the time signal plot range."""
        try:
            x_min = float(self.time_signal_x_min.text())
            x_max = float(self.time_signal_x_max.text())
            ax = self.canvas.figure.axes[0]
            ax.set_xlim(x_min, x_max)
            self.canvas.draw()
        except ValueError:
            logger.warning("Invalid input for time signal range.")
if __name__ == '__main__':
    logger.info("Starting Time Signal Visualizer application.")
    app = QApplication(sys.argv)
    window = TimeSignalVisualizer()
    window.show()
    sys.exit(app.exec_())
