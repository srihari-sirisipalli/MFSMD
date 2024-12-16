import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
    QTableWidget, QTableWidgetItem, QTextEdit, QWidget, QTableWidgetSelectionRange, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt
from src.feature_extraction.time_domain import TimeDomainFeatures  # Assuming TimeDomainFeatures is in the same directory


class FeatureExtractorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Time Domain Feature Extractor")
        self.setGeometry(200, 200, 1000, 600)
        self.selected_files = []
        self.extracted_features = []
        self.init_ui()

    def init_ui(self):
        # Main layout
        self.central_widget = QWidget()
        self.main_layout = QVBoxLayout()

        # File selection area
        self.file_layout = QHBoxLayout()
        self.select_files_button = QPushButton("Select Files")
        self.select_files_button.clicked.connect(self.select_files)
        self.file_layout.addWidget(self.select_files_button)

        self.remove_files_button = QPushButton("Remove Selected Files")
        self.remove_files_button.clicked.connect(self.remove_selected_files)
        self.file_layout.addWidget(self.remove_files_button)

        self.process_files_button = QPushButton("Process Files")
        self.process_files_button.clicked.connect(self.process_files)
        self.file_layout.addWidget(self.process_files_button)

        self.main_layout.addLayout(self.file_layout)

        # Selected files list
        self.file_list_widget = QListWidget()
        self.file_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.main_layout.addWidget(self.file_list_widget)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(12)
        self.results_table.setHorizontalHeaderLabels([
            "File Name", "Mean", "Max", "Min", "Variance", "Std Dev", 
            "Skewness", "Kurtosis", "RMS", "Crest Factor", "Zero Crossings", "Channel"
        ])
        self.main_layout.addWidget(self.results_table)

        # Status display
        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.main_layout.addWidget(self.status_display)

        # Save results button
        self.save_results_button = QPushButton("Save Results to CSV")
        self.save_results_button.clicked.connect(self.save_results)
        self.main_layout.addWidget(self.save_results_button)

        # Set layout
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

    def select_files(self):
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Text Files",
            "",
            "Text Files (*.txt);;All Files (*)",
            options=options
        )

        # Add files to the list if not already added
        for file_path in file_paths:
            if file_path not in self.selected_files:
                self.selected_files.append(file_path)
                self.file_list_widget.addItem(QListWidgetItem(file_path))
        self.status_display.append(f"Selected {len(file_paths)} file(s).")

    def remove_selected_files(self):
        selected_items = self.file_list_widget.selectedItems()
        for item in selected_items:
            file_path = item.text()
            self.selected_files.remove(file_path)
            self.file_list_widget.takeItem(self.file_list_widget.row(item))
        self.status_display.append(f"Removed {len(selected_items)} file(s).")

    def process_files(self):
        if not self.selected_files:
            self.status_display.append("No files selected. Please select files first.")
            return

        self.extracted_features = []
        self.results_table.setRowCount(0)  # Clear previous results

        for file_path in self.selected_files:
            try:
                signal, _, _ = self.parse_time_signal(file_path)
                features = self.extract_features(signal, file_path)
                self.add_feature_to_table(features)
                self.extracted_features.append(features)
                self.status_display.append(f"Processed file: {file_path}")
            except Exception as e:
                self.status_display.append(f"Error processing {file_path}: {e}")

    def parse_time_signal(self, file_path):
        metadata = {}
        signal_values = []

        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('%'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        value = value.strip().lstrip('=')
                        metadata[key.strip('% ').strip()] = value.strip()
                else:
                    try:
                        signal_values.append(float(line.strip()))
                    except ValueError:
                        pass  # Ignore non-numeric lines

        return np.array(signal_values), None, metadata

    def extract_features(self, signal, file_name):
        tdf = TimeDomainFeatures(signal)
        features = {
            "file_name": file_name,
            "mean": tdf.mean(),
            "max": tdf.maximum(),
            "min": tdf.minimum(),
            "variance": tdf.variance(),
            "std_dev": tdf.standard_deviation(),
            "skewness": tdf.skewness(),
            "kurtosis": tdf.kurtosis_factor(),
            "rms": tdf.root_mean_square(),
            "crest_factor": tdf.crest_factor(),
            "zero_crossings": tdf.zero_crossings(),
            "channel": "N/A",  # Placeholder for channel info
        }
        return features

    def add_feature_to_table(self, features):
        row_idx = self.results_table.rowCount()
        self.results_table.insertRow(row_idx)

        for col_idx, key in enumerate(features.keys()):
            value = features[key]
            self.results_table.setItem(row_idx, col_idx, QTableWidgetItem(str(value)))

    def save_results(self):
        if not self.extracted_features:
            self.status_display.append("No results to save. Please process files first.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results to CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if save_path:
            features_df = pd.DataFrame(self.extracted_features)
            features_df.to_csv(save_path, index=False)
            self.status_display.append(f"Results saved to {save_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FeatureExtractorApp()
    main_window.show()
    sys.exit(app.exec_())
