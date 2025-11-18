"""
Plot window for displaying SMDS calculation results.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtWidgets import QMainWindow, QFileDialog, QMessageBox


class PlotWindow(QMainWindow):
    """
    Separate window for displaying matplotlib plots with save options.

    This window displays the results of SMDS calculations as a line plot
    and provides options to save the plot as an image or export the data as CSV.

    Attributes:
        s2_values (np.ndarray): The S2 values to plot
        fig: Matplotlib figure object
        ax: Matplotlib axes object
    """

    def __init__(self, s2_values: np.ndarray, parent=None):
        """
        Initialize the plot window.

        Args:
            s2_values: NumPy array of S2 values to display
            parent: Parent window (optional)
        """
        super().__init__(parent)

        self.setWindowTitle("SMDS Calculation Results")
        self.setGeometry(200, 200, 800, 600)
        self.s2_values = s2_values

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.plot(s2_values, linewidth=2)
        self.ax.set_xlabel('Distance (r)', fontsize=12)
        self.ax.set_ylabel('S2 Value', fontsize=12)
        self.ax.set_title('SMDS (S2) Correlation Function', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)

        # Create canvas widget
        canvas = FigureCanvas(self.fig)

        # Set canvas as central widget
        self.setCentralWidget(canvas)

        # Create menu bar
        self._create_menu()

    def _create_menu(self):
        """Create the menu bar with file operations."""
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        # Save plot as image
        save_plot_action = file_menu.addAction("&Save Plot as Image...")
        save_plot_action.setShortcut("Ctrl+P")
        save_plot_action.triggered.connect(self.save_plot)

        # Export data as CSV
        export_csv_action = file_menu.addAction("&Export Data as CSV...")
        export_csv_action.setShortcut("Ctrl+E")
        export_csv_action.triggered.connect(self.export_csv)

    def save_plot(self):
        """Save the plot as PNG or JPEG image file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;PDF Document (*.pdf);;All Files (*)"
        )

        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Plot saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")

    def export_csv(self):
        """Export S2 values to CSV file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            try:
                # Ensure .csv extension
                if not file_path.endswith('.csv'):
                    file_path += '.csv'

                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['distance_r', 's2_value'])
                    for i, value in enumerate(self.s2_values):
                        writer.writerow([i, value])

                QMessageBox.information(self, "Success", f"Data exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{str(e)}")
