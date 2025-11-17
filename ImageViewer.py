import sys
import os
# Fix Qt plugin path issue
if hasattr(sys, 'frozen'):
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(sys._MEIPASS, 'PySide6', 'plugins', 'platforms')
else:
    import PySide6
    plugin_path = os.path.join(os.path.dirname(PySide6.__file__), 'plugins', 'platforms')
    if os.path.exists(plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
import numpy as np
import csv
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QFileDialog,
    QScrollArea,
    QStatusBar,
    QSlider,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QProgressBar,
)
from PySide6.QtGui import QPixmap, QAction, QImage, QMouseEvent
from PySide6.QtCore import Qt, QThread, Signal
from PIL import Image
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg backend instead of Qt5Agg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from smds import calculate_s2

try:
    from smds import calculate_s2_3d
    HAS_3D_SUPPORT = True
except ImportError:
    HAS_3D_SUPPORT = False


class CalculationThread(QThread):
    """Thread for running SMDS calculation in background"""
    finished = Signal(object)  # Emits the result
    error = Signal(str)  # Emits error message

    def __init__(self, image_data, is_3d=False):
        super().__init__()
        self.image_data = image_data
        self.is_3d = is_3d

    def run(self):
        """Run the calculation in a separate thread"""
        try:
            if self.is_3d:
                result = calculate_s2_3d(self.image_data)
            else:
                result = calculate_s2(self.image_data)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ImageDisplayWidget(QLabel):
    """Custom QLabel that tracks mouse movement and displays pixel values"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.image_data = None
        self.parent_window = parent

    def set_image_data(self, image_data):
        """Store the raw image data for pixel value lookup"""
        self.image_data = image_data

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement to display pixel values"""
        if self.image_data is not None and self.pixmap() and not self.pixmap().isNull():
            # Get mouse position relative to the label
            pos = event.pos()

            # Get the pixmap dimensions
            pixmap_rect = self.pixmap().rect()
            label_rect = self.rect()

            # Calculate the offset (centering)
            x_offset = (label_rect.width() - pixmap_rect.width()) // 2
            y_offset = (label_rect.height() - pixmap_rect.height()) // 2

            # Adjust mouse position relative to the pixmap
            adjusted_x = pos.x() - x_offset
            adjusted_y = pos.y() - y_offset

            # Check if mouse is within the pixmap bounds
            if 0 <= adjusted_x < pixmap_rect.width() and 0 <= adjusted_y < pixmap_rect.height():
                # Calculate scaling factors
                scale_x = self.image_data.shape[1] / pixmap_rect.width()
                scale_y = self.image_data.shape[0] / pixmap_rect.height()

                # Get the actual pixel coordinates in the original image
                pixel_x = int(adjusted_x * scale_x)
                pixel_y = int(adjusted_y * scale_y)

                # Ensure we're within bounds
                if 0 <= pixel_y < self.image_data.shape[0] and 0 <= pixel_x < self.image_data.shape[1]:
                    pixel_value = self.image_data[pixel_y, pixel_x]
                    if self.parent_window:
                        self.parent_window.update_status(
                            f"Position: ({pixel_x}, {pixel_y}) | Pixel Value: {pixel_value}"
                        )
            else:
                if self.parent_window:
                    self.parent_window.update_status("")

        super().mouseMoveEvent(event)


class PlotWindow(QMainWindow):
    """Separate window for displaying matplotlib plots with save options"""

    def __init__(self, s2_values, parent=None):
        super().__init__(parent)

        self.setWindowTitle("SMDS Calculation Results")
        self.setGeometry(200, 200, 800, 600)
        self.s2_values = s2_values

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.plot(s2_values, linewidth=2)
        self.ax.set_xlabel('Index')
        self.ax.set_ylabel('S2 Value')
        self.ax.set_title('SMDS (S2) Calculation Results')
        self.ax.grid(True, alpha=0.3)

        # Create canvas widget
        canvas = FigureCanvas(self.fig)

        # Set canvas as central widget
        self.setCentralWidget(canvas)

        # Create menu bar
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
        """Save the plot as PNG or JPG"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)"
        )

        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Plot saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")

    def export_csv(self):
        """Export S2 values to CSV file"""
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
                    writer.writerow(['r', 's2'])
                    for i, value in enumerate(self.s2_values):
                        writer.writerow([i, value])
                QMessageBox.information(self, "Success", f"Data exported to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{str(e)}")


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Viewer - Main Window")
        self.setGeometry(100, 100, 800, 600)

        # Store opened windows and image data
        self.plot_windows = []
        self.current_image_data = None  # Full data (can be 3D)
        self.current_file_path = None
        self.current_slice_index = 0

        # Create central widget with layout
        central_widget = QWidget()
        self.layout = QVBoxLayout()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        # Create image display widget
        self.image_label = ImageDisplayWidget(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText(
            "No image loaded.\n\n"
            "Use File > Open Image to load a TIF image."
        )
        self.image_label.setWordWrap(True)

        # Create scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        self.layout.addWidget(scroll_area)

        # Create slider for 3D images (initially hidden)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        self.slice_slider.setVisible(False)
        self.layout.addWidget(self.slice_slider)

        # Slider label
        self.slice_label = QLabel("Slice: 0 / 0")
        self.slice_label.setAlignment(Qt.AlignCenter)
        self.slice_label.setVisible(False)
        self.layout.addWidget(self.slice_label)

        # Store current pixmap
        self.current_pixmap = None

        # Create menu bar
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # Create "Open Image" action
        open_action = file_menu.addAction("&Open Image")
        open_action.setShortcut("Ctrl+O")
        open_action.setStatusTip("Open a TIF image file")
        open_action.triggered.connect(self.open_image)

        # Create "Exit" action
        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)

        # Calculate menu
        calculate_menu = menubar.addMenu("&Calculate")

        # Create "Calculate SMDS" action
        smds_action = calculate_menu.addAction("Calculate &SMDS")
        smds_action.setShortcut("Ctrl+S")
        smds_action.setStatusTip("Calculate SMDS (S2) from the current image")
        smds_action.triggered.connect(self.calculate_smds)

        # Status bar with progress bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Create progress bar for status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(0)  # Indeterminate progress
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximumWidth(150)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.status_bar.showMessage("Ready")

    def validate_binary_image(self, image_data):
        """Validate that image contains only 0 and 1 values"""
        unique_values = np.unique(image_data)
        if not np.all(np.isin(unique_values, [0, 1])):
            return False, unique_values
        return True, unique_values

    def numpy_to_qpixmap(self, image_data):
        """Convert numpy array to QPixmap (binary: 1=white, 0=black)"""
        # For binary images, multiply by 255 to get full white/black contrast
        if image_data.dtype in [np.uint8, np.uint16]:
            # Multiply by 255 to make 1 -> 255 (white) and 0 -> 0 (black)
            normalized = (image_data * 255).astype(np.uint8)
        else:
            normalized = image_data.astype(np.uint8) * 255

        height, width = normalized.shape
        bytes_per_line = width

        # Create QImage from numpy array
        q_image = QImage(
            normalized.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_Grayscale8
        )

        return QPixmap.fromImage(q_image)

    def display_current_slice(self):
        """Display the current slice of the image"""
        if self.current_image_data is None:
            return

        # Get the current 2D slice
        if self.current_image_data.ndim == 3:
            current_2d = self.current_image_data[self.current_slice_index, :, :]
        else:
            current_2d = self.current_image_data

        # Update the image label data for hover
        self.image_label.set_image_data(current_2d)

        # Convert to pixmap and display
        self.current_pixmap = self.numpy_to_qpixmap(current_2d)
        scaled_pixmap = self.current_pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def on_slice_changed(self, value):
        """Handle slice slider change"""
        self.current_slice_index = value
        if self.current_image_data is not None and self.current_image_data.ndim == 3:
            self.slice_label.setText(f"Slice: {value} / {self.current_image_data.shape[0] - 1}")
            self.display_current_slice()

    def open_image(self):
        """Open a file dialog to select a TIF image and display it"""
        # Open file dialog to select image
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open TIF Image",
            "",
            "TIF Files (*.tif *.tiff);;All Files (*)"
        )

        # If a file was selected, load and display it
        if file_path:
            try:
                # Load TIF image using PIL
                pil_image = Image.open(file_path)

                # Check if it's a multi-page TIFF (3D image)
                image_stack = []
                try:
                    while True:
                        image_stack.append(np.array(pil_image))
                        pil_image.seek(pil_image.tell() + 1)
                except EOFError:
                    pass  # End of frames

                # Convert to numpy array
                if len(image_stack) > 1:
                    # 3D image
                    image_data = np.stack(image_stack, axis=0)
                else:
                    # 2D image
                    image_data = image_stack[0]

                # Validate binary image
                is_valid, unique_values = self.validate_binary_image(image_data)
                if not is_valid:
                    QMessageBox.critical(
                        self,
                        "Invalid Image",
                        f"Error: Image must contain only binary values (0 and 1).\n\n"
                        f"Found values: {unique_values}"
                    )
                    return

                # Store current image data
                self.current_image_data = image_data
                self.current_file_path = file_path
                self.current_slice_index = 0

                # Configure slider for 3D images
                if image_data.ndim == 3:
                    num_slices = image_data.shape[0]
                    self.slice_slider.setMaximum(num_slices - 1)
                    self.slice_slider.setValue(0)
                    self.slice_slider.setVisible(True)
                    self.slice_label.setText(f"Slice: 0 / {num_slices - 1}")
                    self.slice_label.setVisible(True)
                    dimension_info = f"3D Image: {image_data.shape[0]} slices x {image_data.shape[1]} x {image_data.shape[2]}"
                else:
                    self.slice_slider.setVisible(False)
                    self.slice_label.setVisible(False)
                    dimension_info = f"2D Image: {image_data.shape[1]} x {image_data.shape[0]}"

                # Display the first slice
                self.display_current_slice()

                # Update status
                bit_depth = "16-bit" if image_data.dtype == np.uint16 else "8-bit"
                self.status_bar.showMessage(f"Loaded: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
                self.status_bar.showMessage(f"Error: {str(e)}")

    def calculate_smds(self):
        """Calculate SMDS using the appropriate function and display results"""
        if self.current_image_data is None:
            self.status_bar.showMessage("No image loaded. Please open an image first.")
            QMessageBox.warning(self, "No Image", "Please open an image first.")
            return

        # Check if 3D and if 3D support exists
        is_3d = self.current_image_data.ndim == 3
        if is_3d and not HAS_3D_SUPPORT:
            QMessageBox.warning(
                self,
                "3D Not Supported",
                "3D SMDS calculation is not available.\n\n"
                "Please add the 'calculate_s2_3d' function to smds.py"
            )
            self.status_bar.showMessage("3D calculation not supported")
            return

        # Show progress bar in status bar
        self.progress_bar.setVisible(True)
        self.status_bar.showMessage("Calculating SMDS...")

        # Process events to ensure UI updates
        QApplication.processEvents()

        # Create and start calculation thread
        self.calc_thread = CalculationThread(self.current_image_data, is_3d)
        self.calc_thread.finished.connect(self.on_calculation_finished)
        self.calc_thread.error.connect(self.on_calculation_error)
        self.calc_thread.start()

    def on_calculation_finished(self, s2_values):
        """Handle completion of SMDS calculation"""
        self.progress_bar.setVisible(False)

        # Create a new window to display the plot
        plot_window = PlotWindow(s2_values, self)
        self.plot_windows.append(plot_window)
        plot_window.show()

        self.status_bar.showMessage("SMDS calculation completed")

    def on_calculation_error(self, error_msg):
        """Handle error during SMDS calculation"""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Calculation Error", f"Error calculating SMDS:\n{error_msg}")
        self.status_bar.showMessage(f"Error: {error_msg}")

    def resizeEvent(self, event):
        """Handle window resize"""
        super().resizeEvent(event)
        if self.current_pixmap and not self.current_pixmap.isNull():
            self.display_current_slice()

    def update_status(self, message):
        """Update the status bar message"""
        self.status_bar.showMessage(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageViewer()
    window.show()
    sys.exit(app.exec())
