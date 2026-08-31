"""
Dialog for connected-components calculation settings.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QGroupBox, QFormLayout
)

class ConnectedComponentsSettingsDialog(QDialog):

    """
    Dialog to configure connected-components labeling for the current 3D
    image. Same QFormLayout numeric-dialog pattern as MinkowskiSettingsDialog,
    plus a connectivity choice and a minimum-size filter.
    """

    def __init__(self, parent = None):
        super().__init__(parent)

        # will hold the chosen values after OK - same convention as
        # MinkowskiSettingsDialog's self.resolution/self.unit

        self.connectivity = None
        self.resolution = None
        self.unit = None
        self.min_size = None

        self.setWindowTitle("Connected Components Settings")
        self.setModal(True)
        self.setMinimumWidth(360)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Label connected components of the foreground phase (value 1) "
            "in this 3D image and measure each one's volume."
        )

        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        settings_group = QGroupBox("Connected Components Settings")
        form_layout = QFormLayout()

        # Text carries the actual connectivity number so _accept() can parse
        # it straight back out - avoids a second combo -> int lookup table.
        self.connectivity_combo = QComboBox()
        self.connectivity_combo.addItems([
            "6 (face)", "18 (face + edge)", "26 (face + edge + corner)"
        ])
        self.connectivity_combo.setCurrentIndex(2) # default 26, most permissive
        self.connectivity_combo.setToolTip(
            "Neighbor rule for 'connected' - 6 shares a full face only, 18 also "
            "merges components touching along just an edge, 26 also merges "
            "components touching at only a single corner voxel."
        )
        form_layout.addRow("Connectivity:", self.connectivity_combo)

        self.resolution_spinbox = QDoubleSpinBox()
        self.resolution_spinbox.setRange(0.0001, 100000.0)
        self.resolution_spinbox.setDecimals(4)
        self.resolution_spinbox.setValue(1.0)
        self.resolution_spinbox.setToolTip(
            "Physical size of one voxel (isotropic). Scales voxel counts into "
            "physical volumes - leave at 1.0 for results in voxel units."
        )
        form_layout.addRow("Voxel size:", self.resolution_spinbox)

        self.unit_combo = QComboBox()
        self.unit_combo.addItems(['\u00b5m', 'mm', 'nm', 'voxels'])
        form_layout.addRow("Unit:", self.unit_combo)

        # Real segmented volumes are full of 1-2 voxel speckle components -
        # see the notebook's size-distribution histogram (3156 of 5902
        # components on the test volume were <= 5 voxels).
        self.min_size_spinbox = QSpinBox()
        self.min_size_spinbox.setRange(1, 10_000_000)
        self.min_size_spinbox.setValue(1)
        self.min_size_spinbox.setToolTip(
            "Components smaller than this (in voxels) are dropped from the "
            "results table."
        )
        form_layout.addRow("Minimum size (voxels):", self.min_size_spinbox)

        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)

        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self._accept)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

    def _accept(self):
        self.connectivity = int(self.connectivity_combo.currentText().split()[0])
        self.resolution = self.resolution_spinbox.value()
        self.unit = self.unit_combo.currentText()
        self.min_size = self.min_size_spinbox.value()
        self.accept()

    def get_connectivity(self):
        return self.connectivity

    def get_resolution(self):
        return self.resolution

    def get_unit(self):
        return self.unit

    def get_min_size(self):
        return self.min_size



      