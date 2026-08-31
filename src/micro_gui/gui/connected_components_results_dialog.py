"""
Results dialog for connected-components output: one row per component.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QPushButton, QFileDialog, QMessageBox
)

from .save_dialog_helper import suggested_save_path, remember_save_dir


class ConnectedComponentsResultsDialog(QDialog):
    """
    Read-only results table for a connected-components calculation - one row
    per component, already sorted largest-first by the analysis function.
    Not modal (shown with .show(), not .exec()) so the user can keep it open
    while comparing against the image or running another calculation.
    """

    def __init__(self, table, unit: str, is_3d: bool, total_components:int, parent = None):
        super().__init__(parent)

        # Keep the DataFrame itself (not just what ends up in the table
        # widget) - _export_csv writes this straight to disk later.

        self.table_data = table
        # connected_components_2d/_3d name their columns differently
        # (pixel_count/area vs voxel_count/volume) - is_3d picks which pair
        # of column names and display labels this dialog should use.

        self.count_col = 'voxel_count' if is_3d else 'pixel_count'
        self.measure_col = 'volume' if is_3d else 'area'
        count_label = 'Voxel count' if is_3d else 'Pixel count'
        measure_label = f"Volume ({unit}\u00b3)" if is_3d else f"Area ({unit}\u00b2)"

        self.setWindowTitle("Connected Components Results")
        self.setMinimumWidth(480)
        self._setup_ui(total_components, count_label, measure_label)

    def _setup_ui(self, total_components, count_label, measure_label):
        layout = QVBoxLayout(self)

        # total_components is the raw count BEFORE any min-size filtering;
        # self.table_data is already the filtered table. Showing both makes
        # it obvious how much a filter removed, e.g.
        # "5902 components found - 2746 shown below".

        shown = len(self.table_data)
        summary = QLabel(f"{total_components} components found - {shown} shown after filtering below")
        layout.addWidget(summary)

        # 3 columns: Label, raw element count, physical measure. Row count is
        # fixed up front since we already know exactly how many rows there'll b
        table = QTableWidget(shown, 3)
        table.setHorizontalHeaderLabels(['Label', count_label, measure_label])
        table.horizontalHeader().setStretchLastSection(True) # last column fills leftover width
        table.verticalHeader().setVisible(False)               # no numbered row headers needed
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # read-only results, not an editable grid

        # itertuples() is faster than iterrows() for looping a DataFrame;
        # ._asdict() turns each row into a plain dict so columns can be
        # looked up by name (self.count_col/self.measure_col) instead of a
        # fixed attribute - needed since which column is "the measure"
        # depends on is_3d.
        for row, record in enumerate(self.table_data.itertuples(index=False)):
            record = record._asdict()  # convert namedtuple to dict for easier access by column name
            table.setItem(row, 0, QTableWidgetItem(str(int(record['label']))))
            # counts come back as float64 (regionprops_table's default) but
            # are always whole numbers, so format with no decimal places.
            table.setItem(row, 1, QTableWidgetItem(f"{record[self.count_col]:.0f}"))
            table.setItem(row, 2, QTableWidgetItem(f"{record[self.measure_col]:.6g}"))

        self.table = table
        layout.addWidget(table)

        # Save/Close buttons, right-aligned via the stretch placed before them.
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save to CSV...")
        save_button.clicked.connect(self._export_csv)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)

        button_layout.addStretch()
        button_layout.addWidget(save_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)


    def _export_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV",
            suggested_save_path("connected_components"),  # remembers/suggests the last-used save folder
            "CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return  # user cancelled the save dialog
        if not file_path.endswith('.csv'):
            file_path += '.csv'

        try:
            # table_data is already a DataFrame - pandas writes the CSV
            # directly, no need to hand-loop rows with csv.writer.
            # index=False: don't also write pandas' own 0..N row index.
            self.table_data.to_csv(file_path, index=False)
            remember_save_dir(file_path)  # so the next Save dialog opens in this same folder
            QMessageBox.information(self, "Success", f"Data exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{str(e)}")
