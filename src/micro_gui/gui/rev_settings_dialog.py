"""
Dialog for REV calculation settings.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QSpinBox, QPushButton, QMessageBox,
    QGroupBox, QFormLayout
)

from PySide6.QtCore import Qt

# the class Inherits from QDialog, which provides:
# - Window frame
# - Modal behavior (blocks parent)
# - Accept/reject mechanisms
# - Built-in exec() method

class REVSettingsDialog(QDialog):
     
     """
    Dialog to configure REV calculation parameters.
    
    Allows users to specify:
    - Subvolume sizes (comma-separated list)
    - Number of random samples
    """
     
     def __init__(self, max_size, calc_name = "REV",parent=None):
          
          """
          Initialize the dialog.
        
         Args:
            max_size: Maximum allowed subvolume size (based on image dimensions)
            parent: Parent window
          """


          super().__init__(parent)
          # store parameters
          self.max_size = max_size
          self.img_size_list = None # Will store user input
          self.n_rand_samples = None # Will store user input

          # Set up the dialog
          self.setWindowTitle(F"{calc_name} Calculation Settings")
          self.setModal(True) # Block interaction with main window, blocks main window, User MUST respond before using main window.
          self.setMinimumWidth(400) 

          # Create the UI, builds the visual interface
          self._setup_ui()
     
     def _setup_ui(self):
          """ Set up the dialog UI. """

          layout = QVBoxLayout(self)

          # info label

          info_label = QLabel(
              f"Configure the REV analysis parameters.\n"
              f"Maximum subvolume size: {self.max_size}"
              )
          
          info_label.setWordWrap(True) # Wrap long text to multiple lines
          layout.addWidget(info_label)

          # Settings group (the form)
          settings_group = QGroupBox("REV Settings") # Creates box with title
          form_layout = QFormLayout() # Two-column layout

          # Subvolume sizes input
          self.sizes_input = QLineEdit()
          self.sizes_input.setPlaceholderText("e.g. 32, 64, 128") # Gray hint text (when empty)
          self.sizes_input.setText("32,64,128") # Default values
          form_layout.addRow("Subvolume Sizes:", self.sizes_input) # Label + Field

          # Number of random samples input
          self.samples_spinbox = QSpinBox()
          self.samples_spinbox.setRange(1, 100)
          self.samples_spinbox.setValue(30) # Default value
          ## Hover text (shows tip when mouse hovers)
          self.samples_spinbox.setToolTip("Number of random samples per subvolume size")
          form_layout.addRow("Number of samples:", self.samples_spinbox)

          settings_group.setLayout(form_layout) # Put form inside groupbox
          layout.addWidget(settings_group) # Add groupbox to main layout

          # Buttons
          button_layout = QHBoxLayout()
          self.ok_button = QPushButton("OK")
          self.ok_button.clicked.connect(self._validate_and_accept)

          self.cancel_button = QPushButton("Cancel")
          self.cancel_button.clicked.connect(self.reject) # ← Built-in method

          button_layout.addStretch() # Pushes buttons to the right
          button_layout.addWidget(self.ok_button)
          button_layout.addWidget(self.cancel_button)

          layout.addLayout(button_layout) # Add to main layout

     def _validate_and_accept(self):
          """ Validate inputs and accept the dialog if valid."""

          # parse subvolume sizes
          sizes_text = self.sizes_input.text().strip() # Get text, remove spaces

          try:
               sizes = [int(s.strip()) for s in sizes_text.split(',')]

               # Validate each size
               invalid_sizes = [s for s in sizes if s<=0 or s>= self.max_size]
               
               if invalid_sizes:
                    QMessageBox.warning(
                         self,
                         "Invalid Sizes",
                         f"Sizes must be between 1 and {self.max_size - 1}.\n"
                         f"Invalid values: {invalid_sizes}"
                    )
                    return # ← DON'T close dialog, let user fix it
               
               
               # Remove duplicates and sort
               sizes = sorted(set(sizes)) # set() removes duplicates, sorted() sorts

               if not sizes: # If list is empty
                    QMessageBox.warning(
                         self,
                         "No Sizes",
                         "Please enter at least one  size."
                    )
                    return
               
               self.img_size_list = sizes
               self.n_rand_samples = self.samples_spinbox.value() # ← Get spinbox value

               self.accept() # ← Close dialog with "Accepted" status
               #Sets dialog result to QDialog.Accepted
                # Closes the dialog window
                # Returns control to calling code
                # dialog.exec() returns QDialog.Accepted

          except ValueError:
               QMessageBox.warning(
                    self,
                    "Invalid format",
                    "Please enter sizes as comma-separated numbers.\n"
                    "Example: 32, 64, 128"
               )
     def get_values(self):
          """
          Get the configured values.
          
          Reurns:
          
          Tuple of (img_size_list, n_rand_samples) or (None, None) if cancelled.
          """
          return self.img_size_list, self.n_rand_samples
     




        
     
     
     

    


          



