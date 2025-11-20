import os
import sys

# Fix Qt plugin path
if hasattr(sys, 'frozen'):
    pass
else:
    import PySide6
    plugin_path = os.path.join(os.path.dirname(PySide6.__file__), 'plugins', 'platforms')
    if os.path.exists(plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
        
import numpy as np
from PySide6.QtWidgets import QApplication
from src.micro_gui.gui.plot_window import PlotWindow


# create fake s2 data

s2_values = np.linspace(0, 0.5, 100)

app = QApplication(sys.argv)
window = PlotWindow(s2_values)
window.show()
sys.exit(app.exec())
