"""
Main entry point for the Micro_GUI application.
"""

import sys
import os

# Fix Qt plugin path issue for frozen applications
if hasattr(sys, 'frozen'):
    _pyside6_dir = os.path.join(sys._MEIPASS, 'PySide6')
    # Put our WHOLE bundle (sys._MEIPASS, not just the PySide6 subfolder) at the FRONT
    # of PATH, so every DLL we ship - Qt6Core.dll in PySide6/, but also the VC++ runtime
    # DLLs which live at the bundle root - wins over anything of the same name already
    # on PATH (e.g. a conda env's Library/bin: during testing this shadowed both our
    # PySide6 build with a different Qt version there, AND our runtime DLLs with older
    # ones there, each producing a different failure once the other was fixed).
    # os.add_dll_directory() alone does NOT reliably fix this: it only takes priority
    # over PATH for loads that opt into the "safe" search mode, and PyInstaller's
    # bootloader doesn't enable that as the process-wide default the way a normal
    # python.exe does - so these DLLs' implicit dependencies still resolve via plain
    # PATH search, which prepending to PATH directly does fix.
    os.environ['PATH'] = (
        _pyside6_dir + os.pathsep + sys._MEIPASS + os.pathsep + os.environ.get('PATH', '')
    )
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(_pyside6_dir)
        os.add_dll_directory(sys._MEIPASS)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(
        _pyside6_dir, 'plugins', 'platforms'
    )

    # Detach from the console window instead of building with console=False (a
    # windowed-subsystem exe). Confirmed by direct A/B test: with the VC++ runtime DLLs
    # excluded from the bundle (required - see SMiCA.spec), a windowed-subsystem build
    # crashes (STATUS_STACK_BUFFER_OVERRUN in Qt6Core.dll) on startup, while a
    # console-subsystem build with its console detached at runtime does not. Keep the
    # .spec on console=True. FreeConsole() (rather than hiding the console window via
    # its HWND) avoids a ctypes pitfall - GetConsoleWindow() returns a 64-bit handle,
    # and without an explicit restype ctypes truncates it to 32 bits, so ShowWindow()
    # was acting on a corrupted handle: the console would flash, then behave as if
    # only half-hidden (minimized but still present, closing it killed the app).
    import ctypes
    ctypes.windll.kernel32.FreeConsole()
else:
    import PySide6
    plugin_path = os.path.join(os.path.dirname(PySide6.__file__), 'plugins', 'platforms')
    if os.path.exists(plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

from PySide6.QtWidgets import QApplication
from .gui.image_viewer import ImageViewer


def main():
    """
    Main entry point for the Micro_GUI application.

    This function initializes the Qt application and displays
    the main image viewer window.
    """
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("SMiCA")
    app.setOrganizationName("Microstructure Analysis")

    window = ImageViewer()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
