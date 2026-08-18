# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ImageViewer.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6'],
    noarchive=False,
    optimize=0,
)

# Drop the VC++ runtime DLLs PyInstaller auto-collects from the conda build env.
# Tried bundling fresh copies (even taken straight from System32) instead of excluding
# them outright, expecting a version mismatch to be the issue - but ANY loose copy of
# these next to the app breaks Qt6Core.dll's startup (STATUS_STACK_BUFFER_OVERRUN),
# confirmed by testing both a stale copy and a fresh one and getting the identical
# crash either way. Excluding them entirely and relying on the target machine's own
# installed VC++ Redistributable is what actually works - confirmed by testing.
# The Inno Setup installer (next step) will make sure that redistributable is present.
_RUNTIME_DLL_NAMES = ('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll')
a.binaries = [b for b in a.binaries if b[0].lower() not in _RUNTIME_DLL_NAMES]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SMiCA',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Keep console=True - see the comment in main.py's frozen-app block for why
    # (a windowed-subsystem build crashes here; the console window is hidden at
    # runtime instead, which gets the same result without the crash).
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SMiCA',
)
