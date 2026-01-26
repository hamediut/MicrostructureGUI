# SMiCA: Statistical Microstructure Characterisation & Analysis

A professional GUI application for analyzing microstructure images using correlation functions and statistical analysis methods, specifically designed for materials science and microstructure characterization.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-alpha-orange)

## Features

### Current Features ✅

- **Binary Image Analysis**: Load and analyze binary TIF/TIFF images (2D and 3D)
- **3D Volume Support**: Navigate through multi-page TIFF files with slice-by-slice viewing
- **SMDS Calculations**:
  - Two-point correlation function (S2) and scaled autocovariance (F2) computation in 2D and 3D.
  - Optimized with Numba JIT compilation for performance
- **Representative Elementary Size**:
  - Representative Elementary Size analysis in 2D and 3D images using S2 and F2 functions ([Amiri et al., 2024](https://doi.org/10.1029/2024JH000178)).
- **Interactive Visualization**:
  - Real-time pixel value display on mouse hover
  - Coordinate tracking
  - Matplotlib-based result plotting
- **Data Export**:
  - Save correlation plots as PNG, JPEG, or PDF
  - Export S2 values to CSV for custom plotting and further analysis
- **Professional UI**:
  - PySide6-based Qt interface
  - Progress indicators for long calculations
  - Thread-based background processing (non-blocking UI)

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda (Anaconda or Miniconda) recommended

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/hamediut/SMiCA.git
cd Micro_GUI

# Create and activate conda environment
conda create -n gui_micro python=3.10
conda activate gui_micro

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 2: Using pip with virtual environment

```bash
# Clone the repository
git clone https://github.com/hamediut/SMiCA.git
cd Micro_GUI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Running the Application

#### Method 1: Run the main script
```bash
python ImageViewer.py
```

#### Method 2: Run as installed package (after `pip install -e .`)
```bash
micro-gui
```

#### Method 3: Run from source
```bash
python -m src.micro_gui.main
```

### Basic Workflow

1. **Load an Image**:
   - Click `File > Open Image` (or press `Ctrl+O`)
   - Select a binary TIF/TIFF file (values must be 0 and 1)
   - For 3D images, use the slider to navigate through slices

2. **Calculate SMDS**:
   - Click `Calculate > Calculate SMDS` (or press `Ctrl+S`)
   - Wait for the calculation to complete
   - A new window will display the S2 correlation function plot

3. **Export Results**:
   - In the plot window, use `File > Save Plot as Image` (`Ctrl+P`)
   - Or `File > Export Data as CSV` (`Ctrl+E`) for raw data

### Image Requirements

- **Format**: TIF or TIFF (supports multi-page TIFF for 3D volumes)
- **Values**: Binary images with pixel values of 0 and 1 only
- **Bit depth**: 8-bit or 16-bit grayscale

## Project Structure

```
Micro_GUI/
├── src/
│   └── micro_gui/              # Main package
│       ├── __init__.py
│       ├── main.py             # Entry point
│       ├── gui/                # GUI components
│       │   ├── __init__.py
│       │   ├── image_viewer.py # Main window
│       │   ├── plot_window.py  # Plot display
│       │   └── widgets.py      # Custom widgets
│       ├── analysis/           # Analysis algorithms
│       │   ├── __init__.py
│       │   └── smds.py         # SMDS calculations
│       └── utils/              # Utility functions
│           └── __init__.py
├── tests/                      # Unit tests
│   ├── __init__.py
│   └── test_smds.py
├── docs/                       # Documentation
├── examples/                   # Example images and scripts
├── ImageViewer.py              # Main entry point (backward compatible)
├── smds.py                     # Legacy SMDS module (for compatibility)
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation script
├── .gitignore
└── README.md
```

## Development

### Setting Up for Development

```bash
# Install development dependencies
pip install -r requirements.txt
# Uncomment dev dependencies in requirements.txt first

# Install pre-commit hooks (optional)
# pip install pre-commit
# pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/micro_gui tests/

# Run specific test file
pytest tests/test_smds.py -v
```

### Code Style

This project follows PEP 8 style guidelines. Use `black` for formatting:

```bash
black src/ tests/
```

## Algorithm Details

### SMDs (Statistical Microstructure Descriptors)

The application calculates two-point correlation functions S₂(r), which measure the probability that two points separated by distance r both belong to the same phase in a binary microstructure.

**For 2D images**:
- Calculates correlations in x and y directions
- Averages results for isotropic measure
- Returns values for r = 0 to r_max (half of image size)

**For 3D volumes**:
- Calculates correlations in x, y, and z directions
- Averages all three directions
- Returns values for r = 0 to r_max (half of minimum dimension)

**Performance**:
- JIT-compiled with Numba for near-C performance
- Background threading prevents UI freezing
- Typical processing time: ~1-10 seconds for 256³ volumes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Roadmap

### Planned Features (Phase 2-4)

- [ ] Additional correlation functions (3-point, lineal path)
- [ ] Batch processing for multiple images
- [ ] Image preprocessing tools (filtering, thresholding)
- [ ] Statistical analysis and comparison tools
- [ ] Project save/load functionality
- [ ] Plugin architecture for custom analysis methods
- [ ] GPU acceleration for large volumes
- [ ] Executable distribution (standalone .exe)

## Citation

If you use this software in your research, please cite:

```bibtex
@software{smica_2025,
  title = {SMiCA: Statistical Microstructure Characterisation & Analysis},
  author = {Hamed Amiri},
  year = {2025},
  url = {https://github.com/hamediut/SMiCA}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [PySide6](https://wiki.qt.io/Qt_for_Python) for the GUI
- Uses [Numba](https://numba.pydata.org/) for performance optimization
- Visualization powered by [Matplotlib](https://matplotlib.org/)

## Contact

Hamed Amiri - amiiri.hamed@gmail.com

Project Link: [https://github.com/hamediut/SMiCA](https://github.com/hamediut/SMiCA)

## Troubleshooting

### Common Issues

**Q: "conda is not recognized"**
- Use Anaconda Prompt instead of regular terminal, or
- Initialize conda in your shell: `conda init powershell` (Windows) or `conda init bash` (Linux/Mac)

**Q: "Image must contain only binary values (0 and 1)"**
- Your image needs preprocessing to convert to binary
- Segment your images into binary values (1 for your feature of interest)

**Q: Application is slow/freezing**
- For very large 3D volumes (>512³), calculations may take several minutes
- The UI should remain responsive due to background threading
- Consider downsampling very large datasets

**Q: Import errors when running**
- Make sure you've activated your conda environment: `conda activate gui_micro`
- Verify all dependencies are installed: `pip install -r requirements.txt`

## Version History

- **v0.1.0** (2025-01-17) - Initial alpha release
  - Basic 2D/3D image loading
  - SMDS calculation
  - Plot visualization and export

---

**Note**: This is an alpha version under active development. APIs and features may change.
