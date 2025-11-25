"""
Test file for REV plot window.
Creates fake REV data to test plotting without running actual calculations.

Usage:
    python tests/test_rev_plot_window.py

This allows you to test:
- REV plot window display (S2 and F2 tabs)
- Saving plots functionality
- Without waiting for the full REV calculation
"""

import sys
import os
import numpy as np
import pandas as pd

# Fix Qt plugin path (needed for running standalone)
if hasattr(sys, 'frozen'):
    pass
else:
    import PySide6
    plugin_path = os.path.join(os.path.dirname(PySide6.__file__), 'plugins', 'platforms')
    if os.path.exists(plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

from PySide6.QtWidgets import QApplication
from src.micro_gui.gui.rev_plot_window import RevPlotWindow


def create_fake_rev_data():
    """
    Create fake S2 and F2 data that matches the structure from REV calculation.

    Returns:
        tuple: (s2_dict, f2_dict) with the same structure as REV() function

    Structure:
        s2_dict = {
            'original': numpy_array,      # Full image S2 curve
            '32': DataFrame,              # 32x32x32 subvolume statistics
            '64': DataFrame,              # 64x64x64 subvolume statistics
            '128': DataFrame              # 128x128x128 subvolume statistics
        }

    Each DataFrame has MultiIndex columns:
        ('s2', 'mean'), ('s2', 'std'), ('s2', 'size')
    """

    print("Generating fake REV data...")

    # Define subvolume sizes to test
    sizes = [32, 64, 128]
    num_points = 50  # Number of distance (r) values

    # Initialize dictionaries
    s2_dict = {}
    f2_dict = {}

    # Create distance array
    r = np.arange(num_points)

    # ===== Create "original" image curves =====
    # Simulate realistic decay: high correlation at small r, decays with distance
    s2_original = 0.85 * np.exp(-r / 30) + 0.15  # Decays from ~0.85 to ~0.15
    f2_original = 0.65 * np.exp(-r / 25) + 0.10  # Decays from ~0.65 to ~0.10

    s2_dict['original'] = s2_original
    f2_dict['original'] = f2_original

    print(f"  Original curves: {num_points} points")

    # ===== Create data for each subvolume size =====
    for size in sizes:
        print(f"  Creating data for {size}³ subvolumes...")

        # Smaller subvolumes have more noise (statistical variation)
        # Larger subvolumes converge to the original curve
        noise_level = 0.08 / (size / 32)  # Decreases with size

        # Simulate multiple random samples (as REV does)
        n_samples = 30
        s2_samples = []
        f2_samples = []

        for i in range(n_samples):
            # Add random Gaussian noise to simulate variation between subvolumes
            noise_s2 = np.random.normal(0, noise_level, num_points)
            noise_f2 = np.random.normal(0, noise_level * 0.8, num_points)

            s2_sample = s2_original + noise_s2
            f2_sample = f2_original + noise_f2

            # Ensure values stay in valid range [0, 1]
            s2_sample = np.clip(s2_sample, 0, 1)
            f2_sample = np.clip(f2_sample, 0, 1)

            s2_samples.append(s2_sample)
            f2_samples.append(f2_sample)

        # ===== Create DataFrames in the format REV returns =====
        # Build list of dicts for each (r, value) pair across all samples
        s2_data = []
        f2_data = []

        for sample_idx, (s2_sample, f2_sample) in enumerate(zip(s2_samples, f2_samples)):
            for r_val, s2_val, f2_val in zip(r, s2_sample, f2_sample):
                s2_data.append({'r': r_val, 's2': s2_val})
                f2_data.append({'r': r_val, 'f2': f2_val})

        # Create DataFrames
        df_s2 = pd.DataFrame(s2_data)
        df_f2 = pd.DataFrame(f2_data)

        # Group by r and calculate mean, std, and count
        # This creates MultiIndex columns: ('s2', 'mean'), ('s2', 'std'), ('s2', 'size')
        df_s2_grouped = df_s2.groupby(['r']).agg({'s2': ['mean', 'std', 'size']})
        df_f2_grouped = df_f2.groupby(['r']).agg({'f2': ['mean', 'std', 'size']})

        # Store with size as string key (matching REV function behavior)
        s2_dict[str(size)] = df_s2_grouped
        f2_dict[str(size)] = df_f2_grouped

    print("Fake data generation complete!")
    return s2_dict, f2_dict


def print_data_structure(s2_dict):
    """Helper function to show the data structure."""
    print("\n" + "="*60)
    print("DATA STRUCTURE VERIFICATION")
    print("="*60)

    print(f"\nDictionary keys: {list(s2_dict.keys())}")

    print("\n--- 'original' entry (numpy array) ---")
    print(f"Type: {type(s2_dict['original'])}")
    print(f"Shape: {s2_dict['original'].shape}")
    print(f"First 5 values: {s2_dict['original'][:5]}")

    print("\n--- '32' entry (DataFrame with statistics) ---")
    print(f"Type: {type(s2_dict['32'])}")
    print(f"Shape: {s2_dict['32'].shape}")
    print(f"Columns: {s2_dict['32'].columns.tolist()}")
    print("\nFirst few rows:")
    print(s2_dict['32'].head())

    print("\n--- Accessing mean values ---")
    print(f"s2_dict['32']['s2']['mean'] type: {type(s2_dict['32']['s2']['mean'])}")
    print(f"First 5 mean values: {s2_dict['32']['s2']['mean'].values[:5]}")
    print("="*60 + "\n")


def test_rev_plot_window():
    """Main test function - creates window with fake data."""

    print("\n" + "="*60)
    print("REV PLOT WINDOW TEST")
    print("="*60)

    # Step 1: Generate fake data
    s2_dict, f2_dict = create_fake_rev_data()

    # Step 2: Verify data structure
    print_data_structure(s2_dict)

    # Step 3: Create Qt application
    print("Creating Qt application...")
    app = QApplication(sys.argv)

    # Step 4: Create and show the plot window
    print("Opening REV plot window...")
    print("\nINSTRUCTIONS:")
    print("  - The window has two tabs: $S_2$ and $F_2$")
    print("  - Switch between tabs to see both plots")
    print("  - Try File > Save Plots (Ctrl+S) to test save functionality")
    print("  - Close the window when done testing")
    print()

    try:
        window = RevPlotWindow(s2_dict, f2_dict)
        window.show()

        print("✓ Window displayed successfully!")
        print("  Waiting for user to close window...\n")

        # Run the application event loop
        sys.exit(app.exec())

    except Exception as e:
        print(f"\n✗ ERROR creating window:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_rev_plot_window()
