"""
Unit tests for image utility functions.
"""

import numpy as np
import pytest
from src.micro_gui.utils.image_utils import get_image_info


class TestGetImageInfo:
    """Tests for get_image_info function."""

    def test_uniform_image_ones(self):
        """Test info extraction from uniform image of ones."""
        img = np.ones((50, 50), dtype=np.uint8)
        info = get_image_info(img)

        assert info['shape'] == (50, 50)
        assert info['dtype'] == 'uint8'
        assert info['min'] == 1.0
        assert info['max'] == 1.0
        assert info['mean'] == 1.0

    def test_uniform_image_zeros(self):
        """Test info extraction from uniform image of zeros."""
        img = np.zeros((100, 100), dtype=np.float32)
        info = get_image_info(img)

        assert info['shape'] == (100, 100)
        assert info['dtype'] == 'float32'
        assert info['min'] == 0.0
        assert info['max'] == 0.0
        assert info['mean'] == 0.0

    def test_random_image(self):
        """Test info extraction from random image."""
        np.random.seed(42)  # For reproducibility
        img = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
        info = get_image_info(img)

        assert info['shape'] == (64, 64)
        assert info['dtype'] == 'uint8'
        assert 0.0 <= info['min'] <= 255.0
        assert 0.0 <= info['max'] <= 255.0
        assert 0.0 <= info['mean'] <= 255.0

    def test_3d_image(self):
        """Test info extraction from 3D image."""
        img = np.ones((10, 20, 30), dtype=np.uint16)
        info = get_image_info(img)

        assert info['shape'] == (10, 20, 30)
        assert info['dtype'] == 'uint16'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
