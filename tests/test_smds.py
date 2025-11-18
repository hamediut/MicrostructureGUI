"""
Unit tests for SMDS calculation functions.
"""

import numpy as np
import pytest
from src.micro_gui.analysis.smds import (
    two_point_correlation,
    two_point_correlation3D,
    calculate_s2,
    calculate_s2_3d
)


class TestTwoPointCorrelation:
    """Tests for 2D two-point correlation function."""

    def test_uniform_image_all_ones(self):
        """Test correlation on uniform image of all ones."""
        img = np.ones((50, 50), dtype=np.uint8)
        result_x = two_point_correlation(img, dim=0, var=1)
        result_y = two_point_correlation(img, dim=1, var=1)

        # For uniform image of all 1s, correlation should be 1 everywhere
        assert np.allclose(result_x, 1.0)
        assert np.allclose(result_y, 1.0)

    def test_uniform_image_all_zeros(self):
        """Test correlation on uniform image of all zeros."""
        img = np.zeros((50, 50), dtype=np.uint8)
        result_x = two_point_correlation(img, dim=0, var=1)
        result_y = two_point_correlation(img, dim=1, var=1)

        # For uniform image of all 0s with var=1, correlation should be 0
        assert np.allclose(result_x, 0.0)
        assert np.allclose(result_y, 0.0)

    def test_output_shape(self):
        """Test that output shape matches input dimensions."""
        img = np.random.randint(0, 2, size=(100, 80), dtype=np.uint8)
        result_x = two_point_correlation(img, dim=0, var=1)
        result_y = two_point_correlation(img, dim=1, var=1)

        # Shape should be (y_dim, x_dim) for dim=0
        assert result_x.shape == (80, 100)
        # Shape should be (x_dim, y_dim) for dim=1
        assert result_y.shape == (100, 80)


class TestTwoPointCorrelation3D:
    """Tests for 3D two-point correlation function."""

    def test_uniform_3d_image_all_ones(self):
        """Test 3D correlation on uniform volume of all ones."""
        img = np.ones((20, 20, 20), dtype=np.uint8)
        result = two_point_correlation3D(img, dim=0, var=1)

        # For uniform image of all 1s, correlation should be 1 everywhere
        assert np.allclose(result, 1.0)

    def test_uniform_3d_image_all_zeros(self):
        """Test 3D correlation on uniform volume of all zeros."""
        img = np.zeros((20, 20, 20), dtype=np.uint8)
        result = two_point_correlation3D(img, dim=0, var=1)

        # For uniform image of all 0s with var=1, correlation should be 0
        assert np.allclose(result, 0.0)


class TestCalculateS2:
    """Tests for 2D S2 calculation."""

    def test_calculate_s2_output_size(self):
        """Test that S2 output is approximately half the input size."""
        img = np.random.randint(0, 2, size=(100, 100), dtype=np.uint8)
        result = calculate_s2(img)

        # Result should be approximately half of input dimension
        assert result.shape[0] == 50

    def test_calculate_s2_range(self):
        """Test that S2 values are in valid range [0, 1]."""
        img = np.random.randint(0, 2, size=(50, 50), dtype=np.uint8)
        result = calculate_s2(img)

        # All S2 values should be between 0 and 1
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_calculate_s2_uniform_ones(self):
        """Test S2 on uniform image."""
        img = np.ones((50, 50), dtype=np.uint8)
        result = calculate_s2(img)

        # For uniform image of 1s, S2 should be 1
        assert np.allclose(result, 1.0)


class TestCalculateS2_3D:
    """Tests for 3D S2 calculation."""

    def test_calculate_s2_3d_output_size(self):
        """Test that 3D S2 output is approximately half the minimum dimension."""
        img = np.random.randint(0, 2, size=(40, 40, 40), dtype=np.uint8)
        result = calculate_s2_3d(img)

        # Result should be half of minimum dimension
        assert result.shape[0] == 20

    def test_calculate_s2_3d_range(self):
        """Test that 3D S2 values are in valid range [0, 1]."""
        img = np.random.randint(0, 2, size=(30, 30, 30), dtype=np.uint8)
        result = calculate_s2_3d(img)

        # All S2 values should be between 0 and 1
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_calculate_s2_3d_uniform_ones(self):
        """Test 3D S2 on uniform volume."""
        img = np.ones((20, 20, 20), dtype=np.uint8)
        result = calculate_s2_3d(img)

        # For uniform volume of 1s, S2 should be 1
        assert np.allclose(result, 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
