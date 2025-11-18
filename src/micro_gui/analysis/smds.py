"""
SMDS (Second-order Moment Density Statistics) calculation module.

This module provides functions for calculating two-point correlation functions
and SMDS values for binary microstructure images in 2D and 3D.
"""

import numpy as np
from numba import jit
from typing import Tuple


@jit(nopython=True)
def two_point_correlation(im: np.ndarray, dim: int, var: int = 1) -> np.ndarray:
    """
    Compute the two-point correlation (second-order moment) for a 2D binary image.

    This function calculates the probability that two points separated by a distance r
    in a specified direction both belong to the same phase (defined by var).

    Args:
        im: 2D binary image array (values should be 0 and 1)
        dim: Direction for correlation calculation
            - 0: x-direction
            - 1: y-direction
        var: Pixel value of the phase of interest (default: 1)

    Returns:
        2D array containing correlation values for each direction and distance

    Note:
        Uses Numba JIT compilation for performance optimization.
    """
    if dim == 0:  # x-direction
        dim_1 = im.shape[1]  # y-axis
        dim_2 = im.shape[0]  # x-axis
    elif dim == 1:  # y-direction
        dim_1 = im.shape[0]  # x-axis
        dim_2 = im.shape[1]  # z-axis

    two_point = np.zeros((dim_1, dim_2))
    for n1 in range(dim_1):
        for r in range(dim_2):
            lmax = dim_2 - r
            for a in range(lmax):
                if dim == 0:
                    pixel1 = im[a, n1]
                    pixel2 = im[a + r, n1]
                elif dim == 1:
                    pixel1 = im[n1, a]
                    pixel2 = im[n1, a + r]

                if pixel1 == var and pixel2 == var:
                    two_point[n1, r] += 1
            two_point[n1, r] = two_point[n1, r] / float(lmax)
    return two_point


@jit(nopython=True)
def two_point_correlation3D(im: np.ndarray, dim: int, var: int = 1) -> np.ndarray:
    """
    Compute the two-point correlation (second-order moment) for a 3D binary image.

    This function calculates the probability that two points separated by a distance r
    in a specified direction both belong to the same phase (defined by var).

    Args:
        im: 3D binary image array (values should be 0 and 1)
        dim: Direction for correlation calculation
            - 0: x-direction
            - 1: y-direction
            - 2: z-direction
        var: Pixel value of the phase of interest (default: 1)

    Returns:
        3D array containing correlation values for each direction and distance

    Note:
        Uses Numba JIT compilation for performance optimization.
    """
    if dim == 0:  # x-direction
        dim_1 = im.shape[2]  # y-axis
        dim_2 = im.shape[1]  # z-axis
        dim_3 = im.shape[0]  # x-axis
    elif dim == 1:  # y-direction
        dim_1 = im.shape[0]  # x-axis
        dim_2 = im.shape[1]  # z-axis
        dim_3 = im.shape[2]  # y-axis
    elif dim == 2:  # z-direction
        dim_1 = im.shape[0]  # x-axis
        dim_2 = im.shape[2]  # y-axis
        dim_3 = im.shape[1]  # z-axis

    two_point = np.zeros((dim_1, dim_2, dim_3))
    for n1 in range(dim_1):
        for n2 in range(dim_2):
            for r in range(dim_3):
                lmax = dim_3 - r
                for a in range(lmax):
                    if dim == 0:
                        pixel1 = im[a, n2, n1]
                        pixel2 = im[a + r, n2, n1]
                    elif dim == 1:
                        pixel1 = im[n1, n2, a]
                        pixel2 = im[n1, n2, a + r]
                    elif dim == 2:
                        pixel1 = im[n1, a, n2]
                        pixel2 = im[n1, a + r, n2]

                    if pixel1 == var and pixel2 == var:
                        two_point[n1, n2, r] += 1
                two_point[n1, n2, r] = two_point[n1, n2, r] / float(lmax)
    return two_point


def calculate_s2(image_data: np.ndarray) -> np.ndarray:
    """
    Calculate S2 (SMDS) values from 2D image data.

    This function computes the two-point correlation in multiple directions
    and returns the averaged S2 values as a function of distance.

    Args:
        image_data: 2D binary image array (values should be 0 and 1)

    Returns:
        1D array of averaged S2 values

    Example:
        >>> import numpy as np
        >>> img = np.random.randint(0, 2, size=(100, 100))
        >>> s2 = calculate_s2(img)
        >>> print(s2.shape)  # Will be approximately (50,)
    """
    # S2 in x-direction
    two_pt_dim0 = two_point_correlation(image_data, dim=0, var=1)
    # S2 in y-direction
    two_pt_dim1 = two_point_correlation(image_data, dim=1, var=1)

    # Take average of directions; use half linear size assuming equal dimension sizes
    Nr = two_pt_dim0.shape[0] // 2

    S2_x = np.average(two_pt_dim1, axis=0)[:Nr]
    S2_y = np.average(two_pt_dim0, axis=0)[:Nr]
    S2_average = ((S2_x + S2_y) / 2)[:Nr]

    return S2_average


def calculate_s2_3d(images: np.ndarray, directional: bool = False) -> np.ndarray:
    """
    Calculate average two-point correlation in 3D.

    This function calculates S2 values for 3D image volumes by computing
    correlations in all three principal directions and averaging them.

    Args:
        images: 3D binary image array (values should be 0 and 1)
            Shape: (depth, height, width)
        directional: If True, return directional correlations separately
            (currently not implemented, reserved for future use)

    Returns:
        1D array of averaged S2 values across all three directions

    Note:
        The correlation function is calculated from r=0 to half of the
        smallest dimension in the image. For instance, if the image shape
        is (512, 256, 256), the S2 size will be 128.

    Example:
        >>> import numpy as np
        >>> img_3d = np.random.randint(0, 2, size=(100, 100, 100))
        >>> s2_3d = calculate_s2_3d(img_3d)
        >>> print(s2_3d.shape)  # Will be (50,)
    """
    Nr = min(images.shape) // 2

    two_point_covariance = {}
    for j, direc in enumerate(["x", "y", "z"]):
        two_point_direc = two_point_correlation3D(images, dim=j, var=1)
        two_point_covariance[direc] = two_point_direc

    direc_covariances = {}
    for direc in ["x", "y", "z"]:
        direc_covariances[direc] = np.mean(
            np.mean(two_point_covariance[direc], axis=0), axis=0
        )[:Nr]

    average = (
        np.array(direc_covariances['x']) +
        np.array(direc_covariances['y']) +
        np.array(direc_covariances['z'])
    ) / 3

    return average
