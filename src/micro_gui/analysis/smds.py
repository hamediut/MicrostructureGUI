"""
SMDS (Second-order Moment Density Statistics) calculation module.

This module provides functions for calculating two-point correlation functions
and SMDS values for binary microstructure images in 2D and 3D.
"""
import os
import numpy as np
from numba import jit
from typing import Tuple

import pandas as pd
from typing import List, Dict, Optional


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

def cal_fn( polytope:np.ndarray, n:int)->np.ndarray:
    """This function calculates scaled autocovariance function from Pn function.
    polytope:polytope function it can be two point correlation function (s2) or 
    higher order functions such as p3, p4, etc
    n: order of polytope e.g., n =2 for two-point correlation (s_2), n= 3 for p3h and p3v"""
    numerator = polytope - polytope[0] ** n
    denominator = polytope[0] - polytope[0] ** n
    fn_r = numerator/ denominator
    return fn_r

def calculate_s2_4d(images: np.ndarray):
    """
    Calculate average two-point correlation for a 4D stacks ( a stacks of 3D volumes, not time series).

    This function computes S2 values for 4D image data by calculating
    correlations in all four principal directions and averaging them.

    Args:
        images: 4D binary image array (values should be 0 and 1)
            Shape: (stack_number, depth, height, width)
    Returns:
    df_s2_grouped: a dataframe containing  average S2 values for all 3D volumes in the 4D stack.
    df_f2_grouped: a dataframe containing  average F2 values for all 3D volumes in the 4D stack.
    
    """
    Nr = min(images.shape[1:])//2
    s2_list = []
    f2_list = []

    for i in range(images.shape[0]):
        
        two_point_covariance = {}
        for j, direc in enumerate(["x", "y", "z"]) :
            two_point_direc = two_point_correlation3D(images[i], j, var = 1)
            two_point_covariance[direc] = two_point_direc
    
        direc_covariances = {}
        for direc in ["x", "y", "z"]:
            direc_covariances[direc] = np.mean(np.mean(two_point_covariance[direc], axis=0), axis=0)[: Nr]
        s2_average = (direc_covariances['x'] + direc_covariances['y'] + direc_covariances['z'])/3
        s2_list.append(s2_average)
        f2_list.append(cal_fn(s2_average, n = 2))
    # s2--------------------
    df_s2_list = []
    for k in np.arange(0, len(s2_list)):
        df_s2_list.append(pd.DataFrame(s2_list[k], columns = ['s2']))
    df_s2 = pd.concat(df_s2_list)
    df_s2['r'] = df_s2.index
    # df_s2_grouped = df_s2.groupby(['r']).agg( {'s2': [np.mean, np.std, np.size] } )
    df_s2_grouped = df_s2.groupby(['r']).agg( {'s2': ['mean', 'std', 'size'] } )
    
    # f2--------------------
    df_f2_list = []
    for k in np.arange(0, len(f2_list)):
        df_f2_list.append(pd.DataFrame(f2_list[k], columns = ['f2']))
    df_f2 = pd.concat(df_f2_list)
    df_f2['r'] = df_f2.index
    # df_f2_grouped = df_f2.groupby(['r']).agg( {'f2': [np.mean, np.std, np.size] } )
    df_f2_grouped = df_f2.groupby(['r']).agg( {'f2': ['mean', 'std', 'size'] } )

    return df_s2_grouped, df_f2_grouped

def REV(image: np.ndarray,
            img_size_list: List[int],
            n_rand_samples: int)-> Dict[str, pd.DataFrame]:
    
    """
    This function receives a 3D (XCT) image and calculates average S2 and F2 for the whole image and a number of random subvolumes.
    These average correlation functions can then be analysed to determine the REV for the image.
    Parameters
    ----------
    image: np.ndarray
    This is the 3D image read as numpy array to do REV analysis on.

    img_size_list: List
    list of image sizes to calculate correlation functions. These sizes should be smaller than the whole image.

    n_rand_samples: int
    number of random images used for calculating REV. use 30 or more.

    Returns
    --------
    It returns two dictionary: one for s2 (s2_3d_dict) and one for f2 (f2_3d_dict)
    """
    seed =33
    np.random.seed(seed)
    x_max, y_max, z_max = image.shape[:]

    s2_3d_dict = {}
    f2_3d_dict = {}

    for image_size in img_size_list:

        all_crops = np.zeros((n_rand_samples, image_size, image_size, image_size), dtype = np.uint8)
        for i in range(n_rand_samples):

            x = np.random.randint (0, x_max - image_size)
            y = np.random.randint (0, y_max - image_size)
            z = np.random.randint (0, z_max - image_size)

            crop_image = image[x:x + image_size, y:y + image_size, z:z + image_size]
            all_crops[i] = crop_image
            
        df_s2, df_f2 = calculate_s2_4d(all_crops)
        s2_3d_dict[f'sub_{image_size}'] = df_s2
        f2_3d_dict[f'sub_{image_size}'] = df_f2

        # print(f'{image_size} done !')
    
    # print('Calculating S2 and F2 for the whole volume ...')
    # # calculate S2 and f2 for the whole volume
    # s2_avg_original  = calculate_s2_3d(image)
    # f2_avg_original = cal_fn(s2_avg_original, n = 2)

    # s2_3d_dict['original'] = s2_avg_original
    # f2_3d_dict['original'] = f2_avg_original

        
    return s2_3d_dict, f2_3d_dict


