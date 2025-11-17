import numpy as np
from numba import jit

@jit(nopython=True)
def two_point_correlation(im, dim, var=1):
    """
    This method computes the two point correlation,
    also known as second order moment,
    for a segmented binary image in the three principal directions.
    
    dim = 0: x-direction
    dim = 1: y-direction
    dim = 2: z-direction
    
    var should be set to the pixel value of the pore-space. (Default 1)
    
    The input image im is expected to be two-dimensional.
    """
    if dim == 0: #x_direction
        dim_1 = im.shape[1] #y-axis
        dim_2 = im.shape[0] #x-axis
    elif dim == 1: #y-direction
        dim_1 = im.shape[0] #x-axis
        dim_2 = im.shape[1] #z-axis
        
    two_point = np.zeros((dim_1, dim_2))
    for n1 in range(dim_1):
        for r in range(dim_2):
            lmax = dim_2-r
            for a in range(lmax):
                if dim == 0:
                    pixel1 = im[a, n1]
                    pixel2 = im[a+r, n1]
                elif dim == 1:
                    pixel1 = im[n1, a]
                    pixel2 = im[n1, a+r]

                if pixel1 == var and pixel2 == var:
                    two_point[n1, r] += 1
            two_point[n1, r] = two_point[n1, r]/(float(lmax))
    return two_point

@jit(nopython=True)
def two_point_correlation3D(im, dim, var=1):
    """
    This method computes the two point correlation,
    also known as second order moment,
    for a segmented binary image in the three principal directions.
    
    dim = 0: x-direction
    dim = 1: y-direction
    dim = 2: z-direction
    
    var should be set to the pixel value of the pore-space. (Default 1)
    
    The input image im is expected to be three-dimensional.
    """
    if dim == 0: #x_direction
        dim_1 = im.shape[2] #y-axis
        dim_2 = im.shape[1] #z-axis
        dim_3 = im.shape[0] #x-axis
    elif dim == 1: #y-direction
        dim_1 = im.shape[0] #x-axis
        dim_2 = im.shape[1] #z-axis
        dim_3 = im.shape[2] #y-axis
    elif dim == 2: #z-direction
        dim_1 = im.shape[0] #x-axis
        dim_2 = im.shape[2] #y-axis
        dim_3 = im.shape[1] #z-axis
        
    two_point = np.zeros((dim_1, dim_2, dim_3))
    for n1 in range(dim_1):
        for n2 in range(dim_2):
            for r in range(dim_3):
                lmax = dim_3-r
                for a in range(lmax):
                    if dim == 0:
                        pixel1 = im[a, n2, n1]
                        pixel2 = im[a+r, n2, n1]
                    elif dim == 1:
                        pixel1 = im[n1, n2, a]
                        pixel2 = im[n1, n2, a+r]
                    elif dim == 2:
                        pixel1 = im[n1, a, n2]
                        pixel2 = im[n1, a+r, n2]
                    
                    if pixel1 == var and pixel2 == var:
                        two_point[n1, n2, r] += 1
                two_point[n1, n2, r] = two_point[n1, n2, r]/(float(lmax))
    return two_point

def calculate_s2(image_data):
    """
    Calculate S2 (SMDS) values from image data.

    Parameters:
    -----------
    image_data : numpy.ndarray
        The input image data (can be 8-bit or 16-bit grayscale)

    Returns:
    --------
    numpy.ndarray
        The calculated S2 values
    """
    # Example calculation - replace with your actual SMDS calculation
    # This is a placeholder that calculates some statistics
    two_pt_dim0 = two_point_correlation(image_data, dim = 0, var = 1) #S2 in x-direction
    # print(type(two_pt_dim0), two_pt_dim0.shape)
    two_pt_dim1 = two_point_correlation(image_data, dim = 1, var = 1) #S2 in y-direction
    # print(type(two_pt_dim1), two_pt_dim1.shape)
    #Take average of directions; use half linear size assuming equal dimension sizes
    Nr = two_pt_dim0.shape[0]//2

    S2_x = np.average(two_pt_dim1, axis=0)[:Nr]
    S2_y = np.average(two_pt_dim0, axis=0)[:Nr]
    S2_average = ((S2_x + S2_y)/2)[:Nr]
    # # Convert to float for calculations
    # data = image_data.astype(np.float64)

    # # Example: Calculate mean along rows (you can replace this with your actual formula)
    # s2_values = np.mean(data, axis=1)

    # return s2_values
    return S2_average

def calculate_s2_3d(images, directional = False):
    """
    This function calculates average two-point correlation in 3D.
    Inputs: 
    Can be a 3D image (numpy array)
    or 
    4D array (couple of 3D images) whose first dimension is the number of images: (number_imgs, image_size, image_size, image_size).
    
    Returns:
    in the case of 3D image:
        --> it returns 4 numpy consisting two-point correlation in x, y, and z direction plus the average of them.

    in case of 4D images:
    --> it calculates S_2 and scaled-autocovariance (F_2) for all images,
    and return dataframes containing average s_2 and f_2, plus std and number of samples for each r.

    Note that correlation function is calculated from r = 0 to  half of the smallest dimension in the image.
    For instance if the image shape is (512, 256, 256), s2 size is 128.

    """
    
#     print(len(images.shape))
    Nr = min(images.shape)//2
    # only 1 3D image
    
    two_point_covariance = {}
    for j, direc in enumerate( ["x", "y", "z"]):
        two_point_direc =  two_point_correlation3D(images, dim = j, var = 1)
        two_point_covariance[direc] = two_point_direc
    direc_covariances = {}

    for direc in ["x", "y", "z"]:
        direc_covariances[direc] =  np.mean(np.mean(two_point_covariance[direc], axis=0), axis=0)[: Nr]
    average = (np.array(direc_covariances['x']) + np.array(direc_covariances['y']) + np.array(direc_covariances['z']) )/3
    
    return average
    # return np.array(direc_covariances['x']), np.array(direc_covariances['y']), np.array(direc_covariances['z']), average