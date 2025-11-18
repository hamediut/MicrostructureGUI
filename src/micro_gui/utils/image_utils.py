"""Image utility functions."""

def get_image_info(image_data):
    """
    Get basic information about an image.
    
    Args:
        image_data: numpy array
        
    Returns:
        dict with image info
    """
    return {
        'shape': image_data.shape,
        'dtype': str(image_data.dtype),
        'min': float(image_data.min()),
        'max': float(image_data.max()),
        'mean': float(image_data.mean())
    }