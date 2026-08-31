"""
Connected-component labeling and per-component size measurement for binary
microstructure images.
"""
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table

# Avizo/Dragonfly describe neighbor rules by how many touching neighbors count
# as "connected" - 6/18/26 in 3D (face, face+edge, face+edge+corner), 4/8 in
# 2D (face, face+corner). skimage.measure.label instead takes a "connectivity"
# argument that is the RANK of the neighborhood (1, 2, or 3 in 3D; 1 or 2 in
# 2D), not a neighbor count - these maps translate one vocabulary into the
# other so the rest of the app (and its UI) can stay in the terms users expect.
_CONNECTIVITY_MAP_3D = {6: 1, 18: 2, 26: 3}
_CONNECTIVITY_MAP_2D = {4: 1, 8: 2}


def _label_and_measure(
    image: np.ndarray, connectivity_map: dict, connectivity: int, res: float,
    size_exponent: int, count_col: str, measure_col: str,
) -> dict:
    """Label connected components of the foreground phase (value 1) in a
    binary image, and measure the size of each one.

    Shared core used by connected_components_2d/connected_components_3d -
    not part of the public API.

    Args:
        image: binary array, 2D or 3D. Any nonzero element counts as
            foreground.
        connectivity_map: maps the connectivity numbers a caller can pass
            (e.g. {6: 1, 18: 2, 26: 3}) to the "connectivity" rank
            skimage.measure.label expects.
        connectivity: neighbor rule to use - must be a key in
            connectivity_map.
        res: pixel/voxel size (physical units per pixel/voxel, assumed
            isotropic).
        size_exponent: power to raise `res` to when converting a raw
            element count into a physical size (2 for area, 3 for volume).
        count_col: name to give the raw element-count column in the
            returned table.
        measure_col: name to give the physical-size column in the returned
            table.

    Returns:
        dict with:
            'labels': int array, same shape as `image` - 0 is background,
                1..N are component ids.
            'num_components': N.
            'table': pandas DataFrame, one row per component, sorted by
                `measure_col` descending - columns 'label', `count_col`,
                `measure_col`.

    Raises:
        ValueError: if `connectivity` is not a key in `connectivity_map`.
    """
    if connectivity not in connectivity_map:
        raise ValueError(f"connectivity must be one of {sorted(connectivity_map)}, got {connectivity}")

    labels, num_components = label(
        image.astype(bool),
        connectivity=connectivity_map[connectivity],
        return_num=True,
    )

    # regionprops_table skips label 0 (background) automatically.
    props = regionprops_table(labels, properties=('label', 'area'))
    table = pd.DataFrame(props).rename(columns={'area': count_col})
    table[measure_col] = table[count_col] * res ** size_exponent
    table = table.sort_values(measure_col, ascending=False).reset_index(drop=True)

    return {
        'labels': labels,
        'num_components': num_components,
        'table': table,
    }


def connected_components_2d(image: np.ndarray, connectivity: int = 8, res: float = 1.0) -> dict:
    """Label connected components of the foreground phase (value 1) in a 2D
    binary image, and measure the area of each one.
    """
    return _label_and_measure(
        image, _CONNECTIVITY_MAP_2D, connectivity, res,
        size_exponent=2, count_col='pixel_count', measure_col='area',
    )

def connected_components_3d(image: np.ndarray, connectivity: int = 26, res: float = 1.0) -> dict:
    """Label connected components of the foreground phase (value 1) in a 3D
    binary image, and measure the volume of each one.

    Args:
        image: 3D array (Z, Y, X). Any nonzero voxel counts as foreground.
        connectivity: neighbor rule - 6 (face neighbors only), 18 (+ edge
            neighbors), or 26 (+ corner neighbors).
        res: voxel size (physical units per voxel, assumed isotropic).

    Returns:
        dict with:
            'labels': int array, same shape as `image` - 0 is background,
                1..N are component ids.
            'num_components': N.
            'table': pandas DataFrame, one row per component, sorted by
                volume descending - columns 'label', 'voxel_count', 'volume'.
    """
    return _label_and_measure(
        image, _CONNECTIVITY_MAP_3D, connectivity, res,
        size_exponent=3, count_col='voxel_count', measure_col='volume',
    )
