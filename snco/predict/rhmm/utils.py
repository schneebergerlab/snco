import numpy as np

def interp_nan_inplace(arr, axis):
    """
    Interpolates NaN values in-place along a specified axis using linear interpolation.

    Parameters
    ----------
    arr : np.ndarray
        A NumPy array potentially containing NaN values. This array is modified in-place.
    axis : int
        The axis along which to perform interpolation.
    """
    if arr.ndim <= axis:
        raise ValueError('Axis is out of bounds for the array')

    # Move the target axis to the front for easier iteration
    arr_swap = np.moveaxis(arr, axis, 0)
    shape = arr_swap.shape

    for idx in np.ndindex(*shape[1:]):
        vec = arr_swap[(slice(None),) + idx]
        nan_mask = np.isnan(vec)
        if nan_mask.any():
            real_mask = ~nan_mask
            xp = np.flatnonzero(real_mask)
            fp = vec[real_mask]
            x = np.flatnonzero(nan_mask)
            vec[nan_mask] = np.interp(x, xp, fp)


def sorted_edit_distance(state1, state2):
    dist = 0
    for h1, h2 in zip(sorted(state1), sorted(state2)):
        if h1 != h2:
            dist += 1
    return dist