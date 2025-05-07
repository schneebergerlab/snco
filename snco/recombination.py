import numpy as np
from scipy.ndimage import convolve1d

from .records import NestedDataArray
from .defaults import DEFAULT_RANDOM_SEED


DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def nonzero_range(arr, axis=-1):
    """
    Returns a boolean mask indicating the range of nonzero values along a given axis.

    This function identifies the continuous range of nonzero values along a specified axis
    by using forward and reverse cumulative logical OR operations. It returns `True` for positions
    where there are nonzero values at or flanking the position on both sides, and `False` elsewhere.

    Parameters
    ----------
    arr : ndarray
        Input array of any shape, containing the data to check for nonzero values.
    axis : int, optional
        The axis along which to identify nonzero ranges. The default is the last axis (-1).

    Returns
    -------
    ndarray
        A boolean mask of the same shape as the input array, with `True` where there are nonzero
        values at or flanking positions on both sides and `False` otherwise.

    Example
    -------
    Consider the following 1D array:

    >>> arr = np.array([0, 1, 0, 3, 0, 0, 4, 0])

    Applying `nonzero_range(arr)` will return a boolean mask for the nonzero values:

    >>> nonzero_range(arr)
    array([False,  True, False,  True, False, False,  True, False])

    The mask indicates the range of nonzero values along the array, marking `True` where nonzero
    values occur.
    """
    arr = arr.astype(bool)
    fwd = np.logical_or.accumulate(arr, axis=axis)
    rev = np.flip(
        np.logical_or.accumulate(
            np.flip(arr, axis=axis),
            axis=axis
        ),
        axis=axis
    )
    return np.logical_and(fwd, rev)


def calculate_cm_denominator(co_markers, apply_per_geno):
    """
    Calculate the denominator for recombination rate calculations based on marker data.

    This function calculates the denominators for each chromosome, which are used to scale
    the recombination landscape during computation. The denominators are based on the presence
    of markers flanking each position, calculated using the `nonzero_range` function.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker records object containing marker matrices per barcode and chromosome.

    Returns
    -------
    dict
        A dictionary where the keys are chromosome names and the values are the denominators
        for recombination calculations for each chromosome.
    """
    denom = NestedDataArray(levels=('genotype', 'chrom'))
    for geno, geno_co_markers in co_markers.groupby(by='genotype' if apply_per_geno else 'none'):
        for chrom in co_markers.chrom_sizes:
            denom[geno, chrom] = nonzero_range(
                geno_co_markers[:, chrom].stack_values().sum(axis=-1),
                axis=1
            ).astype(np.float32)
    return denom


def recombination_landscape(co_preds,
                            co_markers=None,
                            apply_per_geno=True,
                            rolling_mean_window_size=1_000_000,
                            nboots=100,
                            min_prob=5e-3,
                            rng=DEFAULT_RNG):
    """
    Calculate the recombination landscape for a PredictionRecords dataset.

    This function estimates the recombination rate per megabase for each chromosome by calculating
    the gradient of haplotype probabilities across bins, and bootstrapping the values to assess
    the uncertainty in the landscape. If a co_markers dataset is provided, it uses it to calculate 
    denominators to scale the recombination rate at chromosome ends, where lack of markers may
    limit sensitivity.

    Parameters
    ----------
    co_preds : PredictionRecords
        PredictionRecords object containing the haplotype predictions.
    co_markers : MarkerRecords, optional
        Marker records object used to scale the recombination rate. If None, no scaling is performed.
    apply_per_geno : bool, optional
        Whether to group barcodes by genotype for recombination landscape calculation. Default is True.
    rolling_mean_window_size : int, optional
        The size of the window for the rolling mean filter (default is 1,000,000).
    nboots : int, optional
        The number of bootstrap samples to draw for uncertainty estimation (default is 100).
    min_prob : float, optional
        The minimum threshold for crossover prediction gradients (default is 5e-3).
    rng : np.random.Generator, optional
        A random number generator for bootstrapping (default is `DEFAULT_RNG`).

    Returns
    -------
    dict
        A dictionary where the keys are chromosome names and the values are arrays of recombination
        rates per megabase, calculated from bootstrapped samples.

    Raises
    ------
    ValueError
        If the barcodes of the co-predictions and co-markers do not match.
    """
    nf = 1_000_000 // co_preds.bin_size
    ws = rolling_mean_window_size // co_preds.bin_size
    filt = np.ones(ws) / ws

    if co_markers is not None:
        if co_preds.barcodes != co_markers.barcodes:
            raise ValueError('Cell barcodes from marker-json-fn and predict-json-fn do not match')
        denominators = calculate_cm_denominator(co_markers, apply_per_geno)
    else:
        denominators = None

    cm_per_mb = NestedDataArray(levels=('genotype', 'chrom',))
    for geno, geno_co_preds in co_preds.groupby(by='genotype' if apply_per_geno else 'none'):
        N = len(geno_co_preds)
        for chrom, nbins in geno_co_preds.nbins.items():
            chrom_hap_probs = geno_co_preds[:, chrom].stack_values()
            chrom_co_probs = np.abs(np.diff(
                chrom_hap_probs,
                n=1,
                axis=1,
                prepend=chrom_hap_probs[:, 0].reshape(-1, 1)
            ))
            # filter gradients smaller than min_prob
            chrom_co_probs = np.where(
                chrom_co_probs >= min_prob, chrom_co_probs, 0
            )
            # filter gradients where there are no markers
            if denominators is not None:
                chrom_denom = denominators[geno, chrom]
                chrom_denom = convolve1d(
                    chrom_denom, filt,
                    axis=1,
                    mode='constant',
                    cval=0
                )
                chrom_co_probs = np.where(
                    chrom_denom > 0,
                    chrom_co_probs,
                    0,
                )
            else:
                chrom_denom = np.ones(shape=(N, nbins))
            # equivalent to rolling sum accounting for edge effects
            chrom_co_probs = convolve1d(
                chrom_co_probs, filt,
                axis=1,
                mode='constant',
                cval=0
            ) * nf
            chrom_cm_per_mb = []
            for _ in range(nboots):
                idx = rng.integers(0, N, size=N)
                chrom_cm_per_mb.append(
                    (chrom_co_probs[idx].sum(axis=0) / chrom_denom[idx].sum(axis=0)) * 100
                )
            cm_per_mb[geno, chrom] = np.stack(chrom_cm_per_mb)
    co_preds.add_metadata(recombination_landscape=cm_per_mb)
    return cm_per_mb
