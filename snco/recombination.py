import numpy as np
from scipy.ndimage import convolve1d

from .opts import DEFAULT_RANDOM_SEED


DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def nonzero_range(arr, axis=-1):
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


def calculate_cm_denominator(co_markers):
    denom = {}
    for chrom in co_markers.chrom_sizes:
        denom[chrom] = nonzero_range(
            co_markers[..., chrom].sum(axis=-1),
            axis=1
        ).astype(np.float32)
    return denom


def recombination_landscape(co_preds,
                            co_markers=None,
                            rolling_mean_window_size=1_000_000,
                            nboots=100,
                            min_prob=0.01,
                            rng=DEFAULT_RNG):

    nf = 1_000_000 // co_preds.bin_size
    ws = rolling_mean_window_size // co_preds.bin_size
    filt = np.ones(ws) / ws
    N = len(co_preds)

    if co_markers is not None:
        if co_preds.barcodes != co_markers.barcodes:
            raise ValueError('Cell barcodes from marker-json-fn and predict-json-fn do not match')
        denominators = calculate_cm_denominator(co_markers)
    else:
        denominators = None

    cm_per_mb = {}
    for chrom, nbins in co_preds.nbins.items():
        chrom_hap_probs = co_preds[..., chrom]
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
            chrom_denom = denominators[chrom]
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
        cm_per_mb[chrom] = np.stack(chrom_cm_per_mb)
    return cm_per_mb
