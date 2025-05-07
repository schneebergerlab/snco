import logging

import numpy as np
import pandas as pd

from snco.utils import load_json
from snco.signal import argmax_smoothed_haplotype
from snco.records import MarkerRecords, NestedData, NestedDataArray
from snco.opts import DEFAULT_RANDOM_SEED

log = logging.getLogger('snco')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def _estimate_marker_background(m, ws=100):
    """
    Estimate background signal by masking out predicted foreground signal.

    Parameters
    ----------
    m : np.ndarray
        Marker count matrix with shape (bins, haplotypes).
    ws : int, default=100
        Width of the convolution window in bins.

    Returns
    -------
    np.ndarray
        Background marker matrix with foreground masked with zeros.
    """
    fg_idx = argmax_smoothed_haplotype(m, ws)
    fg_masked = m.copy()
    fg_masked[np.arange(len(m)), fg_idx] = 0
    return fg_masked


def estimate_overall_background_signal(co_markers, conv_window_size, max_frac_bg,
                                       apply_per_geno=True):
    """
    Estimate and store background signal and barcode-level background contamination.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker records to process.
    conv_window_size : int
        Width of the background convolution window in base pairs.
    max_frac_bg : float
        Maximum tolerated background contamination fraction per barcode.
    apply_per_geno : bool, default=True
        Estimate background separately for each genotype.

    Returns
    -------
    MarkerRecords
        Filtered marker records with updated metadata.
    """
    conv_bins = conv_window_size // co_markers.bin_size
    background_signal = NestedDataArray(levels=('genotype', 'chrom'))
    estimated_background_fraction = NestedData(levels=('cb',), dtype=float)
    for geno, geno_co_markers in co_markers.groupby(by='genotype' if apply_per_geno else 'none'):
        bg_signal = {}
        frac_bg = {}
        for cb in geno_co_markers.barcodes:
            cb_co_markers = geno_co_markers[cb]
            bg_count = 0
            tot_count = 0
            for chrom, m in cb_co_markers.items():
                bg = _estimate_marker_background(m, conv_bins)
                if chrom not in bg_signal:
                    bg_signal[chrom] = bg
                else:
                    bg_signal[chrom] += bg

                bg_count += bg.sum(axis=None)
                tot_count += m.sum(axis=None)
            frac_bg[cb] = float(bg_count / tot_count)
            if frac_bg[cb] > max_frac_bg:
                co_markers.pop(cb)
        bg_signal = {
            chrom: sig / sig.sum(axis=None)
            for chrom, sig in bg_signal.items()
        }
        background_signal[geno] = bg_signal
        estimated_background_fraction.update(frac_bg)

    co_markers.add_metadata(
        background_signal=background_signal,
        estimated_background_fraction=estimated_background_fraction
    )
    return co_markers


def random_bg_sample(m, bg_signal, n_bg, rng=DEFAULT_RNG):
    """
    Randomly sample background signal proportionally to observed counts and background model.

    Parameters
    ----------
    m : np.ndarray
        Marker count matrix with shape (bins, haplotypes).
    bg_signal : np.ndarray
        Background probabilities with shape (bins, haplotypes).
    n_bg : int
        Number of background markers to sample.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Matrix of sampled background markers.
    """
    bg_idx = np.nonzero(m)
    m_valid = m[bg_idx]
    p = m_valid * bg_signal[bg_idx]
    bg = np.zeros_like(m)
    p_denom = p.sum(axis=None)
    if p_denom == 0:
        return bg
    p = p / p_denom
    n_p = p.shape[0]
    bg_c = np.bincount(rng.choice(np.arange(n_p), size=n_bg, replace=True, p=p), minlength=n_p)
    bg[bg_idx] = np.minimum(bg_c, m_valid)
    return bg


def subtract_background(m, bg_signal, frac_bg, return_bg=False, rng=DEFAULT_RNG):
    """
    Subtract background signal from marker matrix.

    Parameters
    ----------
    m : np.ndarray
        Marker count matrix with shape (bins, haplotypes).
    bg_signal : np.ndarray
        Background probabilities with shape (bins, haplotypes).
    frac_bg : float
        Estimated background fraction for barcode.
    return_bg : bool, default=False
        Whether to return background matrix as well as background-subtracted marker matrix
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray or tuple of np.ndarray
        Background-subtracted marker matrix (and optionally the background matrix).
    """
    tot = m.sum(axis=None)
    if tot:
        n_bg = round(tot * frac_bg)
        bg = random_bg_sample(m, bg_signal, n_bg, rng=rng)
    else:
        # no markers on this chromosome
        log.warning('Saw chromosome with no markers when estimating background signal')
        bg = m.copy()
    m_sub = m - bg
    if not return_bg:
        return m_sub
    return m_sub, bg


def clean_marker_background(co_markers, apply_per_geno=True, rng=DEFAULT_RNG):
    """
    Subtract estimated background signal from marker data for each barcode.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker records with background metadata.
    apply_per_geno : bool, default=True
        Whether to clean using genotype-specific background.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    MarkerRecords
        Background-cleaned marker records.
    """
    bg_signal = co_markers.metadata['background_signal']
    frac_bg = co_markers.metadata['estimated_background_fraction']
    if apply_per_geno:
        genotypes = co_markers.metadata['genotypes']
    co_markers_c = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        geno = genotypes[cb] if apply_per_geno else 'ungrouped'
        co_markers_c[cb, chrom] = subtract_background(
            m, bg_signal[geno, chrom], frac_bg[cb], rng=rng
        )
    return co_markers_c
