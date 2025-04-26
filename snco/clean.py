import logging

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d

from .utils import load_json
from .records import MarkerRecords
from .metadata import MetadataDict
from .groupby import genotype_grouper, dummy_grouper
from .opts import DEFAULT_RANDOM_SEED

log = logging.getLogger('snco')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def filter_low_coverage_barcodes(co_markers, min_cov=0, min_cov_per_chrom=0):
    """
    Remove cell barcodes with insufficient total or per-chromosome marker coverage.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker records object containing marker matrices per barcode and chromosome.
    min_cov : int, default=0
        Minimum total number of markers required across all chromosomes.
    min_cov_per_chrom : int, default=0
        Minimum number of markers required per chromosome.

    Returns
    -------
    MarkerRecords
        Filtered marker records.
    """

    def _low_cov_query(cb):
        m_counts = [m.sum(axis=None) for m in co_markers[cb].values()]
        return (sum(m_counts) >= min_cov) & (min(m_counts) >= min_cov_per_chrom)

    return co_markers.query(_low_cov_query)


def filter_genotyping_score(co_markers, min_geno_prob=0.9):
    """
    Remove barcodes with genotyping probability below a given threshold.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker records with metadata containing genotyping probabilities.
    min_geno_prob : float, default=0.9
        Minimum allowed genotype probability.

    Returns
    -------
    MarkerRecords
        Filtered marker records.
    """
    try:
        geno_probs = co_markers.metadata['genotype_probability']
    except KeyError:
        return co_markers

    def _geno_query(cb):
        return geno_probs[cb] >= min_geno_prob

    return co_markers.query(_geno_query)


def predict_foreground_convolution(m, ws=100):
    """
    Predict foreground signal by convolution across bins.

    Parameters
    ----------
    m : np.ndarray
        Marker count matrix with shape (bins, haplotypes).
    ws : int, default=100
        Width of the convolution window in bins.

    Returns
    -------
    np.ndarray
        Array of foreground haplotype indices per bin.
    """
    rs = convolve1d(m, np.ones(ws), axis=0, mode='constant', cval=0)
    fg_idx = rs.argmax(axis=1)
    return fg_idx


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
    fg_idx = predict_foreground_convolution(m, ws)
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
    background_signal = MetadataDict(levels=('group', 'chrom'), dtype=np.ndarray)
    estimated_background_fraction = MetadataDict(levels=('cb',), dtype=float)
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
            frac_bg[cb] = bg_count / tot_count
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
        geno_grouper = genotype_grouper(co_markers)
    else:
        geno_grouper = dummy_grouper(co_markers)
    co_markers_c = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        geno = geno_grouper(cb)
        co_markers_c[cb, chrom] = subtract_background(
            m, bg_signal[geno][chrom], frac_bg[cb], rng=rng
        )
    return co_markers_c


def create_haplotype_imbalance_mask(co_markers, max_imbalance_mask=0.75, min_cb=20,
                                    apply_per_geno=True):
    """
    Create a mask for bins with high haplotype imbalance.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker data with haplotype-specific read counts.
    max_imbalance_mask : float, default=0.75
        Maximum allowed haplotype ratio before masking.
    min_cb : int, default=20
        Minimum number of cell barcodes required per bin.
    apply_per_geno : bool, default=True
        Mask separately per genotype.

    Returns
    -------
    dict
        Dictionary of masks by genotype and chromosome.
    int
        Total number of bins masked.
    """
    imbalance_mask = MetadataDict(levels=('group', 'chrom'), dtype=np.ndarray)
    for geno, geno_co_markers in co_markers.groupby(by='genotype' if apply_per_geno else 'none'):
        tot_signal = {}
        tot_obs = {}
        for _, chrom, m in geno_co_markers.deep_items():
            if chrom not in tot_signal:
                tot_signal[chrom] = m.copy()
                tot_obs[chrom] = np.minimum(m.sum(axis=1), 1)
            else:
                tot_signal[chrom] += m
                tot_obs[chrom] += np.minimum(m.sum(axis=1), 1)
        imbalance_mask[geno] = {}
        n_masked = 0
        for chrom, m in tot_signal.items():
            with np.errstate(invalid='ignore'):
                bin_sum = m.sum(axis=1)
                ratio = m[:, 0] / bin_sum
            np.nan_to_num(ratio, nan=0.5, copy=False)
            ratio_mask = (ratio > max_imbalance_mask) | (ratio < (1 - max_imbalance_mask))
            count_mask = tot_obs[chrom] >= min_cb
            mask = np.logical_and(ratio_mask, count_mask)
            n_masked += mask.sum(axis=None)
            imbalance_mask[geno, chrom] = np.stack([mask, mask], axis=1)
    co_markers.add_metadata(haplotype_imbalance_mask=imbalance_mask)
    return imbalance_mask, n_masked


def apply_haplotype_imbalance_mask(co_markers, max_imbalance_mask=0.9, apply_per_geno=True):
    """
    Apply mask to bins with excessive haplotype imbalance.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker data to mask.
    max_imbalance_mask : float, default=0.9
        Maximum allowed imbalance ratio.
    apply_per_geno : bool, default=True
        Apply masking per genotype.

    Returns
    -------
    MarkerRecords
        Masked marker records.
    int
        Number of bins masked.
    """
    mask, n_masked = create_haplotype_imbalance_mask(
        co_markers, max_imbalance_mask, apply_per_geno=apply_per_geno
    )
    if apply_per_geno:
        geno_grouper = genotype_grouper(co_markers)
    else:
        geno_grouper = dummy_grouper(co_markers)
    co_markers_m = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        geno = geno_grouper(cb)
        co_markers_m[cb, chrom] = np.where(mask[geno][chrom], 0, m)
    return co_markers_m, n_masked


def apply_marker_threshold(co_markers, max_marker_threshold):
    """
    Cap bin counts to a maximum marker threshold.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker data.
    max_marker_threshold : int
        Maximum allowed marker count per bin.

    Returns
    -------
    MarkerRecords
        Thresholded marker records.
    """
    co_markers_t = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        co_markers_t[cb, chrom] = np.minimum(m, max_marker_threshold)
    return co_markers_t


def mask_regions_bed(co_markers, mask_bed_fn):
    """
    Mask regions listed in a BED file.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker data to be masked.
    mask_bed_fn : str
        Path to BED file with regions to mask.

    Returns
    -------
    MarkerRecords
        Masked marker records.
    """
    co_markers_m = co_markers.copy()
    mask_invs = pd.read_csv(
        mask_bed_fn,
        sep='\t',
        usecols=[0, 1, 2],
        names=['chrom', 'start', 'end'],
        dtype={'chrom': str, 'start': int, 'end': int}
    )
    bs = co_markers.bin_size
    for chrom, invs in mask_invs.groupby('chrom'):
        start_bins = np.floor(invs.start // bs).astype(int)
        end_bins = np.ceil(invs.end // bs).astype(int)        
        for cb, m in co_markers_m.iter_chrom(chrom):
            for s, e in zip(start_bins, end_bins):
                m[s: e] = 0
    return co_markers_m


def run_clean(marker_json_fn, output_json_fn, *,
              co_markers=None, cb_whitelist_fn=None, mask_bed_fn=None, bin_size=25_000,
              min_markers_per_cb=0, min_markers_per_chrom=0, max_bin_count=20,
              clean_bg=True, bg_window_size=2_500_000, max_frac_bg=0.2, min_geno_prob=0.9,
              mask_imbalanced=True, max_marker_imbalance=0.75, apply_per_geno=True,
              rng=DEFAULT_RNG):
    """
    Complete pipeline for cleaning single-cell marker data by removing ambient background,
    extremely imbalanced bins and poorly supported barcodes.

    Parameters
    ----------
    marker_json_fn : str
        Input JSON file with marker data.
    output_json_fn : str
        Path to write cleaned marker data.
    co_markers : MarkerRecords, optional
        Pre-loaded marker records.
    cb_whitelist_fn : str, optional
        File path to barcode whitelist.
    mask_bed_fn : str, optional
        BED file of regions to mask.
    bin_size : int, default=25000
        Bin size for aggregation in base pairs.
    min_markers_per_cb : int, default=0
        Minimum total markers required for a barcode.
    min_markers_per_chrom : int, default=0
        Minimum markers per chromosome per barcode.
    max_bin_count : int, default=20
        Maximum marker count allowed per bin.
    clean_bg : bool, default=True
        Whether to subtract estimated background signal.
    bg_window_size : int, default=2500000
        Size of convolution window for background estimation.
    max_frac_bg : float, default=0.2
        Maximum tolerated background contamination.
    min_geno_prob : float, default=0.9
        Minimum genotype confidence required.
    mask_imbalanced : bool, default=True
        Whether to mask bins with haplotype imbalance.
    max_marker_imbalance : float, default=0.75
        Maximum tolerated haplotype ratio before masking.
    apply_per_geno : bool, default=True
        Apply all corrections separately per genotype.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    MarkerRecords
        Cleaned marker records.
    """
    if co_markers is None:
        co_markers = load_json(marker_json_fn, cb_whitelist_fn, bin_size)

    n = len(co_markers)
    if min_markers_per_cb:
        co_markers = filter_low_coverage_barcodes(
            co_markers, min_markers_per_cb, min_markers_per_chrom
        )
        log.info(
            f'Removed {n - len(co_markers)} barcodes with fewer than {min_markers_per_cb} markers '
            f'or fewer than {min_markers_per_chrom} markers per chromosome'
        )
    n = len(co_markers)
    if min_geno_prob:
        co_markers = filter_genotyping_score(co_markers, min_geno_prob)
        log.info(
            f'Removed {n - len(co_markers)} barcodes with genotyping probability < {min_geno_prob}'
        )

    n = len(co_markers)
    # estimate ambient marker rate for each CB and try to scrub common background markers
    log.info('Estimating background marker rates.')
    co_markers = estimate_overall_background_signal(
        co_markers, bg_window_size, max_frac_bg, apply_per_geno=apply_per_geno
    )
    log.info(
        f'Removed {n - len(co_markers)} barcodes with greater than {max_frac_bg * 100}%'
        ' background contamination'
    )
    av_bg = np.mean(
        list(co_markers.metadata['estimated_background_fraction'].values())
    )
    log.info(f'Average estimated background fraction is {av_bg:.3f}')
    if clean_bg:
        log.info('Attempting to filter likely background markers')
        co_markers = clean_marker_background(
            co_markers, apply_per_geno=apply_per_geno, rng=rng)

    if mask_imbalanced:
        # mask any bins that still have extreme imbalance
        # (e.g. due to extreme allele-specific expression differences)
        co_markers, n_masked = apply_haplotype_imbalance_mask(
            co_markers, max_marker_imbalance,
            apply_per_geno=apply_per_geno,
        )
        tot_bins = sum(co_markers.nbins.values())
        log.info(
            f'Masked {n_masked:d}/{tot_bins} bins with '
            f'marker imbalance greater than {max_marker_imbalance}'
        )


    n = len(co_markers)
    co_markers = filter_low_coverage_barcodes(
        co_markers, min_cov=0, min_cov_per_chrom=1
    )
    log.info(
        f'Removed {n - len(co_markers)} barcodes with at least '
        'one chromosome without markers after cleaning'
    )

    # threshold bins that have a large number of reads
    co_markers = apply_marker_threshold(co_markers, max_bin_count)
    log.info(f'Thresholded bins with >{max_bin_count} markers')

    if mask_bed_fn is not None:
        co_markers = mask_regions_bed(co_markers, mask_bed_fn)
        log.info(f'Masked regions blacklisted in {mask_bed_fn}')

    if output_json_fn is not None:
        log.info(f'Writing markers to {output_json_fn}')
        co_markers.write_json(output_json_fn)
    return co_markers
