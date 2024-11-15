import logging

import numpy as np
from scipy.ndimage import convolve1d

from .utils import load_json
from .records import MarkerRecords
from .opts import DEFAULT_RANDOM_SEED

log = logging.getLogger('snco')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def filter_low_coverage_barcodes(co_markers, min_cov=0, min_cov_per_chrom=0):
    co_markers_f = co_markers.copy()
    for cb in co_markers_f.barcodes:
        m_counts = [m.sum(axis=None) for m in co_markers_f[cb].values()]
        if (sum(m_counts) < min_cov) | (min(m_counts) < min_cov_per_chrom):
            co_markers_f.pop(cb)
    return co_markers_f


def predict_foreground_convolution(m, ws=100):
    rs = convolve1d(m, np.ones(ws), axis=0, mode='constant', cval=0)
    fg_idx = rs.argmax(axis=1)
    return fg_idx


def _estimate_marker_background(m, ws=100):
    fg_idx = predict_foreground_convolution(m, ws)
    fg_masked = m.copy()
    fg_masked[np.arange(len(m)), fg_idx] = 0
    return fg_masked


def estimate_overall_background_signal(co_markers, conv_window_size, max_frac_bg):
    conv_bins = conv_window_size // co_markers.bin_size
    bg_signal = {}
    frac_bg = {}
    for cb in co_markers.barcodes:
        cb_co_markers = co_markers[cb]
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
        frac_bg[cb] = np.minimum((bg_count * 2) / tot_count, 0.5)
        if frac_bg[cb] > max_frac_bg:
            co_markers.pop(cb)

    bg_signal = {
        chrom: sig / sig.sum(axis=None)
        for chrom, sig in bg_signal.items()
    }
    co_markers.metadata['background_signal'] = bg_signal
    co_markers.metadata['estimated_background_fraction'] = frac_bg
    return co_markers


def random_bg_sample(m, bg_signal, n_bg, rng=DEFAULT_RNG):
    bg_idx = np.nonzero(m)
    m_valid = m[bg_idx]
    p = m_valid * bg_signal[bg_idx]
    p = p / p.sum(axis=None)
    n_p = p.shape[0]
    bg_c = np.bincount(rng.choice(np.arange(n_p), size=n_bg, replace=True, p=p), minlength=n_p)
    bg = np.zeros_like(m)
    bg[bg_idx] = np.minimum(bg_c, m_valid)
    return bg


def subtract_background(m, bg_signal, frac_bg, return_bg=False, rng=DEFAULT_RNG):
    tot = m.sum(axis=None)
    if tot:
        n_bg = round(tot * frac_bg)
        bg = random_bg_sample(m, bg_signal, n_bg, rng=rng)
    else:
        # no markers on this chromosome
        log.warn('Saw chromosome with no markers when estimating background signal')
        bg = m.copy()
    m_sub = m - bg
    if not return_bg:
        return m_sub
    return m_sub, bg


def clean_marker_background(co_markers, rng=DEFAULT_RNG):
    bg_signal = co_markers.metadata['background_signal']
    frac_bg = co_markers.metadata['estimated_background_fraction']
    co_markers_c = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        co_markers_c[cb, chrom] = subtract_background(
            m, bg_signal[chrom], frac_bg[cb], rng=rng
        )
    return co_markers_c


def create_haplotype_imbalance_mask(co_markers, max_imbalance_mask=0.9, min_cb=5):
    tot_signal = {}
    tot_obs = {}
    for _, chrom, m in co_markers.deep_items():
        if chrom not in tot_signal:
            tot_signal[chrom] = m.copy()
            tot_obs[chrom] = np.minimum(m.sum(axis=1), 1)
        else:
            tot_signal[chrom] += m
            tot_obs[chrom] += np.minimum(m.sum(axis=1), 1)

    imbalance_mask = {}
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
        imbalance_mask[chrom] = np.stack([mask, mask], axis=1)
    co_markers.metadata['haplotype_imbalance_mask'] = imbalance_mask
    return imbalance_mask, n_masked


def apply_haplotype_imbalance_mask(co_markers, max_imbalance_mask=0.9):
    mask, n_masked = create_haplotype_imbalance_mask(co_markers, max_imbalance_mask)
    co_markers_m = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        co_markers_m[cb, chrom] = np.where(mask[chrom], 0, m)
    return co_markers_m, n_masked


def apply_marker_threshold(co_markers, max_marker_threshold):
    co_markers_t = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        co_markers_t[cb, chrom] = np.minimum(m, max_marker_threshold)
    return co_markers_t


def run_clean(marker_json_fn, output_json_fn, *,
              co_markers=None,
              cb_whitelist_fn=None, bin_size=25_000,
              min_markers_per_cb=0, min_markers_per_chrom=0, max_bin_count=20,
              clean_bg=True, bg_window_size=2_500_000, max_frac_bg=0.2,
              mask_imbalanced=True, max_marker_imbalance=0.9,
              rng=DEFAULT_RNG):
    '''
    Removes predicted background markers, that result from ambient nucleic acids, 
    from each cell barcode.
    '''
    if co_markers is None:
        co_markers = load_json(marker_json_fn, cb_whitelist_fn, bin_size)

    n = len(co_markers)
    if min_markers_per_cb:
        co_markers = filter_low_coverage_barcodes(co_markers, min_markers_per_cb, min_markers_per_chrom)
        log.info(
            f'Removed {n - len(co_markers)} barcodes with fewer than {min_markers_per_cb} markers '
            f'or fewer than {min_markers_per_chrom} markers per chromosome'
        )

    n = len(co_markers)
    # estimate ambient marker rate for each CB and try to scrub common background markers
    log.info('Estimating background marker rates.')
    co_markers = estimate_overall_background_signal(co_markers, bg_window_size, max_frac_bg)
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
        co_markers = clean_marker_background(co_markers, rng=rng)

    if mask_imbalanced:
        # mask any bins that still have extreme imbalance
        # (e.g. due to extreme allele-specific expression differences)
        co_markers, n_masked = apply_haplotype_imbalance_mask(
            co_markers, max_marker_imbalance
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

    if output_json_fn is not None:
        log.info(f'Writing markers to {output_json_fn}')
        co_markers.write_json(output_json_fn)
    return co_markers
