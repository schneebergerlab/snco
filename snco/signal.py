import numpy as np
from scipy.ndimage import convolve1d


def predict_foreground_convolution(m, ws=100):
    rs = convolve1d(m, np.ones(ws), axis=0, mode='constant', cval=0)
    fg_idx = rs.argmax(axis=1)
    return fg_idx


def _estimate_marker_background(m, ws=100):
    fg_idx = predict_foreground_convolution(m, ws)
    fg_masked = m.copy()
    fg_masked[np.arange(len(m)), fg_idx] = 0
    return fg_masked


def estimate_overall_background_signal(co_markers, bin_size, conv_window_size):
    conv_bins = conv_window_size // bin_size
    bg_signal = {}
    frac_bg = {}
    for cb, cb_co_markers in co_markers.items():
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

    bg_signal = {
        chrom: sig / sig.sum(axis=None)
        for chrom, sig in bg_signal.items()
    }
    
    return bg_signal, frac_bg


def mask_bg_signal(bg_signal, m):
    bg_signal = np.where(m, bg_signal, 0)
    bg_signal = bg_signal / bg_signal.sum()
    return bg_signal


def subtract_background(m, bg_signal, frac_bg, return_bg=False):
    tot = m.sum(axis=None)
    n_bg = round(tot * frac_bg)
    bg = mask_bg_signal(bg_signal, m) * n_bg
    bg = np.round(np.minimum(m, bg))
    m_sub = m - bg
    if not return_bg:
        return m_sub
    else:
        return m_sub, bg


def clean_marker_background(co_markers, bin_size, conv_filter_size):
    bg_signal, frac_bg = estimate_overall_background_signal(co_markers, bin_size, conv_filter_size)
    co_markers_c = {}
    for cb, cb_co_markers in co_markers.items():
        co_markers_c[cb] = {}
        for chrom, m in cb_co_markers.items():
            co_markers_c[cb][chrom] = subtract_background(m, bg_signal[chrom], frac_bg[cb])
    return co_markers_c


def create_haplotype_imbalance_mask(co_markers, max_imbalance_mask=0.9):
    tot_signal = {}
    for cb, cb_co_markers in co_markers.items():
        for chrom, m in cb_co_markers.items():
            if chrom not in tot_signal:
                tot_signal[chrom] = m
            else:
                tot_signal[chrom] += m

    imbalance_mask = {}
    for chrom, m in tot_signal.items():
        ratio = m[:, 0] / m.sum(axis=1)
        mask = (ratio > max_imbalance_mask) | (ratio < (1 - max_imbalance_mask))
        mask = np.tile(mask, 2).reshape(-1, 2)
        imbalance_mask[chrom] = mask
    return imbalance_mask


def apply_haplotype_imbalance_mask(co_markers, max_imbalance_mask=0.9):
    mask = create_haplotype_imbalance_mask(co_markers, max_imbalance_mask)
    co_markers_m = {}
    for cb, cb_co_markers in co_markers.items():
        co_markers_m[cb] = {}
        for chrom, m in cb_co_markers.items():
            co_markers_m[cb][chrom] = np.where(mask[chrom], 0, m)
    return co_markers_m


def apply_marker_threshold(co_markers, max_marker_threshold):
    co_markers_t = {}
    for cb, cb_co_markers in co_markers.items():
        co_markers_t[cb] = {}
        for chrom, m in cb_co_markers.items():
            co_markers_t[cb][chrom] = np.minimum(m, max_marker_threshold)
    return co_markers_t