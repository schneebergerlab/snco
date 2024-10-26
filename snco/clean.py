import numpy as np
from scipy.ndimage import convolve1d

from .utils import load_json
from .records import MarkerRecords


def predict_foreground_convolution(m, ws=100):
    rs = convolve1d(m, np.ones(ws), axis=0, mode='constant', cval=0)
    fg_idx = rs.argmax(axis=1)
    return fg_idx


def _estimate_marker_background(m, ws=100):
    fg_idx = predict_foreground_convolution(m, ws)
    fg_masked = m.copy()
    fg_masked[np.arange(len(m)), fg_idx] = 0
    return fg_masked


def estimate_overall_background_signal(co_markers, conv_window_size):
    conv_bins = conv_window_size // co_markers.bin_size
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
    co_markers.metadata['background_signal'] = bg_signal
    co_markers.metadata['estimated_background_fraction'] = frac_bg
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
    return m_sub, bg


def clean_marker_background(co_markers, conv_filter_size):
    bg_signal, frac_bg = estimate_overall_background_signal(co_markers, conv_filter_size)
    co_markers_c = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        co_markers_c[cb, chrom] = subtract_background(m, bg_signal[chrom], frac_bg[cb])
    return co_markers_c


def create_haplotype_imbalance_mask(co_markers, max_imbalance_mask=0.9):
    tot_signal = {}
    for _, chrom, m in co_markers.deep_items():
        if chrom not in tot_signal:
            tot_signal[chrom] = m.copy()
        else:
            tot_signal[chrom] += m

    imbalance_mask = {}
    for chrom, m in tot_signal.items():
        with np.errstate(invalid='ignore'):
            bin_sum = m.sum(axis=1)
            ratio = m[:, 0] / bin_sum
        np.nan_to_num(ratio, nan=0.5, copy=False)
        mask = (ratio > max_imbalance_mask) | (ratio < (1 - max_imbalance_mask))
        imbalance_mask[chrom] = np.stack([mask, mask], axis=1)
    co_markers.metadata['haplotype_imbalance_mask'] = imbalance_mask
    return imbalance_mask


def apply_haplotype_imbalance_mask(co_markers, max_imbalance_mask=0.9):
    mask = create_haplotype_imbalance_mask(co_markers, max_imbalance_mask)
    co_markers_m = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        co_markers_m[cb, chrom] = np.where(mask[chrom], 0, m)
    return co_markers_m


def apply_marker_threshold(co_markers, max_marker_threshold):
    co_markers_t = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        co_markers_t[cb, chrom] = np.minimum(m, max_marker_threshold)
    return co_markers_t


def run_clean(marker_json_fn, output_json_fn, *,
              cb_whitelist_fn=None, bin_size=25_000,
              max_bin_count=20, bg_window_size=2_500_000,
              max_marker_imbalance=0.9):
    '''
    Removes predicted background markers, that result from ambient nucleic acids, 
    from each cell barcode.
    '''
    co_markers = load_json(marker_json_fn, cb_whitelist_fn, bin_size)

    # first estimate ambient marker rate for each CB and try to scrub common background markers
    co_markers_c = clean_marker_background(co_markers, bg_window_size)
    # next mask any bins that still have extreme imbalance
    # (e.g. due to extreme allele-specific expression differences)
    co_markers_m = apply_haplotype_imbalance_mask(co_markers_c, max_marker_imbalance)
    # finally threshold bins that have a large number of reads
    co_markers_t = apply_marker_threshold(co_markers_m, max_bin_count)

    co_markers_t.write_json(output_json_fn)
