import numpy as np
import pandas as pd

from ..records import MarkerRecords, NestedDataArray


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
    NestedDataArray
        metadata array of masks by genotype and chromosome.
    int
        Total number of bins masked.
    """
    imbalance_mask = NestedDataArray(levels=('genotype', 'chrom'))
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
    co_markers_m = MarkerRecords.new_like(co_markers)
    if apply_per_geno:
        genotypes = co_markers.metadata['genotypes']
    for cb, chrom, m in co_markers.deep_items():
        geno = genotypes[cb] if apply_per_geno else 'ungrouped'
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

