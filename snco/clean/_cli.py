import logging

import numpy as np

from .background import estimate_overall_background_signal, clean_marker_background
from .filter import filter_low_coverage_barcodes, filter_genotyping_score
from .mask import apply_haplotype_imbalance_mask, apply_marker_threshold, mask_regions_bed
from ..utils import load_json
from ..opts import DEFAULT_RANDOM_SEED


log = logging.getLogger('snco')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


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
