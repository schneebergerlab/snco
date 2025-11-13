import numpy as np

from snco.records import MarkerRecords


def _compute_bias_factor(co_markers, hap_bias_shrinkage=0.75, bc_haplotype=0):
    if co_markers.ploidy_type == 'diploid_bc1':
        expected_ratio = (3, 1) if bc_haplotype == 0 else (1, 3)
    else:
        expected_ratio = (1, 1)
    hap_totals = np.sum([m.sum(axis=0) for m in co_markers.deep_values()], axis=0)
    expected = np.array(expected_ratio, dtype=float)
    expected /= expected.sum()
    observed = hap_totals / hap_totals.sum()
    raw_factor = observed / expected
    bias_factor = 1.0 + hap_bias_shrinkage * (raw_factor - 1.0)
    return bias_factor


def normalise_bin_coverage(co_markers, shrinkage_q=0.99, correct_hap_bias=True, hap_bias_shrinkage=0.75):
    """
    Down-weight bins with extreme coverage to reduce bias from regions with high expression,
    high marker density, or collapsed repeats.

    This function computes per-bin average coverage across all barcodes, then applies
    a shrinkage factor based on a quantile-derived threshold (`lambda`). Bins with
    coverage above the chosen quantile are scaled down, while low-coverage bins remain
    mostly unchanged. Counts are scaled per chromosome and rounded to integers.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker data structure containing count matrices for multiple barcodes and chromosomes.
        Each entry should be a 2D NumPy array of shape (bins, haplotypes).
    shrinkage_q : float, optional, default=0.99
        Quantile used to compute the shrinkage threshold for each chromosome.
        Higher values (e.g., 0.99) apply weaker normalization; lower values
        (e.g., 0.75) apply stronger normalization.
    correct_hap_bias : bool, optional, default True
        Whether to correct for reference/other bias in global counts per haplotype
    hap_bias_shrinkage: float, optional, default=0.5
        The shrinkage parameter used when correcting haplotype bias
    """
    tot = {}
    n_cb = len(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        if chrom not in tot:
            tot[chrom] = m.copy()
        else:
            tot[chrom] += m
    bin_means = {chrom: t / n_cb for chrom, t in tot.items()}
    if correct_hap_bias:
        bias_factor = _compute_bias_factor(co_markers, hap_bias_shrinkage)
    else:
        bias_factor = np.ones(shape=2)

    lambdas = {
        chrom: np.quantile(bm[bm > 0].ravel(), shrinkage_q)
        for chrom, bm in bin_means.items()
    }

    norm_factor = {}
    for chrom, bm in bin_means.items():
        f = lambdas[chrom] / (bm + lambdas[chrom])
        original_sum = bm.sum()
        new_sum = (bm * f).sum()
        scale = original_sum / new_sum if new_sum > 0 else 1.0
        f = np.minimum(1.0, f * scale)
        norm_factor[chrom] = f

    co_markers_n = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        scaled = m * norm_factor[chrom] / bias_factor[None, :]
        co_markers_n[cb, chrom] = np.round(scaled).astype(int)

    return co_markers_n


def normalise_barcode_depth(co_markers):
    total_counts = {cb: co_markers.total_marker_count(cb) for cb in co_markers.barcodes}
    norm_factor = np.median(list(total_counts.values()))
    co_markers_n = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        scaled = m / total_counts[cb] * norm_factor
        co_markers_n[cb, chrom] = np.round(scaled).astype(int)
    return co_markers_n
