import numpy as np
from pomegranate import distributions as pmd

from snco.records import MarkerRecords
from snco.predict.rhmm.dists import ZeroInflated


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


def _estimate_zip_lambda(cb_co_markers):
    m = np.concatenate(list(cb_co_markers.values())).sum(axis=1)
    zip_ = ZeroInflated(
        pmd.Poisson([m[m!=0].mean(),]),
        priors=[(m == 0).mean()]
    ).fit(m.reshape(-1, 1))
    return zip_.distribution.lambdas.numpy()[0]


def normalise_barcode_depth(co_markers, min_norm_factor=2.0):
    """
    Normalise per-barcode marker depths using ZIP-estimated expected counts.

    This scales each barcode’s marker counts by an estimated Poisson mean
    (λ) obtained from a zero-inflated Poisson fit to its concatenated
    marker counts. All barcodes are rescaled to a common target depth
    defined as the median of the estimated λ values, with an enforced
    minimum (`min_norm_factor`). Counts are rounded to integers.

    Parameters
    ----------
    co_markers : MarkerRecords
        Input marker record object containing per-barcode marker count
        arrays.
    min_norm_factor : float, optional
        Lower bound on the global normalisation factor. Ensures that very
        low median λ estimates do not collapse all counts. Default is 2.0.

    Returns
    -------
    MarkerRecords
        A new ``MarkerRecords`` instance with normalised marker counts for
        each barcode and chromosome.

    Notes
    -----
    For each barcode, the expected depth λ is estimated from all
    concatenated marker counts using a zero-inflated Poisson model. The
    final scaling applied is:

        scaled = counts / λ_cb * max(median(λ), min_norm_factor)

    Where ``λ_cb`` is the estimated Poisson mean for barcode ``cb``.
    """
    lambdas = {cb: _estimate_zip_lambda(co_markers[cb]) for cb in co_markers.barcodes}
    norm_factor = np.maximum(np.median(list(lambdas.values())), min_norm_factor)
    co_markers_n = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        scaled = m / lambdas[cb] * norm_factor
        co_markers_n[cb, chrom] = np.round(scaled).astype(int)
    return co_markers_n