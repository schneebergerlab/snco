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


def filter_genotyping_score(co_markers, min_geno_prob=0.9, max_geno_error_rate=0.25):
    """
    Remove barcodes with genotyping probability below a given threshold.

    Parameters
    ----------
    co_markers : MarkerRecords
        Marker records with metadata containing genotyping probabilities.
    min_geno_prob : float, default=0.9
        Minimum allowed genotype probability.
    max_geno_error_rate : float, default=0.25
        The maximum inferred background noise rate from genotyping

    Returns
    -------
    MarkerRecords
        Filtered marker records.
    """
    try:
        geno_probs = co_markers.metadata['genotype_probability']
        genotype_error_rates = co_markers.metadata['genotype_error_rates']
    except KeyError:
        return co_markers

    def _geno_query(cb):
        return (geno_probs[cb] >= min_geno_prob) & (genotype_error_rates[cb] <= max_geno_error_rate)

    return co_markers.query(_geno_query)
