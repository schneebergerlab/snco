import logging
import itertools as it
import numpy as np

from .utils import read_chrom_sizes
from .csl import parse_cellsnp_lite
from .vcf import read_vcf, get_vcf_samples
from ..counts import IntervalMarkerCounts
from ..genotype import genotype_from_inv_counts
from ..utils import genotyping_results_formatter
from snco.records import MarkerRecords
from snco.defaults import DEFAULT_RANDOM_SEED


log = logging.getLogger('snco')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def parse_cellsnp_lite_interval_counts(csl_dir, bin_size, cb_whitelist,
                                       snp_counts_only=False,
                                       keep_genotype=False,
                                       genotype_vcf_fn=None,
                                       validate_barcodes=True,
                                       reference_name='col0'):
    """
    Parse cellsnp-lite data into IntervalMarkerCounts for genotyping and recombination analysis.

    Parameters
    ----------
    csl_dir : str
        Path to the cellsnp-lite directory containing the data files.
    bin_size : int
        The bin size for the intervals.
    cb_whitelist : set
        A set of valid cell barcodes to use.
    snp_counts_only : bool, optional
        Whether to only count SNPs, instead of the number of reads per SNP (default is False).
    keep_genotype : bool, optional
        Whether to store genotype data for later genotyping of barcodes (default is False).
    genotype_vcf_fn : str, optional
        Path to the genotype VCF file (required if `keep_genotype` is True).
    validate_barcodes : bool, optional
        Whether to validate the barcodes (default is True).
    reference_name : str, optional
        The reference name for the genotypes (default is 'col0').

    Returns
    -------
    list of IntervalMarkerCounts
        A list of `IntervalMarkerCounts` objects containing the parsed interval counts for each barcode.

    Raises
    ------
    ValueError
        If `run_genotype` is True but `genotype_vcf_fn` is not supplied.
    """
    dep_mm, alt_mm, barcodes, variants = parse_cellsnp_lite(
        csl_dir, validate_barcodes=validate_barcodes
    )
    if keep_genotype:
        if genotype_vcf_fn is None:
            raise ValueError('must supply genotype_vcf_fn when using run_genotype')
        genotype_alleles = read_vcf(
            genotype_vcf_fn,
            drop_samples=False,
            reference_name=reference_name
        )
    else:
        genotype_alleles = None

    inv_counts = {}

    for cb_idx, var_idx, tot in zip(dep_mm.col, dep_mm.row, dep_mm.data):
        cb = barcodes[cb_idx]
        if cb in cb_whitelist:
            alt = alt_mm[var_idx, cb_idx]
            ref = tot - alt
            if snp_counts_only:
                if alt > ref:
                    alt = 1.0
                    ref = 0.0
                elif ref > alt:
                    alt = 0.0
                    ref = 1.0
                else:
                    continue
            chrom, pos = variants[var_idx]
            bin_idx = pos // bin_size
            inv_counts_idx = (chrom, bin_idx)
            if inv_counts_idx not in inv_counts:
                inv_counts[inv_counts_idx] = IntervalMarkerCounts(chrom, bin_idx)
            if genotype_alleles is None:
                inv_counts[inv_counts_idx][cb][0] += ref
                inv_counts[inv_counts_idx][cb][1] += alt
            else:
                ref_genos, alt_genos = genotype_alleles.get_samples(chrom, pos)
                inv_counts[inv_counts_idx][cb][ref_genos] += ref
                inv_counts[inv_counts_idx][cb][alt_genos] += alt
    return list(inv_counts.values())


def cellsnp_lite_to_co_markers(csl_dir, chrom_sizes_fn, bin_size, cb_whitelist,
                               validate_barcodes=True, snp_counts_only=False,
                               run_genotype=False, genotype_vcf_fn=None,
                               reference_name='col0', genotype_kwargs=None):
    """
    Converts cellSNP-lite output into a MarkerRecords object, which represents
    haplotype marker distributions for each cell barcode. Optionally, genotyping 
    can be performed based on provided VCF files.

    Parameters
    ----------
    csl_dir : str
        Directory containing the cellSNP-lite output files.
    chrom_sizes_fn : str
        Path to a tab-separated file containing chromosome sizes.
    bin_size : int
        The bin size for partitioning the genome into intervals.
    cb_whitelist : set
        A set of cell barcodes to include in the analysis.
    validate_barcodes : bool, optional
        If True, validates sequenced barcodes to remove homopolymers and Ns (default is True).
    snp_counts_only : bool, optional
        Whether to only count SNPs, instead of the number of reads per SNP (default is False).
    run_genotype : bool, optional
        If True, performs genotyping of each barcode using the provided VCF files (default is False).
    genotype_vcf_fn : str, optional
        Path to the VCF file for genotyping. Required if `run_genotype` is True.
    reference_name : str, optional
        The name of the sample used as the reference genome (default is 'col0').
    genotype_kwargs : dict, optional
        Additional parameters for the genotyping process, such as crossing combinations 
        and EM algorithm settings (see `snco.genotype.assign_genotype_with_em`).

    Returns
    -------
    MarkerRecords
        A MarkerRecords object containing haplotype marker distributions for each cell barcode.

    Raises
    ------
    ValueError
        If `run_genotype` is True but `genotype_vcf_fn` is not supplied.
    """
    inv_counts = parse_cellsnp_lite_interval_counts(
        csl_dir, bin_size, cb_whitelist,
        snp_counts_only=snp_counts_only,
        keep_genotype=run_genotype,
        genotype_vcf_fn=genotype_vcf_fn,
        validate_barcodes=validate_barcodes,
        reference_name=reference_name,
    )

    chrom_sizes = read_chrom_sizes(chrom_sizes_fn)
    co_markers = MarkerRecords(
        chrom_sizes,
        bin_size,
        seq_type='csl_snps',
    )

    if run_genotype:
        if genotype_kwargs is None:
            genotype_kwargs = {}
        if genotype_kwargs.get('crossing_combinations', None) is None:
            haplotypes = get_vcf_samples(genotype_vcf_fn, reference_name)
            genotype_kwargs['crossing_combinations'] = [frozenset(g) for g in it.combinations(haplotypes, r=2)]
        log.info(f'Genotyping barcodes with {len(genotype_kwargs["crossing_combinations"])} possible genotypes')
        genotypes, genotype_probs, genotype_nmarkers, inv_counts = genotype_from_inv_counts(inv_counts, **genotype_kwargs)
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug(genotyping_results_formatter(genotypes))
        co_markers.add_metadata(
            genotypes=genotypes,
            genotype_probability=genotype_probs,
            genotyping_nmarkers=genotype_nmarkers
        )

    for ic in inv_counts:
        co_markers.update(ic)

    return co_markers