'''
Functions for converting cellsnp-lite output into snco.MarkerRecords format
'''
import os
import logging
import itertools as it

import numpy as np
from scipy.io import mmread

import pysam

from .utils import read_cb_whitelist, genotyping_results_formatter
from .records import MarkerRecords
from .bam import IntervalMarkerCounts
from .genotype import genotype_from_inv_counts
from .clean import filter_low_coverage_barcodes, filter_genotyping_score
from .opts import DEFAULT_RANDOM_SEED


log = logging.getLogger('snco')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def read_chrom_sizes(chrom_sizes_fn):
    """
    Load a dictionary of chromosome lengths from a 2-column text file or a faidx file.

    Parameters
    ----------
    chrom_sizes_fn : str
        The file path to the chromosome size file. It should be a text file where each line
        contains a chromosome name and its size, separated by a tab.

    Returns
    -------
    dict
        A dictionary where the keys are chromosome names (str) and the values are chromosome sizes (int).
    """
    chrom_sizes = {}
    with open(chrom_sizes_fn) as f:
        for record in f:
            chrom, cs = record.strip().split('\t')[:2]
            chrom_sizes[chrom] = int(cs)
    return chrom_sizes


def get_vcf_samples(vcf_fn, ref_name):
    """
    Extract the list of sample names from a VCF file and add the reference name.

    Parameters
    ----------
    vcf_fn : str
        Path to the VCF file.
    ref_name : str
        The name of sample used as the reference genome to add to the list of samples.

    Returns
    -------
    frozenset
        A frozenset containing the sample names (including the reference sample).
    """
    with pysam.VariantFile(vcf_fn) as vcf:
        samples = set(vcf.header.samples)
    samples.add(ref_name)
    return frozenset(samples)


def parse_sample_alleles(variant, ref_name):
    """
    Parse the alleles for each sample in a variant record and assign them to reference and alternate groups.

    Parameters
    ----------
    variant : pysam.VariantRecord
        The VCF variant record to process.
    ref_name : str
        The reference name for the variant.

    Returns
    -------
    tuple
        A tuple of two frozensets:
        - The first frozenset contains the samples that have the reference allele.
        - The second frozenset contains the samples that have the alternate allele.

    Raises
    ------
    ValueError
        If the variant is not biallelic or if any sample has more than one allele.
    """
    if len(variant.alleles) != 2:
        raise ValueError('Only biallelic variants are allowed')
    ref_samples = set()
    ref_samples.add(ref_name)
    alt_samples = set()
    for sample_id, sample in variant.samples.items():
        if len(sample.alleles) > 1:
            raise ValueError('Only haploid haplotype calls are allowed for each sample')
        if not sample.allele_indices[0]:
            ref_samples.add(sample_id)
        else:
            alt_samples.add(sample_id)
    return frozenset(ref_samples), frozenset(alt_samples)


class VariantRecords:

    """
    A class to store variant records and associated sample allele information.

    Methods
    ----------
    add(contig, pos, sample_alleles=None)
        Add a variant record with its sample alleles to the VariantRecords object.
    get_samples(contig, pos)
        Get the sample alleles for a specific variant.
    """

    def __init__(self):
        """
        Initialize an empty VariantRecords object.
        """
        self._records = []
        self._samples = {}

    def add(self, contig, pos, sample_alleles=None):
        """
        Add a variant with its sample alleles to the VariantRecords object.

        Parameters
        ----------
        contig : str
            The name of the contig (chromosome).
        pos : int
            The position of the variant on the contig.
        sample_alleles : tuple, optional
            A tuple containing two frozensets: one for the reference allele samples
            and one for the alternate allele samples. Default is None.
        """
        self._records.append((contig, pos))
        self._samples[(contig, pos)] = sample_alleles

    def __getitem__(self, index):
        return self._records[index]

    def get_samples(self, contig, pos):
        """
        Get the samples with ref and alt alleles for a specific variant.

        Parameters
        ----------
        contig : str
            The name of the contig (chromosome).
        pos : int
            The position of the variant.

        Returns
        -------
        tuple
            A tuple containing two frozensets: one for the reference allele samples
            and one for the alternate allele samples.
        """
        return self._samples[(contig, pos)]


def read_vcf(vcf_fn, drop_samples=True, reference_name='col0'):
    """
    Read a VCF file and parse the variants into VariantRecords.

    Parameters
    ----------
    vcf_fn : str
        Path to the VCF file to read.
    drop_samples : bool, optional
        Whether to exclude sample data (default is True).
    reference_name : str, optional
        The name of the reference sample to include in the records (default is 'col0').

    Returns
    -------
    VariantRecords
        A VariantRecords object containing the parsed variant data.
    """
    with pysam.VariantFile(vcf_fn, drop_samples=drop_samples) as vcf:
        variants = VariantRecords()
        for record in vcf.fetch():
            if drop_samples:
                sample_alleles = None
            else:
                try:
                    sample_alleles = parse_sample_alleles(record, reference_name)
                except ValueError:
                    continue
            variants.add(record.contig, record.pos, sample_alleles)
    return variants


def parse_cellsnp_lite(csl_dir, validate_barcodes=True):
    """
    Parse the output of cellsnp-lite into sparse matrices.

    Parameters
    ----------
    csl_dir : str
        Path to the directory containing the cellsnp-lite output files.
    validate_barcodes : bool, optional
        Whether to validate the barcodes (default is True).

    Returns
    -------
    dep_mm : scipy.sparse matrix
        The depth matrix.
    alt_mm : scipy.sparse matrix
        The alternate allele matrix.
    barcodes : list of str
        The list of cell barcodes.
    variants : VariantRecords
        The parsed variant records.
    """
    dep_fn = os.path.join(csl_dir, 'cellSNP.tag.DP.mtx')
    alt_fn = os.path.join(csl_dir, 'cellSNP.tag.AD.mtx')
    vcf_fn = os.path.join(csl_dir, 'cellSNP.base.vcf')
    if not os.path.exists(vcf_fn):
        vcf_fn = f'{vcf_fn}.gz'
    barcode_fn = os.path.join(csl_dir, 'cellSNP.samples.tsv')

    dep_mm = mmread(dep_fn)
    alt_mm = mmread(alt_fn).tocsr()
    barcodes = read_cb_whitelist(barcode_fn, validate_barcodes=validate_barcodes)

    # cellsnp-lite vcf files have some missing definitions in the header
    # setting pysam verbosity to 0 prevents warnings
    prev_verbosity = pysam.set_verbosity(0)
    variants = read_vcf(vcf_fn)
    pysam.set_verbosity(prev_verbosity)

    return dep_mm, alt_mm, barcodes, variants


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


def run_loadcsl(cellsnp_lite_dir, chrom_sizes_fn, output_json_fn, *,
                cb_whitelist_fn=None, bin_size=25_000, snp_counts_only=False,
                run_genotype=False, genotype_vcf_fn=None,
                genotype_crossing_combinations=None, reference_genotype_name='col0',
                genotype_em_max_iter=1000, genotype_em_min_delta=1e-3, genotype_em_bootstraps=25,
                min_markers_per_cb=100, min_markers_per_chrom=20, min_geno_prob=0.9,
                validate_barcodes=True, processes=1, rng=DEFAULT_RNG):
    """
    Reads matrix files generated by cellSNP-lite to generate a JSON file of binned
    haplotype marker distributions for each cell barcode. These can be used to
    call recombinations using the downstream `predict` command.

    Parameters
    ----------
    cellsnp_lite_dir : str
        Directory containing the cellSNP-lite output files.
    chrom_sizes_fn : str
        Path to a file containing chromosome sizes.
    output_json_fn : str
        Path to the output JSON file where the marker data will be saved.
    cb_whitelist_fn : str, optional
        Path to a file containing a list of cell barcodes to be used (default is None).
    bin_size : int, optional
        The bin size for partitioning the genome into intervals (default is 25,000).
    snp_counts_only : bool, optional
        Whether to only count SNPs, instead of the number of reads per SNP (default is False).
    run_genotype : bool, optional
        If True, performs genotyping using the provided VCF files (default is False).
    genotype_vcf_fn : str, optional
        Path to the VCF file for genotyping. Required if `run_genotype` is True.
    genotype_crossing_combinations : list, optional
        List of genotype crossing combinations for the genotyping process (default is None).
    reference_genotype_name : str, optional
        The name of the reference sample in the VCF file (default is 'col0').
    genotype_em_max_iter : int, optional
        Maximum number of iterations for the EM algorithm during genotyping (default is 1000).
    genotype_em_min_delta : float, optional
        Minimum change in probability for EM convergence (default is 1e-3).
    genotype_em_bootstraps : int, optional
        Number of bootstrap samples for genotyping (default is 25).
    min_markers_per_cb : int, optional
        Minimum number of markers required for each cell barcode to be included (default is 100).
    min_markers_per_chrom : int, optional
        Minimum number of markers required for each chromosome of a barcode to be included (default is 20).
    min_geno_prob : float, optional
        Minimum genotyping probability required for each barcode to be included (default is 0.9).
    validate_barcodes : bool, optional
        If True, validates barcodes against the whitelist (default is True).
    processes : int, optional
        Number of processes to use for parallel processing - only used for genotyping (default is 1).
    rng : numpy.random.Generator, optional
        Random number generator instance for reproducibility (default is the global default RNG).

    Returns
    -------
    MarkerRecords
        A MarkerRecords object containing haplotype marker distributions for each cell barcode.
    """

    cb_whitelist = read_cb_whitelist(cb_whitelist_fn, validate_barcodes=validate_barcodes)
    co_markers = cellsnp_lite_to_co_markers(
        cellsnp_lite_dir,
        chrom_sizes_fn,
        bin_size=bin_size,
        cb_whitelist=cb_whitelist,
        validate_barcodes=validate_barcodes,
        run_genotype=run_genotype,
        genotype_vcf_fn=genotype_vcf_fn,
        reference_name=reference_genotype_name,
        genotype_kwargs={
            'crossing_combinations': genotype_crossing_combinations,
            'max_iter': genotype_em_max_iter,
            'min_delta': genotype_em_min_delta,
            'n_bootstraps': genotype_em_bootstraps,
            'min_markers_per_cb': min_markers_per_cb,
            'processes': processes,
            'rng': rng
        },
    )
    n = len(co_markers)
    log.info(f'Identified {n} cell barcodes from cellsnp-lite files')
    if min_markers_per_cb or min_markers_per_chrom:
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
    if output_json_fn is not None:
        log.info(f'Writing markers to {output_json_fn}')
        co_markers.write_json(output_json_fn)
    return co_markers
