'''
functions for aggregating marker information from a bam file containing cell barcode, haplotype 
and optionally UMI tags, into the snco.MarkerRecords format.
'''
import logging
from functools import reduce
from collections import defaultdict
import itertools as it
from joblib import Parallel, delayed

import numpy as np
import pysam

from .utils import read_cb_whitelist, genotyping_results_formatter
from .records import MarkerRecords
from .bam import BAMHaplotypeIntervalReader, get_ha_samples, DEFAULT_EXCLUDE_CONTIGS
from .genotype import genotype_from_inv_counts, resolve_genotype_counts_to_co_markers
from .clean import filter_low_coverage_barcodes, filter_genotyping_score
from .opts import DEFAULT_RANDOM_SEED


log = logging.getLogger('snco')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def get_chrom_sizes_bam(bam_fn, exclude_contigs=None):
    """
    Load a dictionary of chromosome lengths from the header of a BAM file.

    Parameters
    ----------
    bam_fn : str
        The BAM file path.
    exclude_contigs : list of str, optional
        List of contig names to exclude (default is `DEFAULT_EXCLUDE_CONTIGS`).

    Returns
    -------
    dict
        A dictionary where keys are chromosome names and values are their respective lengths.
    """
    if exclude_contigs is None:
        exclude_contigs = DEFAULT_EXCLUDE_CONTIGS
    with pysam.AlignmentFile(bam_fn) as bam:
        chrom_sizes = {k: bam.get_reference_length(k)
                       for k in bam.references
                       if k not in exclude_contigs}
    return chrom_sizes


def single_chrom_co_markers(bam_fn, chrom, bin_start, bin_end, **kwargs):
    """
    For a single BAM file/chromosome combination, create a `MarkerRecords` object 
    storing the haplotype marker information.

    Parameters
    ----------
    bam_fn : str
        The BAM file path.
    chrom : str
        The chromosome name.
    bin_start : int
        The starting bin index for processing.
    bin_end : int
        The ending bin index for processing.
    kwargs : dict
        Additional keyword arguments passed to `BAMHaplotypeIntervalReader`.

    Returns
    -------
    list
        A list of IntervalMarkerCounts records for the specified region in the BAM file.
    """
    chrom_inv_counts = []
    with BAMHaplotypeIntervalReader(bam_fn, **kwargs) as bam:
        for bin_idx in range(bin_start, bin_end):
            chrom_inv_counts.append(bam.fetch_interval_counts(chrom, bin_idx))
    return chrom_inv_counts


def chrom_chunks(chrom_sizes, bin_size, nchunks):
    """
    Generate chunks of chromosome bins for parallel processing.

    Parameters
    ----------
    chrom_sizes : dict
        A dictionary where keys are chromosome names and values are their respective lengths.
    bin_size : int
        The bin size in base pairs.
    nchunks : int
        The number of chunks to divide the bins into.

    Yields
    ------
    tuple
        Tuples of `(chrom, start, end)` where `chrom` is the chromosome name and
        `start` and `end` are the bin indices defining a chunk.
    """
    tot = 0
    chrom_nbins = {}
    for chrom, cs in chrom_sizes.items():
        nbins = int(np.ceil(cs / bin_size))
        tot += nbins
        chrom_nbins[chrom] = nbins

    chunk_size = tot // nchunks
    for chrom, nbins in chrom_nbins.items():
        start = 0
        while start < nbins:
            end = min(start + chunk_size, nbins)
            yield chrom, start, end
            start = end


def get_co_markers(bam_fn, processes=1, seq_type=None, run_genotype=False, genotype_kwargs=None, **kwargs):
    """
    Read from a BAM file, identify reads aligning to each haplotype for each cell barcode,
    and summarize the data into a `MarkerRecords` object.

    Parameters
    ----------
    bam_fn : str
        The BAM file path.
    processes : int, optional
        The number of parallel processes to use (default is 1).
    seq_type : str, optional
        The type of sequencing data (e.g., "10x_atac", "10x_rna", "bd_rna", "takara", "wgs").
    run_genotype : bool, optional
        If True, perform genotyping of parental accessions based on interval counts (default is False).
    genotype_kwargs : dict, optional
        Additional arguments passed to the genotyping function (default is None).
    kwargs : dict
        Additional arguments passed to the BAM file processing functions.

    Returns
    -------
    MarkerRecords
        A `MarkerRecords` object containing aggregated haplotype marker information for the cell barcodes.
    """
    chrom_sizes = get_chrom_sizes_bam(bam_fn, exclude_contigs=kwargs.get('exclude_contigs', None))
    bin_size = kwargs.get('bin_size')

    log.debug(f'Starting job pool to process bam with {processes} processes')
    with Parallel(n_jobs=processes, backend='loky') as pool:
        inv_counts = pool(
            delayed(single_chrom_co_markers)(bam_fn, chrom, start, end, **kwargs)
            for chrom, start, end in chrom_chunks(chrom_sizes, bin_size, processes)
        )
        inv_counts = list(it.chain(*inv_counts))
    co_markers = MarkerRecords(
        chrom_sizes,
        bin_size,
        seq_type=seq_type,
    )

    if genotype_kwargs is None:
        genotype_kwargs = {}

    if run_genotype:
        if genotype_kwargs.get('crossing_combinations', None) is None:
            haplotypes = get_ha_samples(bam_fn)
            genotype_kwargs['crossing_combinations'] = [frozenset(g) for g in it.combinations(haplotypes, r=2)]
        if kwargs.get('hap_tag_type', 'star_diploid') != "multi_haplotype":
            raise ValueError('must use "multi_haplotype" type hap tag to perform genotyping')

        log.info(f'Genotyping barcodes with {len(genotype_kwargs["crossing_combinations"])} possible genotypes')
        genotypes, inv_counts = genotype_from_inv_counts(inv_counts, **genotype_kwargs)
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug(genotyping_results_formatter(genotypes))
        co_markers.metadata['genotypes'] = genotypes

    elif kwargs.get('hap_tag_type', 'star_diploid') == "multi_haplotype":
        if genotype_kwargs.get('crossing_combinations', None):
            if len(genotype_kwargs['crossing_combinations']) > 1:
                raise ValueError('When haplotyping is switched off, only one crossing_combination can be provided')
            geno = genotype_kwargs['crossing_combinations'].pop()
        else:
            haplotypes = get_ha_samples(bam_fn)
            if len(haplotypes) == 2:
                geno = frozenset(haplotypes)
            else:
                raise ValueError(
                    'If haplotyping is switched off and crossing_combinations are not provided, the bam file '
                    'can only contain two haplotypes'
                )
        genotypes_dummy = defaultdict(lambda: {'genotype': geno})
        inv_counts = resolve_genotype_counts_to_co_markers(inv_counts, genotypes_dummy)
    for ic in inv_counts:
        co_markers.update(ic)
    return co_markers


def run_loadbam(bam_fn, output_json_fn, *,
                cb_whitelist_fn=None, bin_size=25_000, seq_type=None,
                cb_tag='CB', cb_correction_method='exact',
                umi_tag='UB', umi_collapse_method='directional',
                hap_tag='ha', hap_tag_type='star_diploid',
                run_genotype=False, genotype_crossing_combinations=None,
                genotype_em_max_iter=1000, genotype_em_min_delta=1e-3,
                genotype_em_bootstraps=25,
                min_markers_per_cb=100, min_markers_per_chrom=20, min_geno_prob=0.9,
                exclude_contigs=None, processes=1, rng=DEFAULT_RNG):
 """
    Read a BAM file with cell barcode, UMI, and haplotype tags, and generate a JSON file 
    containing binned haplotype marker distributions for each cell barcode.

    Parameters
    ----------
    bam_fn : str
        The BAM file path.
    output_json_fn : str
        The output JSON file path to store the processed marker records.
    cb_whitelist_fn : str, optional
        Path to a cell barcode whitelist file (default is None).
    bin_size : int, optional
        The size of each bin in base pairs (default is 25,000).
    seq_type : str, optional
        The type of sequencing data (default is None).
    cb_tag : str, optional
        The tag used to identify cell barcodes in the BAM file (default is 'CB').
    cb_correction_method : str, optional
        Method for correcting cell barcode tags (default is 'exact').
    umi_tag : str, optional
        The tag used to identify UMIs in the BAM file (default is 'UB').
    umi_collapse_method : str, optional
        Method for collapsing UMIs (default is 'directional').
    hap_tag : str, optional
        The tag used to identify haplotypes in the BAM file (default is 'ha').
    hap_tag_type : str, optional
        The haplotype tag type (default is 'star_diploid').
    run_genotype : bool, optional
        If True, perform genotyping (default is False).
    genotype_crossing_combinations : list of frozenset, optional
        List of allowed crossing combinations for genotyping (default is None).
    genotype_em_max_iter : int, optional
        Maximum number of EM iterations for genotyping (default is 1000).
    genotype_em_min_delta : float, optional
        Minimum change in probabilities for convergence (default is 1e-3).
    genotype_em_bootstraps : int, optional
        Number of bootstrap re-samples for genotyping (default is 25).
    min_markers_per_cb : int, optional
        Minimum number of markers required per barcode (default is 100).
    min_markers_per_chrom : int, optional
        Minimum number of markers required per chromosome (default is 20).
    min_geno_prob : float, optional
        Minimum genotyping probability, barcodes with lower probabilities are filtered
        (default is 0.9).
    exclude_contigs : list of str, optional
        List of contig names to exclude (default is None).
    processes : int, optional
        Number of parallel processes to use (default is 1).
    rng : np.random.Generator, optional
        Random number generator (default is `DEFAULT_RNG`).

    Returns
    -------
    MarkerRecords
        A `MarkerRecords` object containing processed haplotype marker distributions.
    """
    cb_whitelist = read_cb_whitelist(cb_whitelist_fn, cb_correction_method)
    co_markers = get_co_markers(
        bam_fn, processes=processes,
        bin_size=bin_size,
        seq_type=seq_type,
        cb_tag=cb_tag,
        umi_tag=umi_tag,
        umi_collapse_method=umi_collapse_method,
        hap_tag=hap_tag,
        hap_tag_type=hap_tag_type,
        run_genotype=run_genotype,
        genotype_kwargs={
            'crossing_combinations': genotype_crossing_combinations,
            'max_iter': genotype_em_max_iter,
            'min_delta': genotype_em_min_delta,
            'n_bootstraps': genotype_em_bootstraps,
            'min_markers_per_cb': min_markers_per_cb,
            'processes': processes,
            'rng': rng
        },
        cb_whitelist=cb_whitelist,
        exclude_contigs=exclude_contigs,
    )
    n = len(co_markers)
    log.info(f'Identified {n} cell barcodes from bam file')
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
