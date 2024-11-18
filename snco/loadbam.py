'''
functions for aggregating marker information from a bam file containing cell barcode, haplotype 
and optionally UMI tags, into the snco.MarkerRecords format.
'''
import logging
from functools import reduce
from joblib import Parallel, delayed

import numpy as np
import pysam

from .utils import read_cb_whitelist
from .records import MarkerRecords
from .bam import BAMHaplotypeIntervalReader, DEFAULT_EXCLUDE_CONTIGS
from .clean import filter_low_coverage_barcodes

log = logging.getLogger('snco')


def get_chrom_sizes_bam(bam_fn, exclude_contigs=None):
    '''
    load dict of chromosome lengths from the header of a bam file
    '''
    if exclude_contigs is None:
        exclude_contigs = DEFAULT_EXCLUDE_CONTIGS
    with pysam.AlignmentFile(bam_fn) as bam:
        chrom_sizes = {k: bam.get_reference_length(k)
                       for k in bam.references
                       if k not in exclude_contigs}
    return chrom_sizes


def single_chrom_co_markers(bam_fn, chrom, bin_start, bin_end, seq_type=None, **kwargs):
    '''
    For a single bam file/chrom combination, create a MarkerRecords object
    storing the haplotype marker information.
    '''
    with BAMHaplotypeIntervalReader(bam_fn, **kwargs) as bam:
        chrom_co_markers = MarkerRecords(
            bam.chrom_sizes,
            bam.bin_size,
            seq_type=seq_type,
        )
        for bin_idx in range(bin_start, bin_end):
            chrom_co_markers += bam.fetch_interval_counts(chrom, bin_idx)
    return chrom_co_markers


def chrom_chunks(chrom_sizes, bin_size, nchunks):
    '''produce approximately nchunks slices covering all the bins in chrom_nbins'''
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


def get_co_markers(bam_fn, processes=1, **kwargs):
    '''
    Read from a bam file, identify reads aligning to each haplotype for each cell barcode,
    and summarise into a MarkerRecords object.
    '''
    chrom_sizes = get_chrom_sizes_bam(bam_fn, exclude_contigs=kwargs.get('exclude_contigs', None))
    bin_size = kwargs.get('bin_size')
    log.debug(f'Starting job pool to process bam with {processes} processes')
    with Parallel(n_jobs=processes, backend='loky') as pool:
        co_markers = pool(
            delayed(single_chrom_co_markers)(bam_fn, chrom, start, end, **kwargs)
            for chrom, start, end in chrom_chunks(chrom_sizes, bin_size, processes)
        )
    co_markers = reduce(MarkerRecords.merge, co_markers)
    return co_markers


def run_loadbam(bam_fn, output_json_fn, *,
                cb_whitelist_fn=None, bin_size=25_000, seq_type=None,
                cb_tag='CB', cb_correction_method='exact',
                umi_tag='UB', umi_collapse_method='directional',
                hap_tag='ha', min_markers_per_cb=100, min_markers_per_chrom=20,
                exclude_contigs=None, processes=1):
    '''
    Read bam file with cell barcode, umi and haplotype tags (aligned with STAR solo+diploid), 
    to generate a json file of binned haplotype marker distributions for each cell barcode. 
    These can be used to call recombinations using the downstream `predict` command.
    '''
    cb_whitelist = read_cb_whitelist(cb_whitelist_fn, cb_correction_method)
    co_markers = get_co_markers(
        bam_fn, processes=processes,
        bin_size=bin_size,
        seq_type=seq_type,
        cb_tag=cb_tag,
        umi_tag=umi_tag,
        umi_collapse_method=umi_collapse_method,
        hap_tag=hap_tag,
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
    if output_json_fn is not None:
        log.info(f'Writing markers to {output_json_fn}')
        co_markers.write_json(output_json_fn)
    return co_markers
