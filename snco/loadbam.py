'''
functions for aggregating marker information from a bam file containing cell barcode, haplotype 
and optionally UMI tags, into the snco.MarkerRecords format.
'''
import logging
from functools import reduce
from joblib import Parallel, delayed

import pysam

from .utils import read_cb_whitelist
from .records import MarkerRecords
from .bam import BAMHaplotypeIntervalReader, DEFAULT_EXCLUDE_CONTIGS

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


def single_chrom_co_markers(bam_fn, chrom, seq_type=None, **kwargs):
    '''
    For a single bam file/chrom combination, create a MarkerRecords object
    storing the haplotype marker information.
    '''
    with BAMHaplotypeIntervalReader(bam_fn, **kwargs) as bam:
        chrom_co_markers = MarkerRecords(
            bam.chrom_sizes,
            bam.bin_size,
            bam.cb_whitelist.toset(),
            seq_type=seq_type,
        )
        for bin_idx in range(bam.nbins[chrom]):
            chrom_co_markers += bam.fetch_interval_counts(chrom, bin_idx)
    return chrom_co_markers


def get_co_markers(bam_fn, processes=1, **kwargs):
    '''
    Read from a bam file, identify reads aligning to each haplotype for each cell barcode,
    and summarise into a MarkerRecords object.
    '''
    chrom_sizes = get_chrom_sizes_bam(bam_fn, exclude_contigs=kwargs.get('exclude_contigs', None))
    # todo: better parallelisation (by bin not just by chromosome)
    processes = min(processes, len(chrom_sizes))
    log.debug(f'Starting job pool to process bam with {processes} processes')
    with Parallel(n_jobs=processes, backend='loky') as pool:
        co_markers = pool(
            delayed(single_chrom_co_markers)(bam_fn, chrom, **kwargs)
            for chrom in chrom_sizes
        )
    co_markers = reduce(MarkerRecords.merge, co_markers)
    return co_markers


def run_loadbam(bam_fn, output_json_fn, *,
                cb_whitelist_fn=None, bin_size=25_000, seq_type=None,
                cb_tag='CB', cb_correction_method='exact',
                umi_tag='UB', umi_collapse_method='directional',
                hap_tag='ha', exclude_contigs=None, processes=1):
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
    log.info(f'Identified {len(co_markers)} cell barcodes from bam file')
    log.info(f'Writing markers to {output_json_fn}')
    co_markers.write_json(output_json_fn)
