from functools import reduce
from joblib import Parallel, delayed

from .records import MarkerRecords
from .bam import BAMHaplotypeIntervalReader, get_chrom_sizes_bam


def single_chrom_co_markers(bam_fn, chrom, **kwargs):
    '''
    For a single bam file/chrom combination, create a MarkerRecords object
    storing the haplotype marker information.
    '''
    with BAMHaplotypeIntervalReader(bam_fn, **kwargs) as bam:
        chrom_co_markers = MarkerRecords(
            bam.chrom_sizes,
            bam.bin_size,
            bam.cb_whitelist.toset()
        )
        for bin_idx in range(bam.nbins[chrom]):
            chrom_co_markers += bam.fetch_interval_counts(chrom, bin_idx)
    return chrom_co_markers


def get_co_markers(bam_fn, processes=1, **kwargs):
    chrom_sizes = get_chrom_sizes_bam(bam_fn, exclude_contigs=kwargs.get('exclude_contigs', None))
    # todo: better parallelisation (by bin not just by chromosome)
    with Parallel(n_jobs=min(processes, len(chrom_sizes)), backend='loky') as pool:
        co_markers = pool(
            delayed(single_chrom_co_markers)(bam_fn, chrom, **kwargs)
            for chrom in chrom_sizes
        )
    co_markers = reduce(MarkerRecords.merge, co_markers)
    return co_markers
