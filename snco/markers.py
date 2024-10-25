from functools import reduce
from joblib import Parallel, delayed

from .barcodes import umi_dedup_hap
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
            interval_counts = bam.fetch_interval_counts(chrom, bin_idx)

            # todo: not all datasets need umi deduping, make optional
            interval_counts = {
                cb: umi_dedup_hap(cb_interval_counts)
                for cb, cb_interval_counts
                in interval_counts.items()
            }

            for cb, umi_hap_counts in interval_counts.items():
                for hap_counts in umi_hap_counts.values():
                    if len(hap_counts) != 1:
                        # UMI is ambiguous i.e. some reads map to hap1 and others to hap2
                        continue
                    hap = next(iter(hap_counts))
                    chrom_co_markers[cb, chrom, bin_idx, hap - 1] += 1
    return chrom_co_markers


def get_co_markers(bam_fn, processes=1, **kwargs):
    chrom_sizes = get_chrom_sizes_bam(bam_fn, exclude_contigs=kwargs.get('exclude_contigs', None))
    with Parallel(n_jobs=min(processes, len(chrom_sizes)), backend='loky') as pool:
        co_markers = pool(
            delayed(single_chrom_co_markers)(bam_fn, chrom, **kwargs)
            for chrom in chrom_sizes
        )
    co_markers = reduce(MarkerRecords.merge, co_markers)
    return co_markers
