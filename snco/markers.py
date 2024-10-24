import itertools as it
from functools import reduce
from joblib import Parallel, delayed

from .umi import umi_dedup_hap
from .records import MarkerRecords
from .bam import BAMHaplotypeIntervalReader, get_chrom_sizes_bam


def get_chrom_co_markers(bam_fn, chrom, **kwargs):
    '''
    For a single bam file/chrom combination, create a MarkerRecords object
    storing the haplotype marker information.
    '''
    with BAMHaplotypeIntervalReader(bam_fn, **kwargs) as bam:
        chrom_co_markers = MarkerRecords(
            bam.chrom_sizes,
            bam.bin_size,
            bam.cell_barcode_whitelist
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


def get_co_markers(bam_fns, processes=1, **kwargs):
    chrom_sizes = get_chrom_sizes_bam(bam_fns[0])
    with Parallel(n_jobs=processes, backend='loky') as pool:
        co_markers = pool(
            delayed(get_chrom_co_markers)(bam_fn, chrom, cb_suffix=cb_suffix, **kwargs)
            for (cb_suffix, bam_fn), chrom in it.product(enumerate(bam_fns, 1), chrom_sizes)
        )
    co_markers = reduce(MarkerRecords.merge, co_markers)
    return co_markers
