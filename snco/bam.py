from collections import defaultdict, Counter

import numpy as np
import pysam

ORGANELLAR_CONTIGS = set(['ChrM', 'ChrC'])


def get_chrom_sizes_bam(bam_fn, organellar_contigs=None):
    if organellar_contigs is None:
        organellar_contigs = ORGANELLAR_CONTIGS
    with pysam.AlignmentFile(bam_fn) as bam:
        chrom_sizes = {k: bam.get_reference_length(k)
                       for k in bam.references
                       if k not in organellar_contigs}
    return chrom_sizes


def read_cb_whitelist(barcode_fn):
    '''
    Read a text file of cell barcodes and return them as a list.
    In a multi-column file, barcode must be the first column
    '''
    with open(barcode_fn) as f:
        cb_whitelist = [cb.strip().split('\t')[0] for cb in f.readlines()]
    return cb_whitelist


class BAMHaplotypeIntervalReader:

    def __init__(self, bam_fn, *,
                 bin_size=25_000,
                 cb_suffix=None,
                 cb_tag='CB',
                 umi_tag='UB',
                 hap_tag='ha',
                 cell_barcode_whitelist=None,
                 organellar_contigs=None):
        self._bam_fn = bam_fn
        self.bin_size = bin_size
        self._cb_suffix = cb_suffix
        self._cb_tag = cb_tag
        self._umi_tag = umi_tag
        self._hap_tag = hap_tag
        self._has_cb_checker = cell_barcode_whitelist is not None
        if self._has_cb_checker and cb_suffix:
            cell_barcode_whitelist = [
                f'{cb}_{cb_suffix}' for cb in cell_barcode_whitelist
            ]
        self.cell_barcode_whitelist = set(cell_barcode_whitelist)
        if organellar_contigs is None:
            organellar_contigs = ORGANELLAR_CONTIGS
        self.organellar_contigs = organellar_contigs
        self._open()

    def _open(self):
        self.bam = pysam.AlignmentFile(self._bam_fn)
        self.is_closed = False

        self.chrom_sizes = {k: self.bam.get_reference_length(k)
                            for k in self.bam.references
                            if k not in self.organellar_contigs}
        self.nbins = {}
        for chrom, cs in self.chrom_sizes.items():
            self.nbins[chrom] = int(np.ceil(cs / self.bin_size))

    def _barcode_check(self, cb):
        if self._has_cb_checker:
            return cb in self.cell_barcode_whitelist
        return True

    def fetch_interval_counts(self, chrom, bin_idx):
        assert bin_idx < self.nbins[chrom]
        bin_start = self.bin_size * bin_idx
        bin_end = bin_start + self.bin_size - 1

        interval_counts = defaultdict(lambda: defaultdict(Counter))

        for aln in self.bam.fetch(chrom, bin_start, bin_end):

            # only keep primary, unduplicated alignments
            if aln.is_secondary or aln.is_supplementary or aln.is_duplicate:
                continue

            # only consider fwd mapping read of properly paired 2xreads
            if aln.is_paired:
                if aln.is_reverse or not aln.is_proper_pair:
                    continue

            # only keep alignments that unambiguously tag one of the haplotypes
            hap = aln.get_tag(self._hap_tag)
            if not hap:
                continue

            # only consider reads where the left mapping position is within the bin,
            # to prevent duplicates in adjacent bins
            if aln.reference_start < bin_start:
                continue

            # finally filter for barcodes in the whitelist
            cb = aln.get_tag(self._cb_tag)
            if self._cb_suffix is not None:
                cb = f'{cb}_{self._cb_suffix}'

            if self._barcode_check(cb):
                if self._umi_tag is not None:
                    umi = aln.get_tag(self._umi_tag)
                else:
                    umi = None
                interval_counts[cb][umi][hap] += 1

        return interval_counts

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def close(self):
        self.bam.close()
