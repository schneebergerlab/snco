'''classes for reading bam files and aggregating count information'''
from collections import Counter
from functools import reduce
from dataclasses import dataclass, field
from operator import methodcaller

import numpy as np
import pysam

from .barcodes import umi_dedup_directional


DEFAULT_EXCLUDE_CONTIGS = set([
    'ChrM', 'ChrC', 'chrC', 'chrM', 'M', 'C', 'Mt', 'Pt',
    'Mitochondrial', 'mitochondrial', 'mito',
    'Chloroplast', 'chloroplast', 'plastid',
])


@dataclass
class IntervalMarkerCounts:

    '''
    dataclass to store deduplicated count information
    for a single bin/interval
    '''

    chrom: str
    bin_idx: int
    counts: dict = field(default_factory=dict)

    def __getitem__(self, index):
        if index not in self.counts:
            self.counts[index] = Counter()
        return self.counts[index]

    def __setitem__(self, index, val):
        self.counts[index] = Counter(val)

    def deep_items(self):
        for cb, hap_counts in self.counts.items():
            for hap, val in hap_counts.items():
                yield cb, hap, val

    @classmethod
    def new_like(cls, other):
        return cls(other.chrom, other.bin_idx)


class IntervalUMICounts:

    '''
    class to store count information for a single bin/interval
    prior to UMI deduplication
    '''

    def __init__(self, chrom, bin_idx, umi_collapse_method, hap_tag_type):
        self.chrom = chrom
        self.bin_idx = bin_idx
        self.umi_collapse_method = umi_collapse_method
        self.hap_tag_type = hap_tag_type
        self.has_umi = umi_collapse_method is not None
        self._counts = {}

    def __getitem__(self, index):
        if isinstance(index, str):
            if index not in self._counts:
                self._counts[index] = {}
            return self._counts[index]
        if len(index) == 2:
            cb, umi = index
            cb_counts = self[cb]
            if umi not in cb_counts:
                cb_counts[umi] = Counter()
            return cb_counts[umi]
        if len(index) == 3:
            cb, umi, hap = index
            umi_counts = self[cb, umi]
            return umi_counts[hap]
        raise KeyError(index)

    def __setitem__(self, index, val):
        if isinstance(index, tuple) and len(index) == 3:
            cb, umi, hap = index
            umi_counts = self[cb, umi]
            umi_counts[hap] = val
        else:
            raise NotImplementedError(f'Cannot set item with key {index}')

    def __iter__(self):
        return iter(self._counts)

    def hap_collapse_star_diploid(self, hap_counts):
        hap = list(hap_counts.keys())
        if len(hap) > 1:
            return None
        else:
            return hap[0]

    def hap_collapse_multi_haplotype(self, hap_counts):
        hap = reduce(frozenset.intersection, hap_counts.keys())
        if not len(hap):
            return None
        else:
            return hap

    def collapse(self):
        '''
        Deduplicate UMIs in the interval using one of three methods and
        return as an IntervalMarkerCounts object
        '''

        if self.umi_collapse_method is None:
            umi_eval_method = methodcaller('total')
        else:
            umi_eval_method = lambda hap_counts: 1

        if self.hap_tag_type == "star_diploid":
            hap_eval_method = self.hap_collapse_star_diploid
        else:
            hap_eval_method = self.hap_collapse_multi_haplotype

        def umi_eval(umi_hap_counts):
            total = Counter()
            for hap_counts in umi_hap_counts.values():
                hap = hap_eval_method(hap_counts)
                if hap is not None:
                    total[hap] += umi_eval_method(hap_counts)
            return total

        collapsed = IntervalMarkerCounts(self.chrom, self.bin_idx)
        for cb in self:
            if self.umi_collapse_method == 'directional':
                deduped = umi_dedup_directional(self[cb])
            else:
                deduped = self[cb]
            collapsed[cb] = umi_eval(deduped)
        return collapsed


def get_ha_samples(bam_fn):
    with pysam.AlignmentFile(bam_fn) as bam:
        try:
            return set(','.split(bam.header['HD']['ha']))
        except KeyError:
            for comment in bam.header['CO']:
                if comment.startswith('ha_flag_accessions'):
                    samples = set(comment.split(' ')[1].split(','))
                    break
            else:
                raise ValueError('Could not find ha_flag accession information in header')
    return sorted(samples)


class BAMHaplotypeIntervalReader:

    '''
    File wrapper class for reading aggregated bin counts for each cell barcode/haplotype
    '''

    def __init__(self, bam_fn, *,
                 bin_size=25_000,
                 cb_tag='CB',
                 umi_tag='UB',
                 hap_tag='ha',
                 hap_tag_type='star_diploid',
                 allowed_haplotypes=None,
                 cb_whitelist=None,
                 umi_collapse_method='directional',
                 exclude_contigs=None):
        self._bam_fn = bam_fn
        self.bin_size = bin_size
        self._cb_tag = cb_tag
        self._umi_tag = umi_tag
        self._hap_tag = hap_tag
        self.hap_tag_type = hap_tag_type
        if allowed_haplotypes is not None:
            allowed_haplotypes = frozenset(allowed_haplotypes)
        self.haplotypes = allowed_haplotypes
        self.cb_whitelist = cb_whitelist
        self.umi_collapse_method = umi_collapse_method
        if exclude_contigs is None:
            exclude_contigs = DEFAULT_EXCLUDE_CONTIGS
        self.exclude_contigs = exclude_contigs
        self._open()

    def _open(self):
        self.bam = pysam.AlignmentFile(self._bam_fn)
        self.is_closed = False

        self.chrom_sizes = {k: self.bam.get_reference_length(k)
                            for k in self.bam.references
                            if k not in self.exclude_contigs}
        self.nbins = {}
        for chrom, cs in self.chrom_sizes.items():
            self.nbins[chrom] = int(np.ceil(cs / self.bin_size))
        if self.hap_tag_type == 'multi_haplotype' and self.haplotypes is None:
            self.haplotypes = get_ha_samples(self._bam_fn)

    def fetch_interval_counts(self, chrom, bin_idx):
        '''
        for a specific bin index (positions determined by bin_size)
        read bam file and aggregate alignments into IntervalMarkerCounts
        object.
        '''
        assert bin_idx < self.nbins[chrom]
        bin_start = self.bin_size * bin_idx
        bin_end = bin_start + self.bin_size - 1

        interval_counts = IntervalUMICounts(chrom, bin_idx, self.umi_collapse_method, self.hap_tag_type)

        for aln in self.bam.fetch(chrom, bin_start, bin_end):

            # only keep primary, unduplicated alignments, as multimappers may cause artefacts
            if aln.is_secondary or aln.is_supplementary or aln.is_duplicate:
                continue

            # only consider fwd mapping of properly paired 2xreads, prevents double counting
            if aln.is_paired:
                if aln.is_reverse or not aln.is_proper_pair:
                    continue

            try:
                hap = aln.get_tag(self._hap_tag)
            except KeyError as exc:
                raise IOError(
                    f'bam records do not all have the haplotype tag "{self._hap_tag}"'
                ) from exc

            if self.hap_tag_type == 'star_diploid':
                hap -= 1
                if hap < 0:
                    continue
            elif self.hap_tag_type == 'multi_haplotype':
                hap = frozenset(str(hap).split(',')).intersection(self.haplotypes)
                if hap == self.haplotypes:
                    # read does not align better to one haplotype than others,
                    # so is not useful for haplotyping analysis
                    continue
                if not hap:
                    raise ValueError(
                        f'aln {aln.query_name} has no haplotypes that intersect with {self.haplotypes}'
                    )
            else:
                raise ValueError(f'hap_tag_type "{self.hap_tag_type}" not recognised')
            
            # only consider reads where the left mapping position is within the bin,
            # to prevent duplicates in adjacent bins
            if aln.reference_start < bin_start:
                continue

            try:
                cb = aln.get_tag(self._cb_tag)
                if cb == '-':
                    cb = None
            except KeyError as exc:
                raise IOError(
                    f'bam records do not all have the cell barcode tag "{self._cb_tag}"'
                ) from exc

            # finally filter for barcodes in the whitelist
            # use 1mm correction to match cbs with no more than 1 mismatch to the whitelist
            if self.cb_whitelist is not None:
                cb = self.cb_whitelist.correct(cb)

            if cb is not None:
                if self._umi_tag is not None:
                    try:
                        umi = aln.get_tag(self._umi_tag)
                    except KeyError as exc:
                        raise IOError(
                            f'bam records do not all have the UMI tag "{self._umi_tag}". '
                            'Maybe you meant to run without UMI deduplication?'
                        ) from exc
                else:
                    umi = None
                interval_counts[cb, umi, hap] += 1

        return interval_counts.collapse()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def close(self):
        self.bam.close()
