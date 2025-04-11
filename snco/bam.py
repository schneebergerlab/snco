'''classes for reading bam files and aggregating count information'''
import logging
from collections import Counter
from functools import reduce
from dataclasses import dataclass, field
from operator import methodcaller

import numpy as np
import pysam

from .barcodes import umi_dedup_directional

log = logging.getLogger('snco')

DEFAULT_EXCLUDE_CONTIGS = set([
    'ChrM', 'ChrC', 'chrC', 'chrM', 'M', 'C', 'Mt', 'Pt',
    'Mitochondrial', 'mitochondrial', 'mito',
    'Chloroplast', 'chloroplast', 'plastid',
])


@dataclass
class IntervalMarkerCounts:

    """
    Dataclass to store deduplicated count information for a single bin/interval.

    Attributes
    ----------
    chrom : str
        Chromosome name.
    bin_idx : int
        Index of the bin within the chromosome.
    counts : dict
        A dictionary storing counts for each cell barcode (CB) and haplotype (hap).

    Methods
    -------
    deep_items()
        Yields all items in counts (cell barcode, haplotype, value).
    new_like(other)
        Returns a new instance of IntervalMarkerCounts with the same chromosome and 
        bin index as `other`.
    """

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
        """
        Yields all items in counts (cell barcode, haplotype, value).

        Yields
        ------
        tuple
            (cell barcode, haplotype, count value).
        """
        for cb, hap_counts in self.counts.items():
            for hap, val in hap_counts.items():
                yield cb, hap, val

    @classmethod
    def new_like(cls, other):
        """
        Returns a new instance of IntervalMarkerCounts with the same chromosome 
        and bin index as `other`.

        Parameters
        ----------
        other : IntervalMarkerCounts
            The instance from which to copy chromosome and bin index.

        Returns
        -------
        IntervalMarkerCounts
            A new IntervalMarkerCounts instance with the same chrom and bin_idx as `other`.
        """
        return cls(other.chrom, other.bin_idx)


class IntervalUMICounts:

    """
    Class to store count information for a single bin/interval prior to UMI deduplication.

    Attributes
    ----------
    chrom : str
        Chromosome name.
    bin_idx : int
        Index of the bin within the chromosome.
    umi_collapse_method : str
        The UMI collapse method used ('directional', 'exact' or None).
    hap_tag_type : str
        The haplotype tag type used ('star_diploid', 'multi_haplotype').
    has_umi : bool
        Whether UMI collapse is applied.

    Methods
    -------
    collapse()
        Deduplicates UMIs and returns an IntervalMarkerCounts object with collapsed counts.
    """

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

    def _hap_collapse_star_diploid(self, hap_counts):
        """
        Collapses haplotype counts in star-diploid format.

        Parameters
        ----------
        hap_counts : dict
            A dictionary of haplotype counts.

        Returns
        -------
        str or None
            The collapsed haplotype, or None if it cannot be collapsed.
        """
        hap = list(hap_counts.keys())
        if len(hap) > 1:
            return None
        else:
            return hap[0]

    def _hap_collapse_multi_haplotype(self, hap_counts):
        """
        Collapses haplotype counts for multi-haplotype format.

        Parameters
        ----------
        hap_counts : dict
            A dictionary of haplotype counts.

        Returns
        -------
        frozenset or None
            The supported haplotypes as a frozenset, or None if it cannot be collapsed.
        """
        hap = reduce(frozenset.intersection, hap_counts.keys())
        if not len(hap):
            return None
        else:
            return hap

    def collapse(self):
        """
        Deduplicates UMIs and returns an IntervalMarkerCounts object with collapsed counts.

        Returns
        -------
        IntervalMarkerCounts
            A new IntervalMarkerCounts object containing the collapsed counts.
        """

        if self.umi_collapse_method is None:
            umi_eval_method = methodcaller('total')
        else:
            umi_eval_method = lambda hap_counts: 1

        if self.hap_tag_type == "star_diploid":
            hap_eval_method = self._hap_collapse_star_diploid
        else:
            hap_eval_method = self._hap_collapse_multi_haplotype

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
    """
    Retrieves haplotype samples from a BAM file.

    This function attempts to extract haplotype sample information from the header of the BAM file
    in multiple ways (using the 'ha' field in the HD header, or extracting from the 'ha_flag_accessions' 
    comment or RG fields).

    Parameters
    ----------
    bam_fn : str
        Path to the BAM file.

    Returns
    -------
    list of str
        A sorted list of haplotype sample identifiers.

    Raises
    ------
    ValueError
        If no haplotype information is found in the BAM file header.
    """
    with pysam.AlignmentFile(bam_fn) as bam:
        try:
            return set(bam.header['HD']['ha'].split(','))
        except KeyError:
            for comment in bam.header['CO']:
                if comment.startswith('ha_flag_accessions'):
                    samples = set(comment.split(' ')[1].split(','))
                    break
            else:
                # final attempt, to use RGs
                try:
                    samples = sorted(rg['ID'] for rg in bam.header['RG'])
                except KeyError:
                    raise ValueError('Could not find ha_flag accession information in header')
    return sorted(samples)


class BAMHaplotypeIntervalReader:
    """
    File wrapper class for reading aggregated bin counts for each cell barcode/haplotype.

    This class facilitates the reading of a BAM file, specifically extracting and aggregating
    count information for each interval (bin) within the genome for a given cell barcode (CB)
    and haplotype.

    Attributes
    ----------
    bin_size : int
        The size of each bin used for aggregation (default: 25,000).
    cb_tag : str
        The tag used to store the cell barcode (default: 'CB').
    umi_tag : str
        The tag used to store the UMI (default: 'UB').
    hap_tag : str
        The tag used to store the haplotype (default: 'ha').
    hap_tag_type : str
        The haplotype tag type ('star_diploid' or 'multi_haplotype').
    haplotypes : frozenset
        A set of allowed haplotypes.
    cb_whitelist : object
        An optional whitelist for cell barcodes.
    umi_collapse_method : str
        The method for collapsing UMIs ('directional', None, etc.).
    exclude_contigs : set
        Set of contig names to exclude (default: mitochondrial, plastid).

    Methods
    -------
    fetch_interval_counts(chrom, bin_idx)
        Aggregates alignments into an IntervalMarkerCounts object for the specified bin and chromosome.
    close()
        Closes the BAM file.
    """

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
        """
        File wrapper class for reading aggregated bin counts for each cell barcode/haplotype.
    
        This class facilitates the reading of a BAM file, specifically extracting and aggregating
        count information for each interval (bin) within the genome for a given cell barcode (CB)
        and haplotype.

        Parameters
        ----------
        bam_fn : str
            Path to the BAM file.
        bin_size : int, optional
            The size of each bin used for aggregation (default is 25,000).
        cb_tag : str, optional
            The tag used for the cell barcode (default is 'CB').
        umi_tag : str, optional
            The tag used for the UMI (default is 'UB').
        hap_tag : str, optional
            The tag used for the haplotype (default is 'ha').
        hap_tag_type : str, optional
            The haplotype tag type ('star_diploid' or 'multi_haplotype', default is 'star_diploid').
        allowed_haplotypes : list of str, optional
            List of allowed haplotypes. If None, the haplotypes are extracted from the BAM header.
        cb_whitelist : snco.barcodes.CellBarcodeWhitelist, optional
            A whitelist for cell barcodes, supports 1MM fuzzy matching.
        umi_collapse_method : str, optional
            The method for collapsing UMIs ('directional', 'exact', None).
        exclude_contigs : set, optional
            A set of contig names to exclude from processing (default is mitochondrial and plastid contigs).
        """
        self._bam_fn = bam_fn
        self.bin_size = bin_size
        self.cb_tag = cb_tag
        self.umi_tag = umi_tag
        self.hap_tag = hap_tag
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
        """
        Aggregates alignments into an IntervalMarkerCounts object for the specified bin and chromosome.

        This function fetches alignments from the BAM file for a given chromosome and bin index,
        and aggregates the alignments into an IntervalMarkerCounts object based on the cell barcode,
        UMI, and haplotype.

        Parameters
        ----------
        chrom : str
            The chromosome name.
        bin_idx : int
            The index of the bin within the chromosome.

        Returns
        -------
        IntervalMarkerCounts
            An object containing the aggregated counts for the specified bin.

        Raises
        ------
        IOError
            If the BAM records do not contain the expected tags (e.g., CB, UB, ha).
        ValueError
            If the haplotype information in the BAM file does not match the expected format.
        """
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
                hap = aln.get_tag(self.hap_tag)
            except KeyError as exc:
                raise IOError(
                    f'bam records do not all have the haplotype tag "{self.hap_tag}"'
                ) from exc

            if self.hap_tag_type == 'star_diploid':
                hap -= 1
                if hap < 0:
                    continue
            elif self.hap_tag_type == 'multi_haplotype':
                hap = frozenset(str(hap).split(','))
                if self.haplotypes is not None:
                    hap = hap.intersection(self.haplotypes)
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
                cb = aln.get_tag(self.cb_tag)
                if cb == '-':
                    cb = None
            except KeyError as exc:
                raise IOError(
                    f'bam records do not all have the cell barcode tag "{self.cb_tag}"'
                ) from exc

            # finally filter for barcodes in the whitelist
            # use 1mm correction to match cbs with no more than 1 mismatch to the whitelist
            if self.cb_whitelist is not None:
                cb = self.cb_whitelist.correct(cb)

            if cb is not None:
                if self.umi_tag is not None:
                    try:
                        umi = aln.get_tag(self.umi_tag)
                    except KeyError as exc:
                        raise IOError(
                            f'bam records do not all have the UMI tag "{self.umi_tag}". '
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
