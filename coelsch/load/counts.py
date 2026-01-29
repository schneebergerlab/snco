from collections import Counter
from dataclasses import dataclass, field
from functools import reduce
from operator import methodcaller

from .barcodes import umi_dedup_directional


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

    def __iter__(self):
        return iter(self.counts)

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

    def deep_values(self):
        """
        Yields all values in counts.

        Yields
        ------
        int
            count value
        """
        for hap_counts in self.counts.values():
            yield from hap_counts.values()

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


@dataclass
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
    haplotype_validator : coelsch.load.loadbam.haplotypes.HaplotypeValidator or None
        class used to validate haplotypes of counts

    Methods
    -------
    collapse()
        Deduplicates UMIs and returns an IntervalMarkerCounts object with collapsed counts.
    """
    chrom: str
    bin_idx: int
    umi_collapse_method: str
    hap_tag_type: str
    haplotype_validator: None

    def __post_init__(self):
        self.has_umi = self.umi_collapse_method is not None
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
            return self.haplotype_validator[hap]

    def collapse(self):
        """
        Deduplicates UMIs and returns an IntervalMarkerCounts object with collapsed counts.

        Returns
        -------
        IntervalMarkerCounts
            A new IntervalMarkerCounts object containing the collapsed counts.
        """

        collapsed = IntervalMarkerCounts(self.chrom, self.bin_idx)
        if self.umi_collapse_method is None:
            for cb in self:
                # No UMIs, so counts are directly: self._counts[cb][None][hap]
                hap_counts = self._counts[cb].get(None, {})
                if hap_counts:
                    collapsed[cb] = hap_counts
            return collapsed

        if self.hap_tag_type == "star_diploid":
            hap_eval_method = self._hap_collapse_star_diploid
        else:
            hap_eval_method = self._hap_collapse_multi_haplotype

        def umi_eval(umi_hap_counts):
            total = Counter()
            for hap_counts in umi_hap_counts.values():
                hap = hap_eval_method(hap_counts)
                if hap is not None:
                    total[hap] += 1
            return total

        for cb in self:
            if self.umi_collapse_method == 'directional':
                deduped = umi_dedup_directional(self[cb])
            else:
                deduped = self[cb]
            final_count = umi_eval(deduped)
            if final_count:
                collapsed[cb] = final_count
        return collapsed

