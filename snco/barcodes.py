from copy import copy, deepcopy
from collections import defaultdict
from functools import reduce
from operator import add
import itertools as it

import numpy as np


def edit_dist(umi1, umi2):
    """
    Calculate the edit (Hamming) distance between two unique molecular identifiers (UMIs).

    Parameters
    ----------
    umi1 : str
        First UMI string (DNA sequence).
    umi2 : str
        Second UMI string (DNA sequence).

    Returns
    -------
    int
        Number of differing positions between the two UMIs.
    """
    return sum(i != j for i, j in zip(umi1, umi2))


class BarcodeValidator:
    """
    Validates DNA barcodes for format constraints such as homopolymers and ambiguous bases.

    Parameters
    ----------
    allow_ns : bool, default=False
        If True, allow barcodes to contain the base 'N'.
    allow_homopolymers : bool, default=False
        If True, allow barcodes composed of a single repeated base.
    """
    def __init__(self, allow_ns=False, allow_homopolymers=False):
        self.allow_ns = allow_ns
        self.allow_homopolymers = allow_homopolymers

    def is_valid(self, cb: str) -> bool:
        """
        Verify that a cell barcode meets criteria - no homopolymers or Ns

        Parameters
        ----------
        cb : str
            DNA barcode to validate.

        Returns
        -------
        bool
            True if valid barcode, else False
        """
        if cb is None:
            return False
        if not self.allow_ns and 'N' in cb:
            return False
        if not self.allow_homopolymers and len(set(cb)) == 1:
            return False
        return True

    def check_uniform_length(self, barcodes: list[str]):
        """
        Verify that all barcodes in the list are the same length.

        Parameters
        ----------
        barcodes : list of str
            List of DNA barcodes to validate.

        Raises
        ------
        ValueError
            If any barcode differs in length.
        """
        lengths = {len(cb) for cb in barcodes}
        if len(lengths) > 1:
            raise ValueError("All barcodes must be of equal length.")


class BarcodeCorrector:
    '''
    Correct DNA barcodes to a known whitelist using a 1-mismatch policy.

    Methods
    -------
    correct(cb: str) -> str | None
        Returns the corrected barcode, or None if no valid correction found.
    '''
    def __init__(self, whitelist: set[str]):
        """
        Correct DNA barcodes to a known whitelist using a 1-mismatch policy.

        Parameters
        ----------
        whitelist : set of str
            Valid barcodes for correction.
        """
        self.whitelist = set(whitelist)
        self._blacklist = set()
        self._cache = {}

    def correct(self, cb: str) -> str | None:
        """
        Correct a barcode against the whitelist using 1MM method.

        Parameters
        ----------
        cb : str
            Barcode to correct.

        Returns
        -------
        str or None
            Corrected barcode or None if uncorrectable.
        """
        if cb is None:
            return None
        if cb in self.whitelist:
            return cb
        if cb in self._blacklist:
            return None
        if cb in self._cache:
            return self._cache[cb]

        matches = set()
        for cb_w in self.whitelist:
            if edit_dist(cb, cb_w) == 1:
                matches.add(cb_w)
                if len(matches) > 1:
                    # barcode is within 1 edit of several whitelist barcodes, blacklist it
                    self._blacklist.add(cb)
                    return None
        if matches:
            assert len(matches) == 1
            cb_m = matches.pop()
            self._cache[cb] = cb_m
            return cb_m
        self._cache[cb] = None
        return None


class CellBarcodeWhitelist:
    """
    A barcode validator and corrector for cell barcodes used in single-cell sequencing.

    This class supports optional validation of barcodes for sequence constraints,
    and optional 1-mismatch correction against a provided whitelist of valid barcodes.

    Parameters
    ----------
    whitelist : list or set of str, optional
        A list of valid barcode sequences.
    validate_barcodes : bool, default=True
        Whether to enforce validation rules on barcodes (e.g., no 'N', no homopolymers).
    correction_method : {'exact', '1mm'}, default='exact'
        Barcode correction strategy. If '1mm', barcodes can be corrected to whitelist
        entries within an edit distance of 1, if unambiguous.
    allow_ns : bool, default=False
        Allow barcodes containing ambiguous base 'N'.
    allow_homopolymers : bool, default=False
        Allow barcodes composed of a single repeated nucleotide.

    Attributes
    ----------
    whitelist : set of str or None
        Set of valid barcodes, if provided.
    whitelist_ordered : list of str or None
        Original order of whitelist barcodes, if provided.
    validator : BarcodeValidator or None
        Internal barcode validator object. None if validation is disabled.
    corrector : BarcodeCorrector or None
        Internal corrector used for 1-mismatch correction, if enabled.

    Methods
    -------
    check_proper_barcode(cb: str) -> bool
        Return True if the given barcode is valid under the current constraints.
    correct(cb: str) -> str or None
        Validate and optionally correct a barcode.
    toset() -> set
        Return a set copy of the whitelist.
    tolist() -> list
        Return a list copy of the whitelist.
    """
    
    def __init__(
        self,
        whitelist: list[str] | None = None,
        validate_barcodes: bool = True,
        correction_method: str = "exact",
        allow_ns: bool = False,
        allow_homopolymers: bool = False,
    ):
        """
        A barcode whitelist for cell barcodes used in single-cell sequencing.

        Parameters
        ----------
        whitelist : iterable of str, optional
            A list or set of valid barcodes.
        validate_barcodes: bool, default: True
            Whether to check that barcodes are validate DNA barcodes
        correction_method : {'exact', '1mm'}, default: 'exact'
            Method for correcting barcodes. '1mm' enables correction of barcodes
            within edit distance 1 if unambiguous.
        allow_ns : bool, default: False
            Allow barcodes with 'N' bases.
        allow_homopolymers : bool, default: False
            Allow barcodes made of only one repeated base.
        """
        self.whitelist = set(whitelist) if whitelist else None
        self.whitelist_ordered = list(whitelist) if whitelist else None
        self.validator = (
            BarcodeValidator(allow_ns, allow_homopolymers) if validate_barcodes else None
        )

        if correction_method not in ("exact", "1mm"):
            raise ValueError(f"Unsupported correction method: {correction_method}")

        if whitelist:
            self.validator.check_uniform_length(self.whitelist)

        self.corrector = (
            BarcodeCorrector(self.whitelist) if correction_method == "1mm" else None
        )

    def check_proper_barcode(self, cb: str) -> bool:
        """
        Verify that barcode is valid (no Ns or homopolymers)

        Parameters
        ----------
        cb : str
            DNA barcode to validate.

        Returns
        -------
        bool
            True if valid barcode or validating is switched off, else False
        """
        return self.validator.is_valid(cb) if self.validator else True

    def correct(self, cb: str) -> str | None:
        """
        Validate and correct a barcode against the whitelist.

        Parameters
        ----------
        cb : str
            Barcode to validate and optionally correct.

        Returns
        -------
        str or None
            Corrected barcode, original barcode if valid, or None if invalid and uncorrectable.
        """
        if self.whitelist is None:
            return cb if self.check_proper_barcode(cb) else None
        if cb in self.whitelist:
            return cb
        if not self.check_proper_barcode(cb):
            return None
        return self.corrector.correct(cb) if self.corrector else None

    def __contains__(self, cb):
        if cb is None:
            return False
        return self.whitelist is None or cb in self.whitelist

    def __getitem__(self, idx):
        if not isinstance(idx, (int, np.integer, slice)):
            raise KeyError(
                f'CellBarcodeWhitelist indices must be integers or slices, not {type(idx)}'
            )
        return self.whitelist_ordered[idx]

    def toset(self):
        """
        Return the whitelist as a set.

        Returns
        -------
        set of str
            The set of whitelisted barcodes.
        """
        return copy(self.whitelist)

    def tolist(self):
        """
        Return the whitelist as a list.

        Returns
        -------
        list of str
            The ordered list of whitelisted barcodes.
        """
        return copy(self.whitelist_ordered)


def umi_dedup_directional(
    umi_hap_counts: dict[str, dict[str, int] | int],
    has_haplotype: bool = True
) -> dict:
    '''
    Deduplicate UMIs using the directional method (UMItools)
    for a collection of UMIs aligning to the same gene/genomic bin,
    maintaining information about which haplotype each read supports
    
    Parameters
    ----------
    umi_hap_counts : dict
        Dictionary of UMI -> haplotype -> count mappings if `has_haplotype` is True.
        Otherwise, a dictionary of UMI -> count.
    has_haplotype : bool, default: True
        Indicates whether haplotype information is present in the input.

    Returns
    -------
    dict
        Dictionary of deduplicated UMI counts. If haplotypes were used, each UMI maps
        to a Counter of haplotype counts.
    '''
    edges = defaultdict(set)
    if has_haplotype:
        umi_counts = {umi: hap_counts.total() for umi, hap_counts in umi_hap_counts.items()}
    else:
        umi_counts = umi_hap_counts
    nodes = sorted(umi_counts, key=umi_counts.__getitem__, reverse=True)
    for umi_i, umi_j in it.combinations(nodes, r=2):
        if edit_dist(umi_i, umi_j) <= 1:
            umi_i_count = umi_counts[umi_i]
            umi_j_count = umi_counts[umi_j]
            if umi_i_count >= (2 * umi_j_count + 1):
                edges[umi_i].add(umi_j)
    deduped = deepcopy(umi_hap_counts)
    for parent in nodes:
        for child in edges[parent]:
            try:
                deduped[parent] += deduped.pop(child)
            except KeyError:
                # umi already merged to a different parent
                continue
    return deduped