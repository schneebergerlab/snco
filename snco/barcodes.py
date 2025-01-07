from copy import copy, deepcopy
from collections import defaultdict
from functools import reduce
from operator import add
import itertools as it

import numpy as np


def edit_dist(umi1, umi2):
    '''
    Calculate edit distance between two unique molecular identifiers
    '''
    return sum(i != j for i, j in zip(umi1, umi2))


class CellBarcodeWhitelist:

    def __init__(self, whitelist=None, validate_barcodes=True,
                 correction_method='exact',
                 allow_ns=False, allow_homopolymers=False):
        if whitelist is not None:
            if validate_barcodes:
                self._check_whitelist(whitelist, allow_ns, allow_homopolymers)
            self.whitelist = set(whitelist)
            self.whitelist_ordered = list(whitelist)
        else:
            self.whitelist = None
            self.whitelist_ordered = None
        if correction_method not in ('exact', '1mm'):
            raise ValueError(f'Unrecognised cb correction method {correction_method}')
        if correction_method == '1mm' and whitelist is None:
            raise ValueError('cb-whitelist must be supplied if correction-method is "1mm"')
        self.validate = validate_barcodes
        self.correction_method = correction_method
        self.allow_ns = allow_ns
        self.allow_homopolymers = allow_homopolymers
        self._blacklist = set()
        self._mapping = {}

    def _check_whitelist(self, whitelist, allow_ns, allow_homopolymers):
        cb_len = []
        for cb in whitelist:
            if not allow_ns and cb.count('N'):
                raise ValueError('Cell barcode whitelist contains barcodes with Ns')
            if not allow_homopolymers and len(set(cb)) == 1:
                raise ValueError('Cell barcode whitelist contains homopolymers')
            cb_len.append(len(cb))
        if not all(ln == cb_len[0] for ln in cb_len):
            raise ValueError('Cell barcodes are not all the same length')

    def check_proper_barcode(self, cb):
        if cb is None:
            return False
        if not self.validate:
            return True
        if not self.allow_ns and cb.count('N'):
            return False
        if not self.allow_homopolymers and len(set(cb)) == 1:
            return False
        return True

    def __contains__(self, cb):
        if cb is None:
            return False
        return self.whitelist is None or cb in self.whitelist

    def _find_new_match(self, cb):
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
            self._mapping[cb] = cb_m
            return cb_m
        return None

    def correct(self, cb):
        if cb in self:
            return cb
        if not self.check_proper_barcode(cb):
            return None
        if cb in self._blacklist:
            return None
        if self.correction_method == '1mm':
            try:
                return self._mapping[cb]
            except KeyError:
                cb_m = self._find_new_match(cb)
                return cb_m
        return None

    def __getitem__(self, idx):
        if not isinstance(idx, (int, np.integer, slice)):
            raise KeyError(
                f'CellBarcodeWhitelist indices must be integers or slices, not {type(idx)}'
            )
        return self.whitelist_ordered[idx]

    def toset(self):
        return copy(self.whitelist)

    def tolist(self):
        return copy(self.whitelist_ordered)


def umi_dedup_directional(hap_umi_counts):
    '''
    Deduplicate UMIs using the directional method (UMItools)
    for a collection of UMIs aligning to the same gene/genomic bin,
    maintaining information about which haplotype each read supports
    
    Parameters
    ----------
    umi_counts : dict of Counter
        Dictionary of hap:UMI:count information

    Returns
    -------
    dict of int
        Dictionary of deduplicated UMI:hap:count information
    '''
    edges = defaultdict(set)
    umi_counts = reduce(add, hap_umi_counts.values())
    nodes = sorted(umi_counts, key=umi_counts.__getitem__, reverse=True)
    for umi_i, umi_j in it.combinations(nodes, r=2):
        if edit_dist(umi_i, umi_j) <= 1:
            umi_i_count = umi_counts[umi_i]
            umi_j_count = umi_counts[umi_j]
            if umi_i_count >= (2 * umi_j_count + 1):
                edges[umi_i].add(umi_j)
    deduped = deepcopy(hap_umi_counts)
    for umi_counts in deduped.values():
        for parent in nodes:
            for child in edges[parent]:
                try:
                    umi_counts[parent] += umi_counts.pop(child)
                except KeyError:
                    # umi already merged to a different parent
                    continue
    return deduped
