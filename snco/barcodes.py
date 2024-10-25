from copy import copy, deepcopy
from collections import defaultdict
import itertools as it

import numpy as np


def edit_dist(umi1, umi2):
    '''
    Calculate edit distance between two unique molecular identifiers
    '''
    return sum(i != j for i, j in zip(umi1, umi2))


class CellBarcodeWhitelist:

    def __init__(self, whitelist=None, correction_method='exact',
                 allow_ns=False, allow_homopolymers=False):
        if whitelist is not None:
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

    def check_barcode(self, cb):
        if not self.allow_ns and cb.count('N'):
            return False
        if not self.allow_homopolymers and len(set(cb)) == 1:
            return False
        return self.whitelist is None or cb in self.whitelist

    def __contains__(self, cb):
        return self.check_barcode(cb)

    def _find_new_match(self, cb):
        matches = set()
        for cb_w in self.whitelist:
            if edit_dist(cb, cb_w) == 1:
                matches.add(cb_w)
                if len(matches) > 1:
                    # barcode is within 1 edit of several barcodes, blacklist it
                    self._blacklist.add(cb)
                    return None
        else:
            if len(matches):
                assert len(matches) == 1
                cb_m = matches.pop()
                self._mapping[cb] = cb_m
                return cb_m
            else:
                return None

    def correct(self, cb):
        if cb in self:
            return cb
        elif cb in self._blacklist:
            return None
        elif self.correction_method == '1mm':
            try:
                return self._mapping[cb]
            except KeyError as e:
                cb_m = self._find_new_match(cb)
                return cb_m
        else:
            return None

    def __getitem__(self, idx):
        if not isinstance(idx, (int, np.integer, slice)):
            raise KeyError(f'CellBarcodeWhitelist indices must be integers or slices, not {type(idx)}')
        return self.whitelist_ordered[idx]

    def toset(self):
        return copy(self.whitelist)

    def tolist(self):
        return copy(self.whitelist_ordered)


def read_cb_whitelist(barcode_fn, cb_correction_method='exact'):
    '''
    Read a text file of cell barcodes and return them as a list.
    In a multi-column file, barcode must be the first column
    '''
    if barcode_fn is not None:
        with open(barcode_fn) as f:
            cb_whitelist = [cb.strip().split('\t')[0] for cb in f.readlines()]
    else:
        cb_whitelist = None
    return CellBarcodeWhitelist(cb_whitelist, cb_correction_method)


def umi_dedup_hap(umi_counts):
    '''
    Deduplicate UMIs using the directional method (UMItools)
    for a collection of UMIs aligning to the same gene/genomic bin,
    maintaining information about which haplotype each read supports
    
    Parameters
    ----------
    umi_counts : dict of dict of int
        Dictionary of UMI:hap:count information

    Returns
    -------
    dict of int
        Dictionary of deduplicated UMI:hap:count information
    '''
    edges = defaultdict(set)
    nodes = sorted(umi_counts, key=lambda k: sum(umi_counts[k].values()), reverse=True)
    for umi_i, umi_j in it.combinations(nodes, r=2):
        if edit_dist(umi_i, umi_j) <= 1:
            umi_i_count = sum(umi_counts[umi_i].values())
            umi_j_count = sum(umi_counts[umi_j].values())
            if umi_i_count >= (2 * umi_j_count + 1):
                edges[umi_i].add(umi_j)
    deduped_umi_counts = deepcopy(umi_counts)
    for parent in nodes:
        for child in edges[parent]:
            try:
                deduped_umi_counts[parent].update(deduped_umi_counts.pop(child))
            except KeyError:
                # umi already merged to a different parent
                continue
    return deduped_umi_counts
