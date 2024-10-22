import sys
import json
from collections import defaultdict, Counter
import itertools as it
import numpy as np
import pandas as pd
import pysam
from joblib import Parallel, delayed

from .umi import umi_dedup_hap
from .bam import BAMHaplotypeIntervalReader, get_chrom_sizes_bam


def get_chrom_co_markers(bam_fn, chrom, **kwargs):
    with BAMHaplotypeIntervalReader(bam_fn, **kwargs) as bam:
        nbins = bam.nbins[chrom]
        chrom_co_markers = defaultdict(lambda: np.zeros(shape=(nbins, 2), dtype=np.float64))
        for bin_idx in range(nbins):
            interval_counts = bam.fetch_interval_counts(chrom, bin_idx)
            
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
                    else:
                        hap = next(iter(hap_counts))
                        chrom_co_markers[cb][bin_idx, hap - 1] += 1
    return chrom_co_markers


def get_co_markers(bam_fns, processes=1, **kwargs):
    chrom_sizes = get_chrom_sizes_bam(bam_fns[0])
    with Parallel(n_jobs=processes, backend='loky', verbose=True) as pool:
        res = pool(
            delayed(get_chrom_co_markers)(bam_fn, chrom, **kwargs)
            for bam_fn, chrom in it.product(bam_fns, chrom_sizes)
        )
    co_markers = {}
    for (fn_idx, chrom), chrom_co_markers in zip(it.product(range(len(bam_fns)), chrom_sizes), res):
        for cb, cb_co_markers in chrom_co_markers.items():
            cb = f'{cb}_{fn_idx + 1}'
            if cb not in co_markers:
                co_markers[cb] = {}
            co_markers[cb][chrom] = cb_co_markers
    return co_markers, chrom_sizes


def co_markers_to_json(output_fn, co_markers, chrom_sizes, bin_size):
    co_markers_json_serialisable = {}
    marker_arr_sizes = {chrom: int(cs // bin_size + bool(cs % bin_size)) for chrom, cs, in chrom_sizes.items()}
    for cb, cb_co_markers in co_markers.items():
        d = {}
        for chrom, arr in cb_co_markers.items():
            idx = np.nonzero(arr.ravel())[0]
            val = arr.ravel()[idx]
            d[chrom] = (idx.tolist(), val.tolist())
        co_markers_json_serialisable[cb] = d
    with open(output_fn, 'w') as o:
        return json.dump({
            'cmd': ' '.join(sys.argv),
            'bin_size': bin_size,
            'chrom_sizes': chrom_sizes,
            'shape': marker_arr_sizes,
            'data': co_markers_json_serialisable
        }, fp=o)


def load_co_markers_from_json(co_marker_json_fn):
    with open(co_marker_json_fn) as f:
        co_marker_json = json.load(f)
    co_markers = {}
    bin_size = co_marker_json['bin_size']
    chrom_sizes = co_marker_json['chrom_sizes']
    arr_shapes = co_marker_json['shape']
    for cb, cb_marker_idx in co_marker_json['data'].items():
        cb_co_markers = {}
        for chrom, (idx, val) in cb_marker_idx.items():
            m = np.zeros(shape=arr_shapes[chrom] * 2, dtype=np.float64)
            m[idx] = val
            cb_co_markers[chrom] = m.reshape(-1, 2)
        co_markers[cb] = cb_co_markers
    return co_markers, chrom_sizes, bin_size

