import numpy as np


def normalise_marker_counts(co_markers):
    total_counts = {cb: co_markers.total_marker_count(cb) for cb in co_markers.barcodes}
    norm_factor = np.median(list(total_counts.values()))
    co_markers = co_markers.copy()
    for cb in co_markers.barcodes:
        co_markers[cb] = {
            chrom: m / total_counts[cb] * norm_factor
            for chrom, m in co_markers[cb].items()
        }
    return co_markers
