import numpy as np


def normalise_marker_counts(co_markers, normalise_haplotypes=True):

    # across the whole dataset, hap1 and hap2 should be approximately evenly covered
    if normalise_haplotypes:
        hap_counts = np.sum(np.concatenate(list(co_markers.deep_values())), axis=0)
        hap_norm = hap_counts.sum() / hap_counts / len(hap_counts)
        for m in co_markers.deep_values():
            m *= hap_norm

    total_counts = {cb: co_markers.total_marker_count(cb) for cb in co_markers.barcodes}
    norm_factor = np.median(list(total_counts.values()))
    co_markers = co_markers.copy()
    for cb in co_markers.barcodes:
        co_markers[cb] = {
            chrom: m / total_counts[cb] * norm_factor
            for chrom, m in co_markers[cb].items()
        }
    return co_markers
