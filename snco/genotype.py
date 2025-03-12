from collections import Counter, defaultdict

from joblib import Parallel, delayed

from .bam import IntervalMarkerCounts


def update_probs(probs, sample_markers):
    marker_agg = Counter()
    for haps, marker_count in sample_markers.items():
        n_genos = 0
        genos_supported = []
        for geno in probs:
            if haps.intersection(geno):
                n_genos += 1
                genos_supported.append(geno)
        for geno in genos_supported:
            marker_agg[geno] += probs[geno] * marker_count / n_genos
    agg_sum = sum(marker_agg.values())
    if agg_sum == 0:
        return probs
    return {geno: marker_agg[geno] / agg_sum for geno in probs}


def assign_genotype_with_em(cb, cb_geno_markers, *, crossing_combinations, max_iter=1000, min_delta=1e-3):
    n_genos = len(crossing_combinations)
    probs = {geno: 1 / n_genos for geno in crossing_combinations}
    for _ in range(max_iter):
        prev_probs = probs
        probs = update_probs(probs, cb_geno_markers)
        delta = sum(abs(prev_probs[g] - probs[g]) for g in crossing_combinations)
        if delta < min_delta:
            break
    geno = max(probs, key=probs.__getitem__)
    geno_metadata = {
        'genotype': sorted(geno),
        'genotype_probability': probs[geno],
        'all_probabilities': [(*sorted(g), p) for g, p in probs.items()],
        'genotyping_nmarkers': sum(cb_geno_markers.values())
    }
    return cb, geno_metadata


def parallel_assign_genotypes(genotype_markers, *, processes=1, **kwargs):
    n_cb = len(genotype_markers)
    res = Parallel(processes)(
        delayed(assign_genotype_with_em)(cb, cb_geno_markers, **kwargs)
        for cb, cb_geno_markers in genotype_markers.items()
    )
    return {cb: geno_metadata for cb, geno_metadata in res}


def resolve_genotype_counts_to_co_markers(inv_counts, genotypes):
    resolved_inv_counts = []
    for ic in inv_counts:
        resolved_ic = IntervalMarkerCounts.new_like(ic)
        for cb, cb_haplo_ic in ic.counts.items():
            geno = genotypes[cb]['genotype']
            geno_hap1, geno_hap2 = sorted(geno)
            for haps, count in cb_haplo_ic.items():
                if geno_hap1 in haps:
                    if geno_hap2 not in haps:
                        resolved_ic[cb][0] += count
                elif geno_hap2 in haps:
                    resolved_ic[cb][1] += count
        resolved_inv_counts.append(resolved_ic)
    return resolved_inv_counts


def genotype_from_inv_counts(inv_counts, **kwargs):
    genotype_markers = defaultdict(Counter)
    for ic in inv_counts:
        for cb, cb_counts in ic.counts.items():
            genotype_markers[cb] += cb_counts
    genotypes = parallel_assign_genotypes(genotype_markers, **kwargs)
    inv_counts = resolve_genotype_counts_to_co_markers(inv_counts, genotypes)
    return genotypes, inv_counts
