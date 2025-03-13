from collections import Counter, defaultdict

import numpy as np

from joblib import Parallel, delayed

from .utils import spawn_child_rngs
from .bam import IntervalMarkerCounts
from .opts import DEFAULT_RANDOM_SEED

DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


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


def random_resample_geno_markers(cb_geno_markers, n_resamples, rng):
    genos = list(cb_geno_markers.keys())
    counts = np.array(list(cb_geno_markers.values()))
    tot = counts.sum()
    p = counts / tot
    for _ in range(n_resamples):
        idx = rng.choice(np.arange(len(p)), size=tot, replace=True, p=p)
        yield Counter(genos[i] for i in idx)


def assign_genotype_with_em(cb, cb_geno_markers, *, crossing_combinations,
                            max_iter=1000, min_delta=1e-3, n_bootstraps=25,
                            rng=DEFAULT_RNG):
    n_genos = len(crossing_combinations)
    init_probs = {geno: 1 / n_genos for geno in crossing_combinations}
    prob_bootstraps = defaultdict(list)
    for marker_sample in random_resample_geno_markers(cb_geno_markers, n_bootstraps, rng):
        probs = init_probs
        for _ in range(max_iter):
            prev_probs = probs
            probs = update_probs(probs, marker_sample)
            delta = sum(abs(prev_probs[g] - probs[g]) for g in crossing_combinations)
            if delta < min_delta:
                break
        for geno, p in probs.items():
            prob_bootstraps[geno].append(p)
    probs = {geno: np.mean(p) for geno, p in prob_bootstraps.items()}
    max_prob = max(probs.values())
    # when two or more genotypes are equally likely, select on at random
    genos_with_max_prob = [geno for geno, p in probs.items() if np.isclose(p, max_prob, atol=min_delta)]
    genos_with_max_prob.sort(key=lambda fznset: sorted(fznset))
    geno = rng.choice(genos_with_max_prob)
    geno_metadata = {
        'genotype': geno,
        'genotype_probability': max_prob,
        'all_probabilities': [(*sorted(g), p) for g, p in probs.items()],
        'genotyping_nmarkers': sum(cb_geno_markers.values())
    }
    return cb, geno_metadata


def parallel_assign_genotypes(genotype_markers, *, processes=1, rng=DEFAULT_RNG, **kwargs):
    n_cb = len(genotype_markers)
    # use sorted cb list to ensure deterministic results across runs
    barcodes = sorted(genotype_markers)
    res = Parallel(processes)(
        delayed(assign_genotype_with_em)(cb, genotype_markers[cb], rng=sp_rng, **kwargs)
        for cb, sp_rng in zip(barcodes, spawn_child_rngs(rng))
    )
    return dict(res)


def resolve_genotype_counts_to_co_markers(inv_counts, genotypes):
    resolved_inv_counts = []
    for ic in inv_counts:
        resolved_ic = IntervalMarkerCounts.new_like(ic)
        for cb, cb_haplo_ic in ic.counts.items():
            try:
                geno = genotypes[cb]['genotype']
            except KeyError:
                # cb was removed due to low markers
                continue
            geno_hap1, geno_hap2 = sorted(geno)
            for haps, count in cb_haplo_ic.items():
                if geno_hap1 in haps:
                    if geno_hap2 not in haps:
                        resolved_ic[cb][0] += count
                elif geno_hap2 in haps:
                    resolved_ic[cb][1] += count
        resolved_inv_counts.append(resolved_ic)
    return resolved_inv_counts


def genotype_from_inv_counts(inv_counts, min_markers_per_cb=100, **kwargs):
    genotype_markers = defaultdict(Counter)
    for ic in inv_counts:
        for cb, cb_counts in ic.counts.items():
            genotype_markers[cb] += cb_counts
    if min_markers_per_cb > 0:
        genotype_markers = {cb: g for cb, g in genotype_markers.items()
                            if sum(g.values()) >= min_markers_per_cb}
    genotypes = parallel_assign_genotypes(genotype_markers, **kwargs)
    inv_counts = resolve_genotype_counts_to_co_markers(inv_counts, genotypes)
    return genotypes, inv_counts