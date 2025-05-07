from collections import Counter, defaultdict

import numpy as np

from joblib import Parallel, delayed

from .counts import IntervalMarkerCounts
from snco.utils import spawn_child_rngs
from snco.records import NestedData
from snco.defaults import DEFAULT_RANDOM_SEED

DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def update_probs(probs, sample_markers):
    """
    Update genotype probabilities based on observed marker counts.

    Parameters
    ----------
    probs : dict
        A dictionary of genotype probabilities for a barcode where keys are genotypes 
        (frozensets of two parental haplotypes) and values are the corresponding probabilities.
    sample_markers : dict
        A dictionary where keys are haplotype sets (frozensets) and values are the observed 
        counts of markers supporting that haplotype set, for a given cell barcode.

    Returns
    -------
    dict
        Updated genotype probabilities where keys are genotypes and values are the updated 
        probabilities.
    """
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
    """
    Randomly resample genotype markers with replacement.

    Parameters
    ----------
    cb_geno_markers : dict
        A dictionary where keys are haplotype sets (frozensets) and values are the observed 
        counts of markers supporting that haplotype set, for a given cell barcode.
    n_resamples : int
        The number of resamples to perform.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Yields
    ------
    Counter
        A resampled set of genotype markers with counts, sampled with replacement.
    """
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
    """
    Assign a genotype to a cell barcode using Expectation-Maximization (EM) and bootstrap re-sampling.

    Parameters
    ----------
    cb : str
        The cell barcode for which to assign a genotype.
    cb_geno_markers : dict
        A dictionary where keys are frozensets of haplotypes and values are the observed 
        counts of markers supporting that haplotype set, for a given cell barcode.
    crossing_combinations : list
        List of genotype combinations (frozensets of pairs of haplotypes) to consider for the EM algorithm.
    max_iter : int, optional
        The maximum number of iterations for the EM algorithm (default is 1000).
    min_delta : float, optional
        The minimum change in probabilities between iterations to stop the algorithm (default is 1e-3).
    n_bootstraps : int, optional
        The number of bootstrap samples to use for estimating genotype probabilities (default is 25).
    rng : np.random.Generator, optional
        Random number generator for reproducibility (default is `DEFAULT_RNG`).

    Returns
    -------
    cb : str
        The cell barcode.
    genotype : frozenset
        The assigned genotype.
    genotype_probability : float
        The probability of the assigned genotype.
    genotyping_nmarkers : int
        The number of markers used for genotyping.
    """
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
    return cb, geno, max_prob, sum(cb_geno_markers.values())


def parallel_assign_genotypes(genotype_markers, *, processes=1, rng=DEFAULT_RNG, **kwargs):
    """
    Assign genotypes to multiple cell barcodes in parallel using EM and bootstrap sampling.

    Parameters
    ----------
    genotype_markers : dict
        A dictionary where keys are cell barcodes and values are counters of haplotype markers 
        for each barcode.
    processes : int, optional
        The number of processes to use for parallel computation (default is 1).
    rng : np.random.Generator, optional
        Random number generator for reproducibility (default is `DEFAULT_RNG`).
    **kwargs : keyword arguments
        Additional parameters passed to the `assign_genotype_with_em` function.

    Returns
    -------
    geno_assignments : snco.records.NestedData
        The assigned genotype for each cell barcode
    geno_probabilities : snco.records.NestedData
        The probability that the assignment is correct, calculated using EM
    geno_nmarkers : snco.records.NestedData
        The number of markers that were used in genotype assignment
    """
    n_cb = len(genotype_markers)
    # use sorted cb list to ensure deterministic results across runs
    barcodes = sorted(genotype_markers)
    res = Parallel(processes)(
        delayed(assign_genotype_with_em)(cb, genotype_markers[cb], rng=sp_rng, **kwargs)
        for cb, sp_rng in zip(barcodes, spawn_child_rngs(rng))
    )
    geno_assignments = NestedData(levels=('cb',), dtype=frozenset)
    geno_probabilities = NestedData(levels=('cb',), dtype=float)
    geno_nmarkers = NestedData(levels=('cb',), dtype=int)

    for cb, geno, geno_prob, nmarkers in res:
        geno_assignments[cb] = geno
        geno_probabilities[cb] = float(geno_prob)
        geno_nmarkers[cb] = int(nmarkers)

    return geno_assignments, geno_probabilities, geno_nmarkers


def resolve_genotype_counts_to_co_markers(inv_counts, genotypes):
    """
    Resolve genotype counts for a set of barcodes into a format compatible with crossover calling

    Parameters
    ----------
    inv_counts : list of IntervalMarkerCounts
        A list of `IntervalMarkerCounts` objects, each representing marker counts identifying
        different haplotypes for a set of barcodes.
    genotypes : dict
        A dictionary where keys are cell barcodes and values are their assigned genotype metadata.

    Returns
    -------
    list of IntervalMarkerCounts
        A list of resolved `IntervalMarkerCounts` objects with markers uniquely identifying the two
        haplotypes of the assigned genotype for each cell barcode.
    """
    resolved_inv_counts = []
    for ic in inv_counts:
        resolved_ic = IntervalMarkerCounts.new_like(ic)
        for cb, cb_haplo_ic in ic.counts.items():
            try:
                geno = genotypes[cb]
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
    """
    Generate genotypes and crossover markers from interval counts using EM and bootstrap sampling.

    Parameters
    ----------
    inv_counts : list of IntervalMarkerCounts
        A list of `IntervalMarkerCounts` objects, each representing counts of markers
        across different cell barcodes.
    min_markers_per_cb : int, optional
        The minimum number of markers required for a barcode to be considered (default is 100).
    **kwargs : keyword arguments
        Additional parameters passed to the `parallel_assign_genotypes` and `assign_genotype_with_em`
        functions.

    Returns
    -------
    genotypes : dict
        A dictionary where keys are cell barcodes and values are their assigned genotype metadata.
    resolved_inv_counts : list of IntervalMarkerCounts
        A list of resolved `IntervalMarkerCounts` objects with markers differentiating the two
        haplotypes of the assigned genotype, to be used for recombination mapping.
    """
    genotype_markers = defaultdict(Counter)
    for ic in inv_counts:
        for cb, cb_counts in ic.counts.items():
            genotype_markers[cb] += cb_counts
    if min_markers_per_cb > 0:
        genotype_markers = {cb: g for cb, g in genotype_markers.items()
                            if sum(g.values()) >= min_markers_per_cb}
    genotypes, genotype_probs, genotype_nmarkers = parallel_assign_genotypes(genotype_markers, **kwargs)
    inv_counts = resolve_genotype_counts_to_co_markers(inv_counts, genotypes)
    # convert genotype data from frozenset to str
    genotypes= NestedData(
        levels=('cb',),
        dtype=str,
        data={cb: ':'.join(sorted(geno)) for cb, geno in genotypes.items()}
    )
    return genotypes, genotype_probs, genotype_nmarkers, inv_counts
