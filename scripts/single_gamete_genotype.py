import os
import random
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
from scipy.io import mmread
import pandas as pd

import pysam
from joblib import Parallel, delayed
import click

from snco.utils import read_cb_whitelist
from snco.loadcsl import iter_cellsnp_lite_markers


def get_vcf_samples(sample_vcf_fn, include_reference, reference_name):
    with pysam.VariantFile(sample_vcf_fn) as vcf:
        samples = set(vcf.header.samples)
    if include_reference:
        samples.add(reference_name)
    return sorted(samples)


def cellsnp_lite_to_genotype_markers(csl_dir,
                                     sample_vcf_fn,
                                     cb_whitelist,
                                     crossing_combinations,
                                     include_reference=True,
                                     reference_name='col0',
                                     max_count=20):
    genotype_markers = defaultdict(list)
    for snp in iter_cellsnp_lite_markers(csl_dir, cb_whitelist, sample_vcf_fn):
        if include_reference:
            snp.ref_samples.add(reference_name)
        genotype_markers[snp.cb].append((
            min(snp.ref_count, max_count),
            min(snp.alt_count, max_count),
            [x for x in crossing_combinations if x.intersection(snp.ref_samples)],
            [x for x in crossing_combinations if x.intersection(snp.alt_samples)]
        ))
    return genotype_markers


def update_probs(probs, sample_markers):
    marker_agg = Counter()
    for ref_count, alt_count, ref_accs, alt_accs in sample_markers:
        n_ref = len(ref_accs)
        n_alt = len(alt_accs)
        for acc in ref_accs:
            marker_agg[acc] += (probs[acc] * (bool(ref_count))) / n_ref
        for acc in alt_accs:
            marker_agg[acc] += (probs[acc] * (bool(alt_count))) / n_alt

    agg_sum = sum(marker_agg.values())
    return {acc: marker_agg[acc] / agg_sum for acc in probs}


def assign_genotype_with_em(sample_markers, crossing_combinations, max_iter=1000, min_delta=1e-2):
    n_accs = len(crossing_combinations)
    probs = {acc: 1 / n_accs for acc in crossing_combinations}
    for _ in range(max_iter):
        prev_probs = probs
        probs = update_probs(probs, sample_markers)
        delta = sum(abs(prev_probs[g] - probs[g]) for g in crossing_combinations)
        if delta < min_delta:
            break
    return probs


def bootstrap_genotype_assignment(cb, cb_geno_markers, crossing_combinations,
                                  em_max_iter, em_min_delta,
                                  bootstrap_sample_size,
                                  n_bootstraps, rng):
    prob_boots = defaultdict(list)
    N = len(cb_geno_markers)
    bootstrap_sample_size = min(N, bootstrap_sample_size)
    for _ in range(n_bootstraps):
        samp_idx = rng.integers(0, N, size=bootstrap_sample_size)
        samp = [cb_geno_markers[i] for i in samp_idx]
        acc_probs = assign_genotype_with_em(samp, crossing_combinations)
        for acc, p in acc_probs.items():
            prob_boots[acc].append(p)
    prob_means = {acc: np.mean(p) for acc, p in prob_boots.items()}
    geno = max(prob_means, key=prob_means.__getitem__)
    pm = prob_means[geno]
    return cb, ':'.join(sorted(geno)), len(cb_geno_markers), -np.log10(1 - pm)


def parallel_assign_genotypes(genotype_markers, crossing_combinations,
                              em_max_iter, em_min_delta,
                              bootstrap_sample_size, n_bootstraps,
                              processes, rng):
    n_cb = len(genotype_markers)
    kwargs = dict(
        crossing_combinations=crossing_combinations,
        em_max_iter=em_max_iter,
        em_min_delta=em_min_delta,
        bootstrap_sample_size=bootstrap_sample_size,
        n_bootstraps=n_bootstraps,
        rng=rng
    )
    assignments = Parallel(processes)(
        delayed(bootstrap_genotype_assignment)(cb, cb_geno_markers, **kwargs)
        for cb, cb_geno_markers in genotype_markers.items()
    )
    return pd.DataFrame(
        assignments,
        columns=['cb', 'genotype', 'n_markers', 'nlog_prob']
    )


@click.command()
@click.argument('cellsnplite-dir', required=True)
@click.option('-c', '--cb-whitelist-fn', required=False)
@click.option('-v', '--sample-vcf-fn', required=True)
@click.option('-o', '--output-fn', required=True)
@click.option('-x', '--crosses', 'crossing_combinations',
              required=False, default=None,
              help='Expected combinations of genotypes from VCF in format "col0:tsu0;cvi0:kar1" etc. Default is all possible combinations')
@click.option('--include-reference/--omit-reference', required=False, default=True)
@click.option('-r', '--reference-name', required=False, default='col0')
@click.option('--em-max-iter', default=10_000, type=int)
@click.option('--em-min-delta', default=1e-2, type=float)
@click.option('--bootstrap-sample-size', default=1000, type=int)
@click.option('--n-bootstraps', default=25, type=int)
@click.option('-p', '--processes', default=1, type=int)
@click.option('-r', '--random-seed', default=101, type=int)
def main(cellsnplite_dir, cb_whitelist_fn, sample_vcf_fn, output_fn,
         crossing_combinations, include_reference, reference_name,
         em_max_iter, em_min_delta,
         bootstrap_sample_size, n_bootstraps,
         processes, random_seed):
    cb_whitelist = read_cb_whitelist(cb_whitelist_fn)
    if crossing_combinations is not None:
        crossing_combinations = list(set([frozenset(x.split(":")) for x in crossing_combinations.split(';')]))
    else:
        all_genotypes = get_vcf_samples(sample_vcf_fn, include_reference, reference_name)
        crossing_combinations = [frozenset(x) for x in combinations(all_genotypes, 2)]
    if len(crossing_combinations) < 2:
        raise ValueError('Insufficient crossing combinations to genotype (must be >1 unique, non-reciprocal crosses)')
    genotype_markers = cellsnp_lite_to_genotype_markers(
        cellsnplite_dir, sample_vcf_fn, cb_whitelist,
        crossing_combinations, include_reference, reference_name
    )
    genotype_assignments = parallel_assign_genotypes(
        genotype_markers, crossing_combinations,
        em_max_iter, em_min_delta,
        bootstrap_sample_size, n_bootstraps,
        processes,
        np.random.default_rng(random_seed)
    )
    genotype_assignments.to_csv(
        output_fn, sep='\t', index=False,
        float_format=f'%.3g'
    )

    
if __name__ == '__main__':
    main()