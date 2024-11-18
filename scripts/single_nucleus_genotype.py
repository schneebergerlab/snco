import os
import random
from collections import defaultdict, Counter

import numpy as np
from scipy.io import mmread
import pandas as pd

import pysam
from joblib import Parallel, delayed
import click

from snco.utils import read_cb_whitelist
from snco.loadcsl import iter_cellsnp_lite_markers


def cellsnp_lite_to_genotype_markers(csl_dir,
                                     sample_vcf_fn,
                                     cb_whitelist,
                                     ignore_reference=True,
                                     max_count=20):
    genotype_markers = defaultdict(list)
    for snp in iter_cellsnp_lite_markers(csl_dir, cb_whitelist, sample_vcf_fn):
        if ignore_reference and snp.ref_count > snp.alt_count:
            continue
        genotype_markers[snp.cb].append((
            min(snp.ref_count, max_count),
            min(snp.alt_count, max_count),
            snp.ref_samples,
            snp.alt_samples
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


def assign_genotype_with_em(sample_markers, accessions, max_iter=1000, min_delta=1e-2):
    n_accs = len(accessions)
    probs = {acc: 1 / n_accs for acc in accessions}
    for _ in range(max_iter):
        prev_probs = probs
        probs = update_probs(probs, sample_markers)
        delta = sum(abs(prev_probs[g] - probs[g]) for g in accessions)
        if delta < min_delta:
            break
    return probs


def bootstrap_genotype_assignment(cb, cb_geno_markers,
                                  em_max_iter, em_min_delta,
                                  bootstrap_sample_size,
                                  n_bootstraps, rng):
    accessions = sorted(cb_geno_markers[0][2].union(cb_geno_markers[0][3]))
    prob_boots = defaultdict(list)
    N = len(cb_geno_markers)
    bootstrap_sample_size = min(N, bootstrap_sample_size)
    for _ in range(n_bootstraps):
        samp_idx = rng.integers(0, N, size=bootstrap_sample_size)
        samp = [cb_geno_markers[i] for i in samp_idx]
        acc_probs = assign_genotype_with_em(samp, accessions)
        for acc, p in acc_probs.items():
            prob_boots[acc].append(p)
    prob_means = {acc: np.mean(p) for acc, p in prob_boots.items()}
    geno = max(prob_means, key=prob_means.__getitem__)
    pm = prob_means[geno]
    return cb, geno, len(cb_geno_markers), -np.log10(1 - pm)


def parallel_assign_genotypes(genotype_markers,
                              em_max_iter, em_min_delta,
                              bootstrap_sample_size, n_bootstraps,
                              processes, rng):
    n_cb = len(genotype_markers)
    kwargs = dict(
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
@click.option('--em-max-iter', default=10_000, type=int)
@click.option('--em-min-delta', default=1e-2, type=float)
@click.option('--bootstrap-sample-size', default=1000, type=int)
@click.option('--n-bootstraps', default=25, type=int)
@click.option('-p', '--processes', default=1, type=int)
@click.option('-r', '--random-seed', default=101, type=int)
def main(cellsnplite_dir, cb_whitelist_fn, sample_vcf_fn, output_fn,
         em_max_iter, em_min_delta,
         bootstrap_sample_size, n_bootstraps,
         processes, random_seed):
    cb_whitelist = read_cb_whitelist(cb_whitelist_fn)
    genotype_markers = cellsnp_lite_to_genotype_markers(
        cellsnplite_dir, sample_vcf_fn, cb_whitelist
    )
    genotype_assignments = parallel_assign_genotypes(
        genotype_markers,
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