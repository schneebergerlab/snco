import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from .utils import load_json
from .records import MarkerRecords, PredictionRecords
from .clean import estimate_overall_background_signal, subtract_background, random_bg_sample
from .opts import DEFAULT_RANDOM_SEED


log = logging.getLogger('snco')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def co_invs_to_gt(co_invs, bin_size, chrom_nbins):
    '''
    Convert intervals from a bed file representing haplotypes into a binary numpy array
    '''
    gt = np.full(chrom_nbins, np.nan)

    start_bin = np.ceil(co_invs.start.values / bin_size).astype(int)
    end_bin = np.ceil(co_invs.end.values / bin_size).astype(int)
    haplo = co_invs.haplo.values

    for s, e, h in zip(start_bin, end_bin, haplo):
        gt[s: e] = h
    if np.isnan(gt).any():
        raise ValueError('Supplied intervals in haplo-bed-fn do not completely cover chromosomes')
    return gt


def read_ground_truth_haplotypes(co_invs_fn, chrom_sizes, bin_size=25_000):
    '''
    Read a bed file containing haplotype intervals from a ground truth dataset
    and convert these into binned binary arrays, stored in PredictionRecords object
    '''
    co_invs = pd.read_csv(
        co_invs_fn,
        sep='\t',
        names=['chrom', 'start', 'end', 'sample_id', 'haplo', 'strand']
    )

    gt = PredictionRecords(chrom_sizes, bin_size, set(co_invs.sample_id))

    for sample_id, sample_invs in co_invs.groupby('sample_id'):
        for chrom, n in gt.nbins.items():
            chrom_invs = sample_invs.query('chrom == @chrom')
            gt[sample_id, chrom] = co_invs_to_gt(chrom_invs, bin_size, n)
    return gt


def apply_gt_to_markers(gt, m, bg_rate, bg_signal, rng=DEFAULT_RNG):
    '''
    For an array of markers from one chromosome of a cell barcode,
    estimate the foreground and background signal and then apply the ground truth
    haplotypes to create a simulated marker set for known crossovers
    '''
    gt = gt.astype(int)
    s = len(m)

    # simulate a realistic background signal using the average background across the dataset
    fg, bg = subtract_background(m, bg_signal, bg_rate, return_bg=True, rng=rng)

    # flatten haplotypes
    fg = fg.sum(axis=1)
    bg = bg.sum(axis=1)

    sim = np.zeros(shape=(s, 2))
    idx = np.arange(s)

    # apply fg at ground truth haplotype, and bg on other haplotype
    sim[idx, gt] = fg
    sim[idx, 1 - gt] = bg

    return sim


def simulate_singlets(co_markers, ground_truth, bg_signal, frac_bg, nsim_per_sample, rng=DEFAULT_RNG):
    '''
    Simulate barcodes with crossovers from ground truth
    '''
    sim_co_markers = MarkerRecords.new_like(co_markers)
    sim_co_markers.metadata['ground_truth'] = PredictionRecords.new_like(co_markers)
    for sample_id in ground_truth.barcodes:
        cbs_to_sim = rng.choice(co_markers.barcodes, replace=False, size=nsim_per_sample)
        for cb in cbs_to_sim:
            sim_id = f'{sample_id}:{cb}'
            for chrom in ground_truth.chrom_sizes:
                gt = ground_truth[sample_id, chrom]
                sim_co_markers[sim_id, chrom] = apply_gt_to_markers(
                    gt, co_markers[cb, chrom], frac_bg[cb], bg_signal[chrom], rng=rng
                )
                sim_co_markers.metadata['ground_truth'][sim_id, chrom] = gt
    return sim_co_markers


def simulate_doublets(co_markers, n_doublets, rng=DEFAULT_RNG):
    '''
    Simulate barcodes with doublets.
    '''
    sim_co_markers_doublets = MarkerRecords.new_like(co_markers)
    barcodes = rng.choice(co_markers.barcodes, size=n_doublets * 2, replace=False)
    # sorting by total markers makes m_i and m_j relatively similar in size
    barcodes = sorted(barcodes, key=co_markers.total_marker_count)
    for cb_i, cb_j in zip(barcodes[0::2], barcodes[1::2]):
        sim_id = f'doublet:{cb_i}_{cb_j}'
        for chrom in sim_co_markers_doublets.chrom_sizes:
            m_i = co_markers[cb_i, chrom]
            m_j = co_markers[cb_j, chrom]
            sim_co_markers_doublets[sim_id, chrom] = m_i + m_j
    return sim_co_markers_doublets


def generate_simulated_data(ground_truth, co_markers, conv_window_size=2_500_000,
                            bg_rate=None, nsim_per_sample=100,
                            doublet_rate=0.0, rng=DEFAULT_RNG):
    '''
    For a set of crossover markers from real data, estimate the foreground 
    and background signal for each, and then apply the ground truth
    haplotypes to create a simulated marker set for known crossovers. 
    These can be used for benchmarking.
    '''
    estimate_overall_background_signal(co_markers, conv_window_size, max_frac_bg=1.0)
    bg_signal = co_markers.metadata['background_signal']
    frac_bg = co_markers.metadata['estimated_background_fraction']
    if bg_rate is not None:
        frac_bg = defaultdict(lambda: bg_rate)

    sim_co_markers = simulate_singlets(
        co_markers, ground_truth, bg_signal, frac_bg, nsim_per_sample, rng=rng
    )

    if doublet_rate:
        doublet_rate = int(len(sim_co_markers) * doublet_rate) if doublet_rate < 1 else int(doublet_rate)
        sim_co_markers.merge(
            simulate_doublets(co_markers, doublet_rate, rng=rng),
            inplace=True
        )
    return sim_co_markers


def ground_truth_from_marker_records(co_markers):
    '''
    Extract ground truth information from a marker dataset into a new PredictionRecords object
    '''
    ground_truth = PredictionRecords.new_like(co_markers)
    for cb, sd in co_markers.metadata['ground_truth'].items():
        for chrom, arr in sd.items():
            ground_truth[cb, chrom] = np.array(arr)
    return ground_truth


def run_sim(marker_json_fn, output_json_fn, haplo_bed_fn, *,
            cb_whitelist_fn=None, bin_size=25_000, bg_marker_rate=None,
            bg_window_size=2_500_000, nsim_per_sample=100, n_doublets=0.0,
            rng=DEFAULT_RNG):
    '''
    Simulate realistic haplotype marker distributions using real data from `load`,
    with known haplotypes/crossovers supplied from a bed file.
    '''
    co_markers = load_json(marker_json_fn, cb_whitelist_fn, bin_size)
    ground_truth_haplotypes = read_ground_truth_haplotypes(
        haplo_bed_fn, co_markers.chrom_sizes, bin_size
    )
    log.info(f'Read {len(ground_truth_haplotypes)} ground truth samples from {haplo_bed_fn}')
    sim_co_markers = generate_simulated_data(
        ground_truth_haplotypes,
        co_markers,
        bg_rate=bg_marker_rate,
        conv_window_size=bg_window_size,
        nsim_per_sample=nsim_per_sample,
        doublet_rate=n_doublets,
        rng=rng
    )
    log.info(f'Simulated {len(sim_co_markers)} cells')
    if output_json_fn is not None:
        log.info(f'Writing markers to {output_json_fn}')
        sim_co_markers.write_json(output_json_fn)
    return sim_co_markers