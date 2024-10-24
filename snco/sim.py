from collections import defaultdict

import numpy as np
import pandas as pd

from .records import MarkerRecords, PredictionRecords
from .signal import estimate_overall_background_signal, subtract_background


def co_invs_to_gt(co_invs, bin_size, chrom_nbins):
    '''
    Convert intervals from a bed file representing haplotypes into a binary numpy array
    '''
    gt = np.full(chrom_nbins, np.nan)

    start_bin = np.ceil(co_invs.start.values / bin_size)
    end_bin = np.ceil(co_invs.end.values / bin_size)
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


def apply_gt_to_markers(gt, m, bg_rate, bg_signal):
    '''
    For an array of markers from one chromosome of a cell barcode,
    estimate the foreground and background signal and then apply the ground truth
    haplotypes to create a simulated marker set for known crossovers
    '''
    gt = gt.astype(int)
    s = len(m)

    # simulate a realistic background signal using the average background across the dataset
    fg, bg = subtract_background(m, bg_signal, bg_rate, return_bg=True)

    # flatten haplotypes
    fg = fg.sum(axis=1)
    bg = bg.sum(axis=1)

    sim = np.zeros(shape=(s, 2))
    idx = np.arange(s)

    # apply fg at ground truth haplotype, and bg on other haplotype
    sim[idx, gt] = fg
    sim[idx, 1 - gt] = bg

    return sim


def generate_simulated_data(ground_truth, co_markers, conv_window_size=2_500_000,
                            bg_rate='auto', nsim_per_sample=100):
    '''
    For a set of crossover markers from real data, estimate the foreground 
    and background signal for each, and then apply the ground truth
    haplotypes to create a simulated marker set for known crossovers. 
    These can be used for benchmarking.
    '''
    bg, frac_bg = estimate_overall_background_signal(co_markers, conv_window_size)
    if bg_rate != 'auto':
        frac_bg = defaultdict(lambda: bg_rate)

    # todo: simulate some doublets

    sim_co_markers = MarkerRecords.new_like(co_markers)
    sim_co_markers.set_cb_whitelist(None)
    sim_co_markers.metadata['ground_truth'] = PredictionRecords.new_like(co_markers)
    sim_co_markers.metadata['ground_truth'].set_cb_whitelist(None)
    for sample_id in ground_truth.seen_barcodes:
        cbs_to_sim = np.random.choice(co_markers.seen_barcodes, replace=False, size=nsim_per_sample)
        for cb in cbs_to_sim:
            sim_id = f'{sample_id}:{cb}'
            for chrom in ground_truth.chrom_sizes:
                gt = ground_truth[sample_id, chrom]
                sim_co_markers[sim_id, chrom] = apply_gt_to_markers(
                    gt, co_markers[cb, chrom], frac_bg[cb], bg[chrom]
                )
                sim_co_markers.metadata['ground_truth'][sim_id, chrom] = gt
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
