from collections import defaultdict

import numpy as np
import pandas as pd

from .records import MarkerRecords, PredictionRecords
from .signal import estimate_overall_background_signal, subtract_background


def co_invs_to_gt(co_invs, chrom_nbins):
    gt = np.full(chrom_nbins, np.nan)
    for _, inv in co_invs.iterrows():
        gt[inv.start_bin: inv.end_bin] = inv.haplo
    if np.isnan(gt).any():
        raise ValueError('Supplied intervals in haplo-bed-fn do not completely cover chromosomes')
    return gt


def read_ground_truth_haplotypes(co_invs_fn, chrom_sizes, bin_size=25_000):

    co_invs = pd.read_csv(
        co_invs_fn,
        sep='\t',
        names=['chrom', 'start', 'end', 'sample_id', 'haplo', 'strand']
    )
    co_invs['start_bin'] = co_invs.start // bin_size + (co_invs.start % bin_size).astype(bool)
    co_invs['end_bin'] = co_invs.end // bin_size + (co_invs.end % bin_size).astype(bool)

    nbins = {}
    for chrom, cs in chrom_sizes.items():
        nbins[chrom] = int(cs // bin_size + bool(cs % bin_size))

    gt = PredictionRecords(chrom_sizes, bin_size, set(co_invs.sample_id))

    for sample_id, sample_invs in co_invs.groupby('sample_id'):
        for chrom, n in nbins.items():
            chrom_invs = sample_invs.query('chrom == @chrom')
            gt[sample_id, chrom] = co_invs_to_gt(chrom_invs, n)
    return gt


def apply_gt_to_markers(gt, m, bg_rate, bg_signal):
    gt = gt.astype(int)
    s = len(m)

    # simulate a realistic background signal using the average background across the dataset
    fg, bg = subtract_background(m, bg_signal, bg_rate, return_bg=True)

    # flatten haplotypes
    fg = fg.sum(axis=1)
    bg = bg.sum(axis=1)

    sim = np.zeros(shape=(s, 2))
    idx = np.arange(s)
    sim[idx, gt] = fg
    sim[idx, 1 - gt] = bg

    return sim


def generate_simulated_data(ground_truth, co_markers, conv_window_size=2_500_000,
                            bg_rate='auto', nsim_per_sample=100):

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
    ground_truth = PredictionRecords.new_like(co_markers)
    for cb, sd in co_markers.metadata['ground_truth'].items():
        for chrom, arr in sd.items():
            ground_truth[cb, chrom] = np.array(arr)
    return ground_truth
