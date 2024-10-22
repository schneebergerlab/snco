from collections import defaultdict

import numpy as np
import pandas as pd

from .signal import estimate_overall_background_signal, subtract_background


def co_invs_to_gt(co_invs, chrom_nbins):
    gt = np.empty(shape=chrom_nbins)
    for _, inv in co_invs.iterrows():
        gt[inv.start_bin: inv.end_bin] = inv.haplo
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

    gt = {}
    
    for sample_id, sample_invs in co_invs.groupby('sample_id'):
        sample_gt = {}
        for chrom, n in nbins.items():
            chrom_invs = sample_invs.query('chrom == @chrom')
            sample_gt[chrom] = co_invs_to_gt(chrom_invs, n)
        gt[sample_id] = sample_gt
    return gt


def apply_gt_to_markers(gt, m, bg_rate, bg_signal):
    gt = gt.astype(int)
    s = len(m)
    nmarkers = m.sum(axis=None)
    nbg = int(round(nmarkers * bg_rate))

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


def generate_simulated_data(ground_truth, co_markers,
                            bin_size=25_000, conv_window_size=2_500_000,
                            bg_rate='auto', nsim_per_sample=100):

    bg, frac_bg = estimate_overall_background_signal(co_markers, bin_size, conv_window_size)
    if bg_rate != 'auto':
        frac_bg = defaultdict(lambda: bg_rate)
    
    sim_data = {}
    for sample_id, gt in ground_truth.items():
        cbs_to_sim = np.random.choice(list(co_markers.keys()), replace=False, size=nsim_per_sample)
        for cb in cbs_to_sim:
            sim_id = f'{sample_id}:{cb}'
            m = co_markers[cb]
            s = {}
            for chrom, chrom_gt_haplo in gt.items():
                s[chrom] = apply_gt_to_markers(chrom_gt_haplo, m[chrom], frac_bg[cb], bg[chrom])
            sim_data[sim_id] = s
    return sim_data