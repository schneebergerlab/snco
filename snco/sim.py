import numpy as np
import pandas as pd


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
            if not len(chrom_invs):
                # no COs on this chromosome for this sample, randomly select a haplotype
                sample_gt[chrom] = np.repeat(np.random.randint(2), n)
            else:
                sample_gt[chrom] = co_invs_to_gt(chrom_invs, n)
        gt[sample_id] = sample_gt
    return gt


def apply_gt_to_markers(gt, m, bg_rate):
    gt = gt.astype(int)
    m = m.sum(1)
    s = len(m)
    idx = np.arange(s)
    nmarkers = sum(m)
    bg = np.bincount(
        np.random.choice(idx, replace=True, p=m / nmarkers, size=int(nmarkers * bg_rate)),
        minlength=s
    )
    bg = np.minimum(bg, m)
    fg = m - bg

    sim = np.zeros(shape=(s, 2))
    sim[idx, gt] = fg
    sim[idx, 1 - gt] = bg
    
    return sim


def generate_simulated_data(ground_truth, co_markers, bg_rate=0.05, nsim_per_sample=100):
    sim_data = {}
    for sample_id, gt in ground_truth.items():
        cbs_to_sim = np.random.choice(list(co_markers.keys()), replace=False, size=nsim_per_sample)
        for cb in cbs_to_sim:
            sim_id = f'{sample_id}:{cb}'
            m = co_markers[cb]
            s = {}
            for chrom, chrom_gt_haplo in gt.items():
                s[chrom] = apply_gt_to_markers(chrom_gt_haplo, m[chrom], bg_rate)
            sim_data[sim_id] = s
    return sim_data