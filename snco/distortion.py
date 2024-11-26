import itertools as it
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

LOD = 2 * np.log(10)

def segregation_distortion_chroms(chrom_haps, order, bin_size):
    res = []
    for positions in it.product(*(np.arange(c.shape[1]) for c in chrom_haps.values())):
        _, ct = stats.contingency.crosstab(*[
            (ch[:, p] > 0.5)
            for ch, p in zip(chrom_haps.values(), positions)
        ])
        if order == 1:
            exp = ct.sum() // 2
            chi2, pval, *_ = stats.chi2_contingency([ct, [exp, exp]], lambda_='log-likelihood')
        else:
            chi2, pval, *_ = stats.chi2_contingency(ct, lambda_='log-likelihood')
        res.append([*chrom_haps.keys(), *(p * bin_size for p in positions), chi2 / LOD, pval])
    res = pd.DataFrame(res, columns=[
        *(f'chrom_{i}' for i in range(1, order + 1)),
        *(f'pos_{i}' for i in range(1, order + 1)),
        'lod_score', 'pval'
    ])
    return res


def downsample_chrom(chrom_co_preds, bin_size, resolution):
    assert resolution >= bin_size and not resolution % bin_size
    cs = int(resolution / bin_size)
    splits = np.arange(cs, chrom_co_preds.shape[1], cs)
    return np.stack([m.mean(axis=1) for m in np.split(chrom_co_preds, splits, axis=1)], axis=1)


def segregation_distortion(co_preds, order=1, bin_size=25_000, resolution=2_500_000, processes=1):
    
    co_preds_low_res = {
        c: downsample_chrom(co_preds[..., c], co_preds.bin_size, resolution)
        for c in co_preds.chrom_sizes
    }
    with Parallel(n_jobs=processes) as pool:
        res = pool(
            delayed(segregation_distortion_chroms)(
                {c: co_preds_low_res[c] for c in chrom_perm}, order=order, bin_size=resolution
            ) for chrom_perm in it.combinations(co_preds.chrom_sizes, r=order)
        )
    res = pd.concat(res).reset_index(drop=True)
    res['pval'] = stats.false_discovery_control(res.pval)
    return res