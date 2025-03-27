import logging

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import convolve

from .utils import read_gtf_gene_locs, read_eqtl_results

log = logging.getLogger('snco')


def lod_peak_finder(lod_scores, pvals, lod_threshold, pval_threshold,
                    rel_lod_threshold, rel_prominence, ci_lod_drop,
                    min_dist):
    drop_threshold = max(max(lod_scores) - ci_lod_drop, 0)
    # insert infs at beginning and end of chromosome to prevent prominence edge effects
    lod_scores_padded = np.insert(lod_scores, [0, len(lod_scores)], [-np.inf, -np.inf])
    peak_idx, _ = find_peaks(
        lod_scores_padded,
        height=max(lod_threshold, max(lod_scores) * rel_lod_threshold),
        prominence=max(lod_scores) * rel_prominence,
        distance=min_dist
    )
    peaks = []
    for p_idx in peak_idx:
        if pvals[p_idx - 1] > pval_threshold:
            continue
        l_ci_idx = p_idx
        while (lod_scores_padded[l_ci_idx - 1] > drop_threshold):
            l_ci_idx -= 1
        r_ci_idx = p_idx
        while (lod_scores_padded[r_ci_idx + 1] > drop_threshold):
            r_ci_idx += 1
        # shift idx due to padding
        yield p_idx - 1, l_ci_idx - 1, r_ci_idx - 1



def call_eqtl_peaks_gene(gene_eqtl_results, lod_threshold=5, rel_lod_threshold=0.1,
                         pval_threshold=1e-2, rel_prominence=0.25, ci_lod_drop=1.5,
                         min_dist_between_eqtls=3e6, bin_size=25_000):
    min_dist = min_dist_between_eqtls // bin_size
    eqtl_peaks = []
    for _, chrom_eqtl_results in gene_eqtl_results.groupby('chrom'):
        chrom_eqtl_results = chrom_eqtl_results.sort_values('pos')
        lod_peaks = lod_peak_finder(
            chrom_eqtl_results.lod_score.values,
            chrom_eqtl_results.pval.values,
            lod_threshold=lod_threshold,
            pval_threshold=pval_threshold,
            rel_lod_threshold=rel_lod_threshold,
            rel_prominence=rel_prominence,
            ci_lod_drop=ci_lod_drop,
            min_dist=min_dist
        )
        for peak_idx, left_ci_idx, right_ci_idx in lod_peaks:
            e = chrom_eqtl_results.iloc[peak_idx].copy()
            e['left_ci_eqtl'] = chrom_eqtl_results.iloc[left_ci_idx].pos
            e['right_ci_eqtl'] = chrom_eqtl_results.iloc[right_ci_idx].pos + bin_size
            eqtl_peaks.append(e)
    eqtl_peaks = pd.DataFrame(eqtl_peaks)
    return eqtl_peaks


def call_eqtl_peaks(eqtl_results, gene_locs, cis_range=5e5, **kwargs):

    lod_threshold = kwargs.get('lod_threshold', 5)
    pval_threshold = kwargs.get('pval_threshold', 1e-2)
    def lod_filt(gene_eqtl_results):
        return (gene_eqtl_results.lod_score.max() > lod_threshold) & \
               (gene_eqtl_results.pval.min() < pval_threshold)

    eqtl_peaks = (eqtl_results.groupby('gene_id', as_index=False, sort=False)
                              .filter(lod_filt)
                              .groupby('gene_id', as_index=False, sort=False)
                              .apply(call_eqtl_peaks_gene, **kwargs)
                              .reset_index(drop=True))
    eqtl_peaks = pd.merge(
        eqtl_peaks,
        gene_locs,
        left_on='gene_id',
        right_index=True,
        how='left',
        suffixes=['_eqtl', '_gene']
    )
    eqtl_peaks['eqtl_type'] = eqtl_peaks.eval(
        'chrom_eqtl == chrom_gene & '
        '(left_ci_eqtl - @cis_range) <= pos_gene <= (right_ci_eqtl + @cis_range)'
    ).map({True: 'cis', False: 'trans'})
    eqtl_peaks = eqtl_peaks[[
        'chrom_eqtl', 'pos_eqtl', 'left_ci_eqtl', 'right_ci_eqtl',
        'gene_id', 'chrom_gene', 'pos_gene', 'eqtl_type',
        'lod_score', 'pval',
        *eqtl_results.columns.tolist()[5:]
    ]]
    return eqtl_peaks


def run_peakcall(eqtl_tsv_fn, output_tsv_fn, gtf_fn, *, eqtl_results=None,
                 lod_threshold=5, rel_lod_threshold=0.1,
                 pval_threshold=1e-2, rel_prominence=0.25, ci_lod_drop=1.5,
                 min_dist_between_eqtls=3e6, cis_eqtl_range=5e5):
    if eqtl_results is None:
        eqtl_results, bin_size = read_eqtl_results(eqtl_tsv_fn)
        log.info(f'Read eQTL results for {len(eqtl_results.gene_id.unique())} genes')
    gene_locs = read_gtf_gene_locs(gtf_fn)
    log.info('Calling eQTL peaks')
    eqtl_peaks = call_eqtl_peaks(
        eqtl_results, gene_locs,
        lod_threshold=lod_threshold,
        rel_lod_threshold=rel_lod_threshold,
        pval_threshold=pval_threshold,
        rel_prominence=rel_prominence,
        ci_lod_drop=ci_lod_drop,
        min_dist_between_eqtls=min_dist_between_eqtls,
        cis_range=cis_eqtl_range
    )
    if output_tsv_fn is not None:
        log.info(f'Writing peaks to {output_tsv_fn}')
        eqtl_peaks.to_csv(
            output_tsv_fn,
            sep='\t',
            index=False,
            header=True,
            float_format='%.4g'
        )
    return eqtl_peaks