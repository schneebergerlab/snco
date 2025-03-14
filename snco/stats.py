import logging
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d

from .utils import load_json
from .sim import ground_truth_from_marker_records


log = logging.getLogger('snco')


def total_markers(cb_co_markers):
    tot = 0
    for m in cb_co_markers.values():
        tot += m.sum(axis=None)
    return np.log10(tot)


def n_crossovers(cb_co_preds, min_co_prob=5e-3):
    nco = 0
    for p in cb_co_preds.values():
        p_co = np.abs(np.diff(p))
        p_co = np.where(p_co >= min_co_prob, p_co, 0)
        nco += p_co.sum(axis=None)
    return nco


def accuracy_score(cb_co_markers, cb_co_preds, max_score=10):
    nom = 0
    denom = 0
    for chrom, m in cb_co_markers.items():
        p = cb_co_preds[chrom]
        nom += (m[:, 0] * (1 - p)).sum() + (m[:, 1] * p).sum()
        denom += m.sum(axis=None)
    return np.minimum(-np.log2(1 - (nom / denom)), max_score)


def uncertainty_score(cb_co_preds):
    auc = 0
    for p in cb_co_preds.values():
        hu = np.abs(p - (p > 0.5))
        auc += np.trapz(hu).sum(axis=None)
    with np.errstate(divide='ignore'):
        return np.maximum(np.log10(auc), 0)


def coverage_score(cb_co_markers, max_score=10):
    cov = 0
    tot = 0
    for m in cb_co_markers.values():
        idx, = np.nonzero(m.sum(axis=1))
        try:
            cov += idx[-1] - idx[0] + 1
        except IndexError:
            cov += 0
        tot += len(m)
    return np.minimum(-np.log2(1 - (cov / tot)), max_score)


def mean_haplotype(cb_co_preds):
    return np.concatenate(list(cb_co_preds.values())).mean()


def geno_to_string(genotype):
    if genotype is None:
        return None
    else:
        return ':'.join(sorted(genotype))


def calculate_quality_metrics(co_markers, co_preds, nco_min_prob=2.5e-3, max_phred_score=10):
    qual_metrics = []
    genotypes = co_markers.metadata.get(
        'genotypes', defaultdict(
            lambda: {'genotype': None, 'genotype_probability': np.nan, 'genotyping_nmarkers': np.nan}
        )
    )
    bg_frac = co_markers.metadata.get(
        'estimated_background_fraction', defaultdict(lambda: np.nan)
    )
    doublet_rate = co_preds.metadata.get(
        'doublet_probability', defaultdict(lambda: np.nan)
    )
    for cb, cb_co_markers in co_markers.items():
        cb_co_preds = co_preds[cb]
        qual_metrics.append([
            cb,
            geno_to_string(genotypes[cb].get('genotype')),
            genotypes[cb].get('genotype_probability'),
            np.log10(genotypes[cb].get('genotyping_nmarkers')),
            total_markers(cb_co_markers),
            bg_frac.get(cb, np.nan),
            n_crossovers(cb_co_preds, min_co_prob=nco_min_prob),
            accuracy_score(cb_co_markers, cb_co_preds, max_score=max_phred_score),
            uncertainty_score(cb_co_preds),
            doublet_rate.get(cb, np.nan),
            coverage_score(cb_co_markers),
            mean_haplotype(cb_co_preds)
        ])
    qual_metrics = pd.DataFrame(
        qual_metrics,
        columns=['cb', 'geno_pred', 'geno_prob', 'geno_n_marker_reads',
                 'co_n_marker_reads', 'bg_fraction', 'n_crossovers',
                 'accuracy_score', 'uncertainty_score',
                 'doublet_probability',
                 'coverage_score', 'mean_haplotype']
    )
    return qual_metrics


def gt_haplotype_accuracy_score(cb_co_preds, cb_co_gt, thresholded=False, max_score=10):
    dev = 0
    nbins = 0
    for chrom, p in cb_co_preds.items():
        if thresholded:
            p = (p > 0.5).astype(np.float32)
        gt = cb_co_gt[chrom]
        dev += np.abs(p - gt).sum(axis=None)
        nbins += len(p)
    with np.errstate(divide='ignore'):
        return np.minimum(-np.log2(dev / nbins), max_score)


def _co_score(p, gt, ws=40):
    assert not ws % 2
    filt = np.ones(ws) / ws
    filt[: ws // 2] = np.negative(filt[: ws // 2])
    gt_c = convolve1d((gt - 0.5) * 2, filt, mode='nearest')
    p_c = convolve1d((p - 0.5) * 2, filt, mode='nearest')
    return np.trapz(gt_c * p_c)


def gt_co_score(cb_co_preds, cb_co_gt, window_size=40):
    n_co = n_crossovers(cb_co_gt)
    if not n_co:
        return np.nan
    co = 0
    for chrom, p in cb_co_preds.items():
        gt = cb_co_gt[chrom]
        co += _co_score(p, gt, window_size)
    return np.log10(np.maximum(co / n_co, 1))


def _max_detectable_cos(m, gt):
    co_idx = np.where(np.diff(gt))[0] + 1
    m_seg = np.array_split(m, co_idx, axis=0)
    seg_haps = gt[np.insert(co_idx, 0, 0)].astype(int)
    supported_haps = []
    for seg, h in zip(m_seg, seg_haps):
        support = seg[:, h].sum()
        if support:
            supported_haps.append(h)
    return len(np.where(np.diff(supported_haps))[0])


def gt_max_detectable_cos(cb_co_markers, cb_co_gt):
    dcos = 0
    for chrom, m in cb_co_markers.items():
        gt = cb_co_gt[chrom]
        dcos += _max_detectable_cos(m, gt)
    return dcos


def calculate_score_metrics(co_markers, co_preds, ground_truth, max_phred_score=10):
    score_metrics = []
    for cb, cb_co_preds in co_preds.items():
        cb_co_markers = co_markers[cb]
        if cb.split(':')[0] != 'doublet':
            cb_co_gt = ground_truth[cb]
            score_metrics.append([
                cb,
                int(n_crossovers(cb_co_gt)),
                gt_max_detectable_cos(cb_co_markers, cb_co_gt),
                gt_haplotype_accuracy_score(cb_co_preds, cb_co_gt, max_score=max_phred_score),
                gt_haplotype_accuracy_score(
                    cb_co_preds, cb_co_gt, thresholded=True, max_score=max_phred_score
                ),
                gt_co_score(cb_co_preds, cb_co_gt),
            ])
        else:
            score_metrics.append([
                cb, np.nan, np.nan, np.nan, np.nan, np.nan
            ])
    score_metrics = pd.DataFrame(
        score_metrics,
        columns=['cb', 'gt_n_crossovers', 'gt_detectable_cos',
                 'gt_accuracy_score', 'gt_thresholded_acc_score', 'gt_co_score']
    )
    return score_metrics


def write_metric_tsv(output_tsv_fn, qual_metrics, score_metrics=None, precision=3):
    if score_metrics is not None:
        qual_metrics = qual_metrics.merge(score_metrics, on='cb', how='outer')
    qual_metrics.to_csv(output_tsv_fn, sep='\t', index=False, float_format=f'%.{precision}g')


def run_stats(marker_json_fn, pred_json_fn, output_tsv_fn, *,
              co_markers=None, co_preds=None,
              cb_whitelist_fn=None, bin_size=25_000,
              nco_min_prob_change=2.5e-3, output_precision=3):
    '''
    Scores the quality of data and predictions for a set of haplotype calls
    generated with `predict`.
    '''
    if co_markers is None:
        co_markers = load_json(marker_json_fn, cb_whitelist_fn, bin_size)
    if co_preds is None:
        co_preds = load_json(
            pred_json_fn, cb_whitelist_fn, bin_size, data_type='predictions'
        )

    if set(co_preds.barcodes) != set(co_markers.barcodes):
        raise ValueError('Cell barcodes from marker-json-fn and predict-json-fn do not match')

    log.info('Calculating quality metrics')
    qual_metrics = calculate_quality_metrics(co_markers, co_preds)
    if 'ground_truth' in co_markers.metadata:
        ground_truth_haplotypes = ground_truth_from_marker_records(co_markers)
        log.info('Using ground truth info to calculate benchmarking metrics')
        score_metrics = calculate_score_metrics(co_markers, co_preds, ground_truth_haplotypes)
    else:
        score_metrics = None

    log.info(f'Writing stats to {output_tsv_fn}')
    write_metric_tsv(output_tsv_fn, qual_metrics, score_metrics, precision=output_precision)
