import os
import logging
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d, binary_dilation
from scipy import stats

import torch
from pomegranate import distributions as pmd
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM

from .records import MarkerRecords
from .predict import RigidHMM, detect_crossovers, DEFAULT_RNG, DEFAULT_DEVICE
from .utils import load_json
from .stats import run_stats


log = logging.getLogger('snco')


def median_absolute_deviation(arr):
    return np.median(np.abs(arr - np.median(arr)))

def create_allele_ratio_mask(co_markers, expected_ratio='auto', nmad_mask=5, correction=1e-2):
    allele_ratio_mask = {}
    for chrom in co_markers.chrom_sizes:
        m = co_markers[..., chrom]
        m_norm = (m / m.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]).sum(axis=0)
        tot = m_norm.sum(axis=1)
        marker_mask = tot > 0
        allele_ratio = (m_norm[:, 0] + correction) / (tot + correction)
        if expected_ratio == 'auto':
            expected_ratio = np.median(allele_ratio[marker_mask])
        mad = median_absolute_deviation(allele_ratio[marker_mask])
        allele_ratio[tot == 0] = expected_ratio
        allele_ratio_mask[chrom] = binary_dilation(np.logical_or(
            allele_ratio < (expected_ratio - mad * nmad_mask),
            allele_ratio > (expected_ratio + mad * nmad_mask),
        ))
    return allele_ratio_mask


def apply_allele_ratio_mask(co_markers, expected_ratio='auto', nmad_mask=5, correction=1e-2):
    mask = create_allele_ratio_mask(co_markers, expected_ratio, nmad_mask, correction)
    co_markers_m = MarkerRecords.new_like(co_markers)
    for cb, chrom, m in co_markers.deep_items():
        co_markers_m[cb, chrom] = np.where(mask[chrom][:, np.newaxis], 0, m)
    return co_markers_m


def approximate_heterozygous_mask(m, ws=100, bc_haplotype=0):
    rs = convolve1d(m, np.ones(ws) / ws, axis=0, mode='constant', cval=0)
    return rs[:, bc_haplotype] < (2 * rs[:, 1 - bc_haplotype])


def calculate_parameters_bc(m, ws=100, bc_haplotype=0):
    het_mask = approximate_heterozygous_mask(m, ws)
    fg_lambda = m[het_mask, 1 - bc_haplotype].mean()
    bg_lambda = m[~het_mask, 1 - bc_haplotype].mean()
    empty_fraction = (m < bg_lambda).all(axis=1).mean()
    return fg_lambda, bg_lambda, empty_fraction


class BC1RigidHMM(RigidHMM):

    def _create_rigid_chain(self, params, haplotype):
        self._distributions[haplotype] = []
        for _ in range(self.rfactor):
            d = GeneralMixtureModel([pmd.Poisson(params), pmd.Poisson([self.bg_lambda, self.bg_lambda])],
                                    priors=[1 - self.empty_fraction, self.empty_fraction])
            self._model.add_distribution(d)
            self._distributions[haplotype].append(d)

    def initialise_model(self, fg_lambda, bg_lambda, empty_fraction):
        self.fg_lambda = fg_lambda
        self.bg_lambda = bg_lambda
        self.empty_fraction = empty_fraction
        self._model = DenseHMM(frozen=True)
        log.debug(f'moving model to device: {self._device}')
        self._model.to(self._device)
        for haplotype in self.haplotypes:
            params = [fg_lambda * 2, bg_lambda] if haplotype == 0 else [fg_lambda, fg_lambda]
            self._create_rigid_chain(params, haplotype)
        self._add_transitions()
        log.debug(
            f'Finished initialising model with {self._model.n_distributions} distributions'
        )

    def estimate_params(self, X):
        X_fg = []
        X_bg = []
        X_empty_fraction = []
        X = np.concatenate(X, axis=0)
        fg, bg, empty_fraction = calculate_parameters_bc(X, self.rfactor)
        return fg, bg, empty_fraction

    def fit(self, X):
        fg_lambda, bg_lambda, empty_fraction = self.estimate_params(X)
        log.debug(
            'Estimated model parameters from data: '
            f'fg_lambda {fg_lambda:.2g}, bg_lambda {bg_lambda:.2g}, empty_fraction {empty_fraction:.2f}'
        )
        self.initialise_model(fg_lambda, bg_lambda, empty_fraction)


def create_bc1rhmm(co_markers, cm_per_mb=4.5,
                segment_size=1_000_000, terminal_segment_size=50_000,
                model_lambdas=None, empty_fraction=0.1, device=DEFAULT_DEVICE):
    bin_size = co_markers.bin_size
    rfactor = segment_size // bin_size
    term_rfactor = terminal_segment_size // bin_size
    trans_prob = cm_per_mb * (bin_size / 1e8)
    rhmm = BC1RigidHMM(rfactor, term_rfactor, trans_prob, device)
    if model_lambdas is None:
        X = list(co_markers.deep_values())
        rhmm.fit(X)
    else:
        bg_lambda, fg_lambda = sorted(model_lambdas)
        rhmm.initialise_model(fg_lambda, bg_lambda, empty_fraction)
    return rhmm


def run_bc1predict(marker_json_fn, output_json_fn, *,
                   cb_whitelist_fn=None, bin_size=25_000,
                   segment_size=1_000_000, terminal_segment_size=50_000,
                   cm_per_mb=4.5, model_lambdas=None, empty_fraction=0.1,
                   generate_stats=True,
                   output_precision=3, processes=1,
                   batch_size=1_000, device=DEFAULT_DEVICE,
                   rng=DEFAULT_RNG):
    '''
    Uses rigid hidden Markov model to predict the haplotypes of each cell barcode
    at each genomic bin.
    '''
    co_markers = load_json(marker_json_fn, cb_whitelist_fn, bin_size)
    co_markers = apply_allele_ratio_mask(co_markers)
    rhmm = create_bc1rhmm(
        co_markers,
        cm_per_mb=cm_per_mb,
        segment_size=segment_size,
        terminal_segment_size=terminal_segment_size,
        model_lambdas=model_lambdas,
        empty_fraction=empty_fraction,
        device=device,
    )
    co_preds = detect_crossovers(
        co_markers, rhmm, batch_size=batch_size, processes=processes
    )

    if generate_stats:
        output_tsv_fn = f'{os.path.splitext(output_json_fn)[0]}.stats.tsv'
        run_stats(
            None, None, output_tsv_fn,
            co_markers=co_markers,
            co_preds=co_preds,
            output_precision=output_precision
        )

    log.info(f'Writing predictions to {output_json_fn}')
    co_preds.write_json(output_json_fn, output_precision)
