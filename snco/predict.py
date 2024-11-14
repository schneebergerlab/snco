import os
import logging
from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

import torch
from pomegranate import distributions as pmd
from pomegranate.hmm import DenseHMM

from .logger import progress_bar
from .sim import simulate_doublets
from .utils import load_json
from .records import PredictionRecords
from .clean import predict_foreground_convolution
from . import stats
from .opts import DEFAULT_RANDOM_SEED


log = logging.getLogger('snco')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
DEFAULT_DEVICE = torch.device('cpu')


class RigidHMM:

    haplotypes = (0, 1)
    _Transitions = namedtuple(
        'Transitions',
        ['self_loop', 'rigid', 'co', 'start', 'end'],
        defaults=[0.0, 0.0, 0.0, 0.0, 0.0]
    )

    def __init__(self, rfactor, term_rfactor, trans_prob, device=DEFAULT_DEVICE):
        self.rfactor = rfactor
        self.term_rfactor = term_rfactor
        self.trans_prob = trans_prob
        self.term_prob = 1 / (self.rfactor - self.term_rfactor)
        assert 0.0 < self.trans_prob < 1.0
        assert (self.trans_prob + self.term_prob) < 1.0
        self.fg_lambda = None
        self.bg_lambda = None
        self._transition_probs = {}
        self._calculate_transition_probs()
        self._model = None
        self._distributions = {}
        self._hap2_states = np.arange(rfactor) + rfactor
        self._device = device

    def _calculate_transition_probs(self):
        for i in range(self.rfactor):
            # first state of chain
            if i == 0:
                self._transition_probs[i] = self._Transitions(
                    self_loop=1.0 - self.trans_prob - self.term_prob,
                    rigid=self.trans_prob,
                    start=self.term_prob,
                )
            # last state of chain
            elif (i + 1) == self.rfactor:
                self._transition_probs[i] = self._Transitions(
                    self_loop=1.0 - self.trans_prob - self.term_prob,
                    co=self.trans_prob,
                    end=self.term_prob
                )
            # internal point of the chain
            else:
                start_prob = self.term_prob if i <= (self.rfactor - self.term_rfactor) else 0.0
                end_prob = self.term_prob if (i + 1) >= self.term_rfactor else 0.0
                self._transition_probs[i] = self._Transitions(
                    self_loop=0.0,
                    rigid=1.0 - end_prob,
                    start=start_prob,
                    end=end_prob
                )

    def _create_rigid_chain(self, params, haplotype):
        self._distributions[haplotype] = []
        for _ in range(self.rfactor):
            d = pmd.Poisson(params)
            self._model.add_distribution(d)
            self._distributions[haplotype].append(d)

    def _add_transitions(self):

        # manually set all edges to -inf first, then update
        # this prevents a bug where edges in large models are occasionally
        # initialised with NaNs by pomegranate
        # see: https://github.com/jmschrei/pomegranate/issues/1078

        n = self._model.n_distributions
        self._model.starts = torch.full(
            (n,), -np.inf, dtype=self._model.dtype, device=self._device
        )
        self._model.ends = torch.full(
            (n,), -np.inf, dtype=self._model.dtype, device=self._device
        )
        self._model.edges = torch.full(
            (n, n), -np.inf, dtype=self._model.dtype, device=self._device
        )

        for hap in self.haplotypes:
            for i in range(self.rfactor):
                if self._transition_probs[i].self_loop:
                    self._model.add_edge(
                        self._distributions[hap][i],
                        self._distributions[hap][i],
                        self._transition_probs[i].self_loop
                    )
                if self._transition_probs[i].rigid:
                    self._model.add_edge(
                        self._distributions[hap][i],
                        self._distributions[hap][i + 1],
                        self._transition_probs[i].rigid
                    )
                if self._transition_probs[i].co:
                    self._model.add_edge(
                        self._distributions[hap][i],
                        self._distributions[1 - hap][0],
                        self._transition_probs[i].co
                    )
                if self._transition_probs[i].start:
                    self._model.add_edge(
                        self._model.start,
                        self._distributions[hap][i],
                        self._transition_probs[i].start
                    )
                if self._transition_probs[i].end:
                    self._model.add_edge(
                        self._distributions[hap][i],
                        self._model.end,
                        self._transition_probs[i].end
                    )

    def initialise_model(self, fg_lambda, bg_lambda):
        # todo: implement sparse version for when rfactor is very large
        self.fg_lambda = fg_lambda
        self.bg_lambda = bg_lambda
        self._model = DenseHMM(frozen=True)
        log.debug(f'moving model to device: {self._device}')
        self._model.to(self._device)
        for haplotype in self.haplotypes:
            params = [fg_lambda, bg_lambda] if haplotype == 0 else [bg_lambda, fg_lambda]
            self._create_rigid_chain(params, haplotype)
        self._add_transitions()
        log.debug(
            f'Finished initialising model with {self._model.n_distributions} distributions'
        )

    def estimate_params(self, X):
        X_fg = []
        X_bg = []
        for x in X:
            fg_idx = predict_foreground_convolution(x, self.rfactor)
            idx = np.arange(len(x))
            X_fg.append(x[idx, fg_idx])
            X_bg.append(x[idx, 1 - fg_idx])
        X_fg = np.concatenate(X_fg)
        X_bg = np.concatenate(X_bg)
        return np.mean(X_fg), np.mean(X_bg)

    def fit(self, X):
        fg_lambda, bg_lambda = self.estimate_params(X)
        log.debug(
            'Estimated model parameters from data: '
            f'fg_lambda {fg_lambda:.2g}, bg_lambda {bg_lambda:.2g}'
        )
        self.initialise_model(fg_lambda, bg_lambda)

    def predict(self, X, batch_size=128):
        proba = []
        for X_batch in np.array_split(X, int(np.ceil(len(X) / batch_size))):
            X_batch = torch.from_numpy(X_batch)
            if self._device is not None:
                X_batch.to(self._device)
            p_batch = self._model.predict_proba(X_batch).cpu().numpy()
            # fix nans introduced by pomegranate on extremely tiny probabilities (I think...)
            p_batch[np.isnan(p_batch)] = 0.0
            p_batch = p_batch[..., self._hap2_states].sum(axis=2)
            proba.append(p_batch)
        return np.concatenate(proba, axis=0)

    @property
    def params(self):
        return {
            'rfactor': self.rfactor,
            'term_rfactor': self.term_rfactor,
            'trans_prob': self.trans_prob,
            'fg_lambda': self.fg_lambda,
            'bg_lambda': self.bg_lambda,
        }


def create_rhmm(co_markers, cm_per_mb=4.5,
                segment_size=1_000_000, terminal_segment_size=50_000,
                model_lambdas=None, device=DEFAULT_DEVICE):
    bin_size = co_markers.bin_size
    rfactor = segment_size // bin_size
    term_rfactor = terminal_segment_size // bin_size
    trans_prob = cm_per_mb * (bin_size / 1e8)
    rhmm = RigidHMM(rfactor, term_rfactor, trans_prob, device)
    if model_lambdas is None:
        X = list(co_markers.deep_values())
        rhmm.fit(X)
    else:
        bg_lambda, fg_lambda = sorted(model_lambdas)
        rhmm.initialise_model(fg_lambda, bg_lambda)
    return rhmm


def detect_crossovers(co_markers, rhmm, batch_size=1_000, processes=1):
    seen_barcodes = co_markers.barcodes
    co_preds = PredictionRecords.new_like(co_markers)
    torch.set_num_threads(processes)
    chrom_progress = progress_bar(
        co_markers.chrom_sizes,
        label='Predicting COs',
        item_show_func=str,
    )
    with chrom_progress:
        for chrom in chrom_progress:
            X = np.array([co_markers[cb, chrom] for cb in seen_barcodes])
            X_pred = rhmm.predict(X, batch_size=batch_size)
            for cb, p in zip(seen_barcodes, X_pred):
                co_preds[cb, chrom] = p
    co_preds.metadata['rhmm_params'] = rhmm.params
    return co_preds


def k_nearest_neighbours_classifier(X_train, y_train, k_neighbours):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    kd = KDTree(X_train)

    def _knn(X_predict):
        _, knn_idx = kd.query(X_predict, k=k_neighbours)
        return y_train[knn_idx].mean(axis=1)

    return _knn


def predict_doublet_barcodes(true_co_markers, true_co_preds,
                             sim_co_markers, sim_co_preds,
                             k_neighbours, rng=DEFAULT_RNG):
    X_true = []
    cb_true = []
    for cb, cb_co_markers in true_co_markers.items():
        cb_co_preds = true_co_preds[cb]
        X_true.append([
            stats.accuracy_score(cb_co_markers, cb_co_preds),
            stats.uncertainty_score(cb_co_preds),
            stats.coverage_score(cb_co_markers),
            np.log10(stats.n_crossovers(cb_co_preds) + 1)
        ])
        cb_true.append(cb)
    X_true = np.array(X_true)
    X_doublet = []
    for cb, cb_co_markers in sim_co_markers.items():
        cb_co_preds = sim_co_preds[cb]
        X_doublet.append([
            stats.accuracy_score(cb_co_markers, cb_co_preds),
            stats.uncertainty_score(cb_co_preds),
            stats.coverage_score(cb_co_markers),
            np.log10(stats.n_crossovers(cb_co_preds) + 1)
        ])
    N = len(sim_co_markers)
    X_train = np.concatenate(
        [X_true[rng.integers(0, len(X_true), size=N)], X_doublet],
        axis=0
    )
    y_train = np.repeat([0, 1], [N, N])
    n_neighbours = min(int(N // 2), k_neighbours)
    knn_classifier = k_nearest_neighbours_classifier(X_train, y_train, k_neighbours)
    doublet_pred = knn_classifier(X_true)
    doublet_n = (doublet_pred > 0.5).sum()
    log.info(f'Identified {doublet_n} putative doublets ({doublet_n / len(doublet_pred) * 100:.2f}%)')
    log.debug(
        pd.crosstab(
            pd.Series(knn_classifier(X_train) > 0.5, name='Prediction').map({False: 'hq', True: 'doublet'}),
            pd.Series(y_train, name='Simulation').map({0: 'real', 1: 'sim'}),
        )
    )
    true_co_preds.metadata['doublet_probability'] = {
        cb: p for cb, p in zip(cb_true, doublet_pred)
    }
    return true_co_preds


def doublet_detector(co_markers, co_preds, rhmm,
                     n_doublets, k_neighbours,
                     batch_size=1000, device=DEFAULT_DEVICE,
                     processes=1, rng=DEFAULT_RNG):
    if n_doublets > 1:
        n_sim = int(min(n_doublets, len(co_markers) // 2))
    else:
        n_sim = int(len(co_markers) * n_doublets)

    if k_neighbours > 1:
        k_neighbours = int(min(k_neighbours, n_doublets))
    else:
        k_neighbours = int(n_doublets * k_neighbours)

    log.info(f'Simulating {n_sim} doublets')
    sim_co_markers = simulate_doublets(co_markers, n_sim)
    log.info(f'Predicting crossovers for simulated doublets')
    sim_co_preds = detect_crossovers(
        sim_co_markers, rhmm, batch_size=batch_size, processes=processes
    )
    log.info('Predicting doublets using simulated doublets '
             f'and {k_neighbours} nearest neighbours')
    co_preds = predict_doublet_barcodes(
        co_markers, co_preds,
        sim_co_markers, sim_co_preds,
        k_neighbours, rng=rng,
    )
    return co_preds


def run_predict(marker_json_fn, output_json_fn, *,
                co_markers=None,
                cb_whitelist_fn=None, bin_size=25_000,
                segment_size=1_000_000, terminal_segment_size=50_000,
                cm_per_mb=4.5, model_lambdas=None,
                predict_doublets=True, n_doublets=0.25, k_neighbours=0.25,
                generate_stats=True,
                output_precision=3, processes=1,
                batch_size=1_000, device=DEFAULT_DEVICE,
                rng=DEFAULT_RNG):
    '''
    Uses rigid hidden Markov model to predict the haplotypes of each cell barcode
    at each genomic bin.
    '''
    if co_markers is None:
        co_markers = load_json(marker_json_fn, cb_whitelist_fn, bin_size)
    rhmm = create_rhmm(
        co_markers,
        cm_per_mb=cm_per_mb,
        segment_size=segment_size,
        terminal_segment_size=terminal_segment_size,
        model_lambdas=model_lambdas,
        device=device,
    )
    co_preds = detect_crossovers(
        co_markers, rhmm, batch_size=batch_size, processes=processes
    )
    if predict_doublets:
        co_preds = doublet_detector(
            co_markers, co_preds, rhmm, n_doublets, k_neighbours,
            batch_size=batch_size, processes=processes,
            device=device, rng=rng,
        )

    if generate_stats:
        output_tsv_fn = f'{os.path.splitext(output_json_fn)[0]}.stats.tsv'
        stats.run_stats(
            None, None, output_tsv_fn,
            co_markers=co_markers,
            co_preds=co_preds,
            output_precision=output_precision
        )

    if output_json_fn is not None:
        log.info(f'Writing predictions to {output_json_fn}')
        co_preds.write_json(output_json_fn, output_precision)
    return co_preds


def run_doublet(marker_json_fn, pred_json_fn, output_json_fn, *,
                cb_whitelist_fn=None, bin_size=25_000,
                n_doublets=0.25, k_neighbours=0.25,
                generate_stats=True, output_precision=3, batch_size=1_000,
                processes=1, device=DEFAULT_DEVICE, rng=DEFAULT_RNG):
    co_markers = load_json(marker_json_fn, cb_whitelist_fn, bin_size)
    co_preds = load_json(
        pred_json_fn, cb_whitelist_fn, bin_size, data_type='predictions'
    )

    if set(co_preds.barcodes) != set(co_markers.barcodes):
        raise ValueError('Cell barcodes from marker-json-fn and predict-json-fn do not match')

    rparams = co_preds.metadata['rhmm_params']
    rhmm = RigidHMM(
        rparams['rfactor'],
        rparams['term_rfactor'],
        rparams['trans_prob'],
        device
    )
    rhmm.initialise_model(rparams['fg_lambda'], rparams['bg_lambda'])
    
    co_preds = doublet_detector(
        co_markers, co_preds, rhmm,
        n_doublets, k_neighbours,
        batch_size=batch_size, processes=processes,
        device=device, rng=rng,
    )

    if generate_stats:
        output_tsv_fn = f'{os.path.splitext(output_json_fn)[0]}.stats.tsv'
        stats.run_stats(
            None, None, output_tsv_fn,
            co_markers=co_markers,
            co_preds=co_preds,
            output_precision=output_precision
        )
    
    if output_json_fn is not None:
        log.info(f'Writing predictions to {output_json_fn}')
        co_preds.write_json(output_json_fn, output_precision)
    return co_preds