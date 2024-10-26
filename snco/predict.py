import logging
from collections import namedtuple
import numpy as np

import torch
from pomegranate import distributions as pmd
from pomegranate import hmm as pmh

from .utils import load_json
from .records import PredictionRecords
from .clean import predict_foreground_convolution

DEFAULT_DEVICE = torch.device('cpu')

log = logging.getLogger('snco')


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
        self._model = pmh.DenseHMM(frozen=True)
        self._model.to(self._device)
        for haplotype in self.haplotypes:
            params = [fg_lambda, bg_lambda] if haplotype == 0 else [bg_lambda, fg_lambda]
            self._create_rigid_chain(params, haplotype)
        self._add_transitions()

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
        self.initialise_model(fg_lambda, bg_lambda)

    def predict(self, X, batch_size=128):
        proba = []
        for X_batch in np.array_split(X, int(np.ceil(len(X) / batch_size))):
            X_batch = torch.from_numpy(X_batch)
            if self._device is not None:
                X_batch.to(self._device)
            p_batch = self._model.predict_proba(X_batch)
            p_batch = p_batch[..., self._hap2_states].sum(axis=2).cpu().numpy()
            proba.append(p_batch)
        return np.concatenate(proba, axis=0)


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
    seen_barcodes = co_markers.seen_barcodes
    co_preds = PredictionRecords.new_like(co_markers)
    torch.set_num_threads(processes)
    for chrom in co_markers.chrom_sizes:
        X = np.array([co_markers[cb, chrom] for cb in seen_barcodes])
        X_pred = rhmm.predict(X, batch_size=batch_size)
        for cb, p in zip(seen_barcodes, X_pred):
            co_preds[cb, chrom] = p
    return co_preds


def run_predict(marker_json_fn, output_json_fn, *,
                cb_whitelist_fn=None, bin_size=25_000,
                segment_size=1_000_000, terminal_segment_size=50_000,
                cm_per_mb=4.5, model_lambdas=None,
                output_precision=2, processes=1,
                batch_size=1_000, device=DEFAULT_DEVICE):
    '''
    Uses rigid hidden Markov model to predict the haplotypes of each cell barcode
    at each genomic bin.
    '''
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
    co_preds.write_json(output_json_fn, output_precision)
