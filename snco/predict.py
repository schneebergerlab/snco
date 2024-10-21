import json
from collections import namedtuple, defaultdict
import numpy as np

import torch
from pomegranate import distributions as pmd
from pomegranate import hmm as pmh

from .signal import predict_foreground_convolution


class RigidHMM:

    haplotypes = (0, 1)
    _Transitions = namedtuple(
        'Transitions',
        ['self_loop', 'rigid', 'co', 'start', 'end'],
        defaults=[0.0, 0.0, 0.0, 0.0, 0.0]
    )

    
    def __init__(self, rfactor, term_rfactor, trans_prob):
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
        for i in range(self.rfactor):
            d = pmd.Poisson(params)
            self._model.add_distribution(d)
            self._distributions[haplotype].append(d)

    def _add_transitions(self):
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
        self._model = pmh.DenseHMM(frozen=True)
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

    def predict(self, X):
        X = torch.from_numpy(X)
        proba = self._model.predict_proba(X)
        proba = proba[..., self._hap2_states].sum(axis=2).numpy()
        return proba


def create_rhmm(co_markers, bin_size=25_000, cm_per_mb=4.5,
                segment_size=1_000_000, terminal_segment_size=50_000,
                model_lambdas='auto'):
    rfactor = segment_size // bin_size
    term_rfactor = terminal_segment_size // bin_size
    trans_prob = cm_per_mb * (bin_size / 1e8)
    rhmm = RigidHMM(rfactor, term_rfactor, trans_prob)
    if model_lambdas == 'auto':
        X = [m for cb_co_markers in co_markers.values() for m in cb_co_markers.values()]
        rhmm.fit(X)
    else:
        rhmm.initialise_model(*model_lambdas)
    return rhmm


def detect_crossovers(co_markers, rhmm, chrom_sizes, processes=1):
    cb_whitelist = list(co_markers.keys())
    co_preds = defaultdict(dict)
    torch.set_num_threads(processes)
    for chrom in chrom_sizes:
        X = np.array([co_markers[cb][chrom] for cb in cb_whitelist])
        X_pred = rhmm.predict(X)
        for cb, p in zip(cb_whitelist, X_pred):
            co_preds[cb][chrom] = p
    return dict(co_preds)


def co_preds_to_json(output_fn, co_preds, chrom_sizes, bin_size, precision=2):
    co_preds_json_serialisable = {}
    for cb, cb_co_preds in co_preds.items():
        d = {}
        for chrom, pred in cb_co_preds.items():
            d[chrom] = [round(float(p), precision) for p in pred]
        co_preds_json_serialisable[cb] = d
    with open(output_fn, 'w') as o:
        return json.dump({
            'bin_size': bin_size,
            'chrom_sizes': chrom_sizes,
            'data': co_preds_json_serialisable
        }, fp=o)