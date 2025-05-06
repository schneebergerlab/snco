import os
import logging
from copy import deepcopy
from dataclasses import dataclass
import numpy as np

import torch
from pomegranate import distributions as pmd
from pomegranate.hmm import DenseHMM
from pomegranate.gmm import GeneralMixtureModel

from .utils import interp_nan_inplace


log = logging.getLogger('snco')
DEFAULT_DEVICE = torch.device('cpu')


@dataclass
class Transition:
    self_loop: float = 0.0
    rigid: float = 0.0
    co: float = 0.0
    start: float = 0.0
    end: float = 0.0


class RigidHMM:
    """
    Implements a rigid Hidden Markov Model (rHMM) for modeling haplotype transitions
    with fixed structure and constrained transition probabilities.

    Parameters
    ----------
    states : list of tuple, list or set
        The haplotype states to produce a model for
    rfactor : int
        Number of bins in a genomic segment - constrains the minimum distance between transitions.
    term_rfactor : int
        Number of bins in a terminal genomic segment - constrains the minimum distance between 
        end of sequence and transitions.
    trans_prob : float
        Probability of transition between rigid states.
    device : torch.device, optional
        Device on which to allocate model (default: CPU).
    """

    def __init__(self, states, rfactor, term_rfactor, trans_prob,
                 fg_lambda, bg_lambda, empty_fraction,
                 device=DEFAULT_DEVICE):
        for hap_comb in states:
            if not isinstance(hap_comb, (tuple, list, set, frozenset)):
                raise ValueError(
                    'states should be a list/tuple of tuples, which represent haplotype combinations'
                )
            for hap in hap_comb:
                if hap not in (0, 1):
                    raise ValueError('haplotypes can only be 0 or 1')
        self.states = states
        self.nstates = len(self.states)
        self._state_haplo = np.mean(self.states, axis=1)
        self.rfactor = rfactor
        self.term_rfactor = term_rfactor
        self.trans_prob = trans_prob
        self.term_prob = 1 / (self.rfactor - self.term_rfactor)
        assert 0.0 < self.trans_prob < 1.0
        assert (self.trans_prob + self.term_prob) < 1.0
        self.fg_lambda = fg_lambda
        self.bg_lambda = bg_lambda
        self.empty_fraction = empty_fraction
        self._device = device
        self.initialise_model()

    def _calculate_transition_probs(self):
        for i in range(self.rfactor):
            # first state of chain
            if i == 0:
                self._transition_probs[i] = Transition(
                    self_loop=1.0 - self.trans_prob - self.term_prob,
                    rigid=self.trans_prob,
                    start=self.term_prob,
                )
            # last state of chain
            elif (i + 1) == self.rfactor:
                self._transition_probs[i] = Transition(
                    self_loop=1.0 - self.trans_prob - self.term_prob,
                    co=self.trans_prob / (self.nstates - 1),
                    end=self.term_prob
                )
            # internal point of the chain
            else:
                start_prob = self.term_prob if i <= (self.rfactor - self.term_rfactor) else 0.0
                end_prob = self.term_prob if (i + 1) >= self.term_rfactor else 0.0
                self._transition_probs[i] = Transition(
                    self_loop=0.0,
                    rigid=1.0 - end_prob,
                    start=start_prob,
                    end=end_prob
                )

    @property
    def _empty_poisson(self):
        if self.bg_lambda is None:
            raise ValueError('Cannot get bg_dist before model is fit or initialised')
        return pmd.Poisson([self.bg_lambda, self.bg_lambda])

    def _nonempty_poisson(self, hap):
        if hap == 0:
            return pmd.Poisson([self.fg_lambda, self.bg_lambda])
        elif hap == 1:
            return pmd.Poisson([self.bg_lambda, self.fg_lambda])
        else:
            raise ValueError('hap can only be 0 or 1')

    def _create_distribution(self, state):
        dists = [self._empty_poisson,]
        priors = [self.empty_fraction,]
        per_state_prior = (1 - self.empty_fraction) / len(state)
        for hap in state:
            dists.append(self._nonempty_poisson(hap))
            priors.append(per_state_prior)
        return GeneralMixtureModel(dists, priors=priors)

    def _create_rigid_chain(self, state):
        dist = self._create_distribution(state)
        self._distributions[state] = []
        for _ in range(self.rfactor):
            dist = deepcopy(dist)
            self._model.add_distribution(dist)
            self._distributions[state].append(dist)

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

        for state in self.states:
            for i in range(self.rfactor):
                if self._transition_probs[i].self_loop:
                    self._model.add_edge(
                        self._distributions[state][i],
                        self._distributions[state][i],
                        self._transition_probs[i].self_loop
                    )
                if self._transition_probs[i].rigid:
                    self._model.add_edge(
                        self._distributions[state][i],
                        self._distributions[state][i + 1],
                        self._transition_probs[i].rigid
                    )
                if self._transition_probs[i].co:
                    for other in self.states:
                        if state != other:
                            self._model.add_edge(
                                self._distributions[state][i],
                                self._distributions[other][0],
                                self._transition_probs[i].co
                            )
                if self._transition_probs[i].start:
                    self._model.add_edge(
                        self._model.start,
                        self._distributions[state][i],
                        self._transition_probs[i].start
                    )
                if self._transition_probs[i].end:
                    self._model.add_edge(
                        self._distributions[state][i],
                        self._model.end,
                        self._transition_probs[i].end
                    )

    def initialise_model(self):
        """
        Initializes the DenseHMM model using estimated Poisson parameters.

        Parameters
        ----------
        fg_lambda : float
            Foreground (signal) Poisson mean.
        bg_lambda : float
            Background Poisson mean.
        empty_fraction : float
            Fraction of empty bins in zero inflated model.
        """
        self._transition_probs = {}
        self._calculate_transition_probs()
        self._distributions = {}
        self._chains = {}
        self._model = DenseHMM(frozen=True)
        for state in self.states:
            self._create_rigid_chain(state)
        self._add_transitions()
        log.debug(f'moving model to device: {self._device}')
        self._model.to(self._device)
        log.debug(
            f'Finished initialising model with {self._model.n_distributions} distributions'
        )

    def predict(self, X, batch_size=128):
        """
        Predicts haplotype probabilities for input marker arrays.

        Parameters
        ----------
        X : list of np.ndarray
            List of 2D arrays containing haplotype-specific read/variant counts per barcode.
        batch_size : int, optional
            Batch size for model prediction (default: 128).

        Returns
        -------
        np.ndarray
            Array of predicted probabilities.
        """
        proba = []
        for X_batch in np.array_split(X, int(np.ceil(len(X) / batch_size))):
            batch_size, chrom_size = X_batch.shape[:2]
            X_batch = torch.from_numpy(X_batch)
            if self._device is not None:
                X_batch.to(self._device)
            p_batch = self._model.predict_proba(X_batch).cpu().numpy()
            # sum over all chain distributions representing the same state
            p_batch = p_batch.reshape(batch_size, chrom_size, self.nstates, self.rfactor).sum(axis=3)
            # fix nans introduced by pomegranate on positions where all state logprobs are -inf
            # seems to occur
            interp_nan_inplace(p_batch, axis=1) # interp along chromosome axis
            proba.append(p_batch)
        proba = np.concatenate(proba, axis=0)
        # convert to single value per bin, representing the probability of alt hap
        return proba @ self._state_haplo

    @property
    def params(self):
        return {
            'states': [list(s) for s in self.states],
            'rfactor': int(self.rfactor),
            'term_rfactor': int(self.term_rfactor),
            'trans_prob': float(self.trans_prob),
            'fg_lambda': float(self.fg_lambda),
            'bg_lambda': float(self.bg_lambda),
            'empty_fraction': float(self.empty_fraction),
        }