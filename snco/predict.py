import os
import logging
from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

import torch
from pomegranate import distributions as pmd
from pomegranate.hmm import DenseHMM
from pomegranate.gmm import GeneralMixtureModel

from .logger import progress_bar
from .utils import load_json
from .records import PredictionRecords, NestedData
from .clean import predict_foreground_convolution
from .sim import simulate_doublets
from . import stats
from .opts import DEFAULT_RANDOM_SEED


log = logging.getLogger('snco')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
DEFAULT_DEVICE = torch.device('cpu')


def interp_nan_inplace(arr):
    """
    Interpolates NaN values in-place within a NumPy array using linear interpolation.

    Parameters
    ----------
    arr : np.ndarray
        A NumPy array potentially containing NaN values. This array is modified in-place.
    """
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        real_mask = ~nan_mask
        xp = real_mask.ravel().nonzero()[0]
        fp = arr[real_mask]
        x = nan_mask.ravel().nonzero()[0]
        arr[nan_mask] = np.interp(x, xp, fp)


class RigidHMM:
    """
    Implements a rigid Hidden Markov Model (rHMM) for modeling haplotype transitions
    with fixed structure and constrained transition probabilities.

    Parameters
    ----------
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
        self.empty_fraction = None
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
            d = GeneralMixtureModel([pmd.Poisson(params), pmd.Poisson([self.bg_lambda, self.bg_lambda])],
                                    priors=[1 - self.empty_fraction, self.empty_fraction])
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

    def initialise_model(self, fg_lambda, bg_lambda, empty_fraction):
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
        self.fg_lambda = fg_lambda
        self.bg_lambda = bg_lambda
        self.empty_fraction = empty_fraction
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
        """
        Estimates zero inflated Poisson mixture parameters from input data.

        Parameters
        ----------
        X : list of np.ndarray
            List of 2D arrays containing haplotype-specific read/variant counts per barcode.

        Returns
        -------
        fg_lambda : float
            Estimated mean for foreground component.
        bg_lambda : float
            Estimated mean for background component.
        empty_fraction : float
            Estimated fraction of empty bins in zero inflated model.
        """
        X_fg = []
        X_bg = []
        for x in X:
            fg_idx = predict_foreground_convolution(x, self.rfactor)
            idx = np.arange(len(x))
            X_fg.append(x[idx, fg_idx])
            X_bg.append(x[idx, 1 - fg_idx])
        X_fg = np.concatenate(X_fg)
        X_bg = np.concatenate(X_bg)
        X_ordered = np.stack([X_fg, X_bg], axis=1)
        fg_lambda = np.mean(X_fg)
        bg_lambda = np.mean(X_bg)
        gmm = GeneralMixtureModel([
                pmd.Poisson([fg_lambda, bg_lambda], frozen=True),
                pmd.Poisson([bg_lambda, bg_lambda], frozen=True)
            ],
        ).fit(X_ordered)
        empty_fraction = gmm.priors.numpy()[1]
        return fg_lambda, bg_lambda, empty_fraction

    def fit(self, X):
        """
        Fits the rHMM model to input data by estimating parameters and initializing the model.

        Parameters
        ----------
        X : list of np.ndarray
            List of 2D arrays containing haplotype-specific read/variant counts per barcode.
        """
        fg_lambda, bg_lambda, empty_fraction = self.estimate_params(X)
        log.debug(
            'Estimated model parameters from data: '
            f'fg_lambda {fg_lambda:.2g}, bg_lambda {bg_lambda:.2g}, empty_fraction {empty_fraction:.2g}'
        )
        self.initialise_model(fg_lambda, bg_lambda, empty_fraction)

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
            X_batch = torch.from_numpy(X_batch)
            if self._device is not None:
                X_batch.to(self._device)
            p_batch = self._model.predict_proba(X_batch).cpu().numpy()
            p_batch = p_batch[..., self._hap2_states].sum(axis=2)
            # fix nans introduced by pomegranate on positions where all state logprobs are -inf
            # seems to occur 
            for p in p_batch:
                interp_nan_inplace(p)
            proba.append(p_batch)
        return np.concatenate(proba, axis=0)

    @property
    def params(self):
        return {
            'rfactor': int(self.rfactor),
            'term_rfactor': int(self.term_rfactor),
            'trans_prob': float(self.trans_prob),
            'fg_lambda': float(self.fg_lambda),
            'bg_lambda': float(self.bg_lambda),
            'empty_fraction': float(self.empty_fraction),
        }


def create_rhmm(co_markers, cm_per_mb=4.5,
                segment_size=1_000_000, terminal_segment_size=50_000,
                model_lambdas=None, empty_fraction=None, device=DEFAULT_DEVICE):
    """
    Constructs a RigidHMM instance and fits it to the crossover marker data.

    Parameters
    ----------
    co_markers : MarkerRecords
        MarkerRecords dataset with haplotype specific read/variant information.
    cm_per_mb : float, optional
        Recombination rate in centimorgans per megabase (default: 4.5).
    segment_size : int, optional
        Size of internal genomic segments, i.e. minimum distance between crossovers 
        (default: 1,000,000 bp).
    terminal_segment_size : int, optional
        Size of terminal genomic segments, i.e. minimum distance between crossovers and 
        ends of chromosomes (default: 50,000 bp).
    model_lambdas : tuple of float, optional
        Tuple of (background_lambda, foreground_lambda). If None, estimate from data.
    empty_fraction : float, optional
        Estimated fraction of empty bins in zero inflated model. If None, estimate from data.
    device : torch.device, optional
        Device to initialize the model on (default: cpu).

    Returns
    -------
    RigidHMM
        Initialized and optionally fitted RigidHMM instance.
    """
    bin_size = co_markers.bin_size
    rfactor = segment_size // bin_size
    term_rfactor = terminal_segment_size // bin_size
    trans_prob = cm_per_mb * (bin_size / 1e8)
    rhmm = RigidHMM(rfactor, term_rfactor, trans_prob, device)
    if model_lambdas is None or empty_fraction is None:
        X = list(co_markers.deep_values())
        rhmm.fit(X)
    else:
        bg_lambda, fg_lambda = sorted(model_lambdas)
        rhmm.initialise_model(fg_lambda, bg_lambda, empty_fraction)
    return rhmm


def detect_crossovers(co_markers, rhmm, batch_size=128, processes=1):
    """
    Applies an rHMM to predict crossovers from marker data.

    Parameters
    ----------
    co_markers : MarkerRecords
        MarkerRecords dataset with haplotype specific read/variant information.
    rhmm : RigidHMM
        Fitted RigidHMM model.
    batch_size : int, optional
        Batch size for prediction (default: 128).
    processes : int, optional
        Number of threads for prediction (default: 1).

    Returns
    -------
    PredictionRecords
        PredictionRecords dataset with haplotype probabilities.
    """
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
    co_preds.add_metadata(
        rhmm_params=NestedData(levels=('other', ), dtype=(int, float), data=rhmm.params)
    )
    return co_preds


def k_nearest_neighbours_classifier(X_train, y_train, k_neighbours):
    """
    Constructs a k-nearest neighbors classifier using KDTree.

    Parameters
    ----------
    X_train : array-like
        Training feature matrix.
    y_train : array-like
        Labels corresponding to training data.
    k_neighbours : int
        Number of neighbors to use in prediction.

    Returns
    -------
    callable
        A function that takes an array `X_predict` and returns predicted probabilities.
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    kd = KDTree(X_train)

    def _knn(X_predict):
        _, knn_idx = kd.query(X_predict, k=k_neighbours)
        return y_train[knn_idx].mean(axis=1)

    return _knn


def generate_doublet_prediction_features(co_markers, co_preds):
    """
    Generates feature vectors for predicting doublet barcodes.

    Parameters
    ----------
    co_markers : MarkerRecords
        MarkerRecords dataset with haplotype specific read/variant information.
    co_preds : PredictionRecords
        PredictionRecords dataset with matched haplotype probabilities.

    Returns
    -------
    X : np.ndarray
        Feature matrix for each barcode.
    barcodes : list of str
        Corresponding barcode identifiers.
    """
    X = []
    barcodes = []
    for cb, cb_co_markers in co_markers.items():
        cb_co_preds = co_preds[cb]
        X.append([
            stats.accuracy_score(cb_co_markers, cb_co_preds),
            stats.uncertainty_score(cb_co_preds),
            stats.coverage_score(cb_co_markers),
            np.log10(stats.n_crossovers(cb_co_preds) + 1)
        ])
        barcodes.append(cb)
    X = np.array(X)
    return X, barcodes


def min_max_normalise(*X_arrs):
    """
    Applies min-max normalization across multiple feature matrices.

    Parameters
    ----------
    *X_arrs : list of np.ndarray
        One or more feature matrices to normalize jointly.

    Returns
    -------
    list of np.ndarray
        Normalized feature matrices.
    """
    X_min = np.min([X.min(axis=0) for X in X_arrs], axis=0)
    X_max = np.max([X.max(axis=0) for X in X_arrs], axis=0)
    return [(X - X_min) / (X_max - X_min) for X in X_arrs]


def predict_doublet_barcodes(true_co_markers, true_co_preds,
                             sim_co_markers, sim_co_preds,
                             k_neighbours, rng=DEFAULT_RNG):
    """
    Predicts doublets among barcodes by training a KNN classifier on simulated doublets.

    Parameters
    ----------
    true_co_markers : MarkerRecords
        Original MarkerRecords dataset with haplotype specific read/variant information.
    true_co_preds : PredictionRecords
        Haplotype predictions for original markers.
    sim_co_markers : MarkerDataset
        Simulated MarkerRecords dataset with synthetic doublets.
    sim_co_preds : PredictionRecords
        Haplotype predictions for simulated markers.
    k_neighbours : int
        Number of neighbors to use in KNN.
    rng : np.random.Generator, optional
        Random number generator instance.

    Returns
    -------
    PredictionRecords
        Updated predictions for original dataset including doublet probability annnotations.
    """
    X_true, cb_true = generate_doublet_prediction_features(
        true_co_markers, true_co_preds
    )
    X_doublet, _ = generate_doublet_prediction_features(
        sim_co_markers, sim_co_preds
    )
    N = len(sim_co_markers)
    X_train = np.concatenate(
        [X_true[rng.integers(0, len(X_true), size=N)], X_doublet],
        axis=0
    )
    y_train = np.repeat([0, 1], [N, N])

    X_train, X_true = min_max_normalise(X_train, X_true)

    k_neighbours = min(int(N // 2), k_neighbours)
    knn_classifier = k_nearest_neighbours_classifier(X_train, y_train, k_neighbours)
    doublet_pred = knn_classifier(X_true)
    doublet_n = (doublet_pred > 0.5).sum()
    log.info(
        f'Identified {doublet_n} putative doublets ({doublet_n / len(doublet_pred) * 100:.2f}%)'
    )
    if log.getEffectiveLevel() <= logging.DEBUG:
        X_pred_series = pd.Series(
            knn_classifier(X_train) > 0.5, name='Prediction'
        ).map({False: 'hq', True: 'doublet'})
        y_pred_series = pd.Series(y_train, name='Simulation').map({0: 'real', 1: 'sim'})
        log.debug(pd.crosstab(X_pred_series, y_pred_series))
    
    doublet_probs = dict(zip(cb_true, doublet_pred))
    true_co_preds.add_metadata(
        doublet_probability=NestedData(levels=('cb', ), dtype=float, data=doublet_probs)
    )
    return true_co_preds


def doublet_detector(co_markers, co_preds, rhmm, n_doublets, k_neighbours,
                     batch_size=1000, processes=1, rng=DEFAULT_RNG):
    """
    Detects and flags doublet cell barcodes using simulated doublets and KNN.

    Parameters
    ----------
    co_markers : MarkerDataset
        Original MarkerRecords dataset with haplotype specific read/variant information.
    co_preds : PredictionRecords
        Haplotype predictions for original markers.
    rhmm : RigidHMM
        Fitted rHMM model.
    n_doublets : float or int
        Number or fraction of simulated doublets.
    k_neighbours : float or int
        Number or fraction of neighbors to use in KNN.
    batch_size : int, optional
        Batch size for rHMM prediction (default: 1000).
    processes : int, optional
        Number of threads (default: 1).
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    PredictionRecords
        Updated predictions for dataset including doublet probability annnotations.
    """
    if n_doublets > 1:
        n_sim = int(min(n_doublets, len(co_markers) // 2))
    else:
        if not n_doublets:
            raise ValueError('n-doublets must be >0 for doublet detection')
        n_sim = int(len(co_markers) * n_doublets)

    if k_neighbours > 1:
        k_neighbours = int(min(k_neighbours, n_sim))
    else:
        k_neighbours = int(n_sim * k_neighbours)

    log.info(f'Simulating {n_sim} doublets')
    sim_co_markers = simulate_doublets(co_markers, n_sim)
    log.info('Predicting crossovers for simulated doublets')
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
                generate_stats=True, nco_min_prob_change=2.5e-3,
                output_precision=3, processes=1,
                batch_size=1_000, device=DEFAULT_DEVICE,
                rng=DEFAULT_RNG):
    """
    Runs the full haplotype prediction pipeline from marker JSON to output.

    Parameters
    ----------
    marker_json_fn : str
        Path to input JSON with haplotype specific marker data.
    output_json_fn : str
        Path to output predictions JSON file.
    co_markers : MarkerDataset, optional
        Loaded haplotype specific marker dataset.
    cb_whitelist_fn : str, optional
        Path to barcode whitelist file.
    bin_size : int, optional
        Genomic bin size (default: 25,000).
    segment_size : int, optional
        Size of internal segments for modeling (default: 1,000,000).
    terminal_segment_size : int, optional
        Size of terminal segments (default: 50,000).
    cm_per_mb : float, optional
        Centimorgan per megabase rate (default: 4.5).
    model_lambdas : tuple of float, optional
        Optional Poisson lambdas.
    predict_doublets : bool, optional
        Whether to detect doublets (default: True).
    n_doublets : float or int, optional
        Number or fraction of doublets to simulate (default: 0.25).
    k_neighbours : float or int, optional
        Number or fraction of neighbors for KNN (default: 0.25).
    generate_stats : bool, optional
        Whether to generate prediction statistics (default: True).
    nco_min_prob_change : float, optional
        Threshold for detecting non-crossover changes (default: 2.5e-3).
    output_precision : int, optional
        Decimal precision for JSON output.
    processes : int, optional
        Number of threads (default: 1).
    batch_size : int, optional
        Batch size for prediction.
    device : torch.device, optional
        Device to run models on.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    PredictionRecords
        Final predictions including crossover and optional doublet annotations.
    """
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
            batch_size=batch_size, processes=processes, rng=rng,
        )

    if generate_stats:
        output_tsv_fn = f'{os.path.splitext(output_json_fn)[0]}.stats.tsv'
        stats.run_stats(
            None, None, output_tsv_fn,
            co_markers=co_markers,
            co_preds=co_preds,
            nco_min_prob_change=nco_min_prob_change,
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
    """
    Loads pre-existing crossover predictions and performs doublet detection.

    Parameters
    ----------
    marker_json_fn : str
        Path to crossover marker JSON.
    pred_json_fn : str
        Path to precomputed predictions JSON.
    output_json_fn : str
        Output path to write updated predictions.
    cb_whitelist_fn : str, optional
        Whitelist file for barcodes.
    bin_size : int, optional
        Genomic bin size (default: 25,000).
    n_doublets : float or int, optional
        Number or fraction of doublets to simulate.
    k_neighbours : float or int, optional
        Number or fraction of KNN neighbors.
    generate_stats : bool, optional
        Whether to generate output statistics (default: True).
    output_precision : int, optional
        Decimal precision for output (default: 3).
    batch_size : int, optional
        Batch size for HMM prediction.
    processes : int, optional
        Number of threads to use.
    device : torch.device, optional
        Model device (default: CPU).
    rng : np.random.Generator, optional
        Random number generator instance.

    Returns
    -------
    PredictionRecords
        Crossover/haplotype predictions with doublet probabilities added.
    """
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
        device=device
    )
    rhmm.initialise_model(rparams['fg_lambda'], rparams['bg_lambda'], rparams['empty_fraction'])

    co_preds = doublet_detector(
        co_markers, co_preds, rhmm,
        n_doublets, k_neighbours,
        batch_size=batch_size, processes=processes,
        rng=rng,
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
