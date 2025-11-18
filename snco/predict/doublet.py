import logging

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import pearsonr

import torch

from .rhmm import RigidHMM
from .crossovers import detect_crossovers
from ..records import NestedData
from ..sim import simulate_doublets
from .. import stats
from ..defaults import DEFAULT_RANDOM_SEED
from ..main.logger import progress_bar


log = logging.getLogger('snco')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
DEFAULT_DEVICE = torch.device('cpu')


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

def per_chrom_cov_corr(co_markers):
    av_cov = {}
    for chrom in co_markers.chrom_sizes:
        av_cov[chrom] = co_markers[:, chrom].stack_values().sum(axis=None) / len(co_markers)
    chroms = list(av_cov)
    av_cov = list(av_cov.values())
    corrs = {}
    for cb, cb_co_markers in co_markers.items():
        cov = [cb_co_markers[chrom].sum() for chrom in chroms]
        corrs[cb] = np.log(1 - (pearsonr(cov, av_cov)[0] + 1) / 2)
    return corrs


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

    corrs = per_chrom_cov_corr(co_markers)
    for cb, cb_co_markers in co_markers.items():
        cb_co_preds = co_preds[cb]
        X.append([
            stats.accuracy_score(cb_co_markers, cb_co_preds),
            np.exp(co_preds.metadata['logprobs'][cb] / co_markers.total_marker_count(cb)),
            corrs[cb],
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


def detect_doublets(co_markers, co_preds, rhmm, n_doublets, k_neighbours,
                    max_iter=25, early_stopping_thresh=0.01, inertia=0.95,
                    batch_size=128, processes=1, rng=DEFAULT_RNG):
    """
    Detects and flags doublet cell barcodes using simulated doublets and iterative KNN.
    Doublets are simulated by combining random barcodes and predicting crossover patterns.
    Summary statistics are then compared to the real data to identify likely doublets and singlets.
    n_iter rounds of KNN are performed to improve the separation of singlets and doublets.

    Parameters
    ----------
    co_markers : MarkerDataset
        Original MarkerRecords dataset with haplotype specific read/variant information.
    co_preds : PredictionRecords
        Haplotype predictions for original markers.
    rhmm : RigidHMM
        Fitted rHMM model.
    n_doublets : float or int
        Number or fraction of simulated doublets used per iteration.
    k_neighbours : float or int
        Number or fraction of neighbors to use in KNN.
        The maximum number used is sqrt(n_doublets * 2)
    max_iter : int
        Maximum number of iterations of KNN to do.
    early_stopping_thresh : float
        The mean change in doublet probability at which to stop the doublet prediction.
    inertia : float
        The weighting of the observed doublet rate vs the expectation on each update.
    batch_size : int, optional
        Batch size for rHMM prediction (default: 128).
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
        n_sim = int(min(n_doublets, len(co_markers)))
    else:
        if not n_doublets:
            raise ValueError('n-doublets must be >0 for doublet detection')
        n_sim = int(len(co_markers) * n_doublets)

    if k_neighbours < 1.0:
        k_neighbours = int(n_sim * k_neighbours)
    k_neighbours = int(min(k_neighbours, max(np.sqrt(n_sim * 2), 10)))

    X_true, cb_true = generate_doublet_prediction_features(
        co_markers, co_preds,
    )

    n_sim_pool = n_sim * min(max_iter, 5)
    log.info(f'Simulating {n_sim_pool} doublets')
    sim_co_markers = simulate_doublets(co_markers, n_sim_pool, rng=rng)
    sim_co_preds = detect_crossovers(
        sim_co_markers, rhmm,
        sample_paths=False,
        batch_size=batch_size,
        processes=processes,
    )

    X_doublet, _ = generate_doublet_prediction_features(
        sim_co_markers, sim_co_preds,
    )

    doublet_rate = np.nan
    neg_idx = rng.integers(0, len(X_true), size=n_sim)
    pos_idx = rng.integers(0, len(X_doublet), size=n_sim)
    y_train = np.repeat([0, 1], [n_sim, n_sim])
    p_doublet = None
    p_delta = 1.0

    def _prog_func(i):
        i = 0 if i is None else i
        return f'[{i}/{max_iter}], DR:{doublet_rate:.0f}% ΔP: {p_delta:.2f}'

    doublet_progress = progress_bar(
        list(range(1, max_iter + 1)),
        label=f'Detecting doublets',
        item_show_func=_prog_func,
    )


    with doublet_progress:
        for it in doublet_progress:
            X_train = np.concatenate([X_true[neg_idx], X_doublet[pos_idx]], axis=0)
            X_train_n, X_true_n = min_max_normalise(X_train, X_true)
            knn = k_nearest_neighbours_classifier(X_train_n, y_train, k_neighbours)
            p_update = knn(X_true_n)
            p_delta = np.mean(np.abs(p_update - p_doublet)) if p_doublet is not None else 1.0
            if p_delta < early_stopping_thresh:
                break
            p_doublet = p_update
            doublet_rate = (p_doublet > 0.5).mean() * 100
            w = np.clip((1.0 - p_doublet) ** 1.5, 1e-12, None)
            w /= w.sum()
            neg_idx = rng.choice(len(X_true), size=n_sim, replace=True, p=w)
            pos_idx = rng.integers(0, len(X_doublet), size=n_sim)

    doublet_probs = dict(zip(cb_true, p_doublet))
    co_preds.add_metadata(
        doublet_probability=NestedData(
            levels=('cb',), dtype=float, data=doublet_probs
        )
    )

    return co_preds
