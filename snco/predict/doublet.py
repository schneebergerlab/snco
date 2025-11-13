import logging

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

import torch

from .rhmm import RigidHMM
from .crossovers import detect_crossovers
from ..records import NestedData
from ..sim import simulate_doublets
from .. import stats
from ..defaults import DEFAULT_RANDOM_SEED


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
            #np.log10(stats.n_crossovers(cb_co_preds) + 1)
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


def classify_doublet_barcodes(true_co_markers, true_co_preds,
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


def detect_doublets(co_markers, co_preds, rhmm, n_doublets, k_neighbours,
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
        sim_co_markers, rhmm,
        sample_paths=False,
        batch_size=batch_size, processes=processes
    )
    log.info('Classifying doublets using simulated doublets '
             f'and {k_neighbours} nearest neighbours')
    co_preds = classify_doublet_barcodes(
        co_markers, co_preds,
        sim_co_markers, sim_co_preds,
        k_neighbours, rng=rng,
    )
    return co_preds