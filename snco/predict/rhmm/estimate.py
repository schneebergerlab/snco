import logging

import numpy as np

import torch
from pomegranate import distributions as pmd
from pomegranate.gmm import GeneralMixtureModel

from snco.signal import align_foreground_column, detect_homozygous_bins


def estimate_haploid_emissions(X, window=40):
    """
    Estimate foreground/background Poisson parameters for haploid data using a
    zero-inflated model.

    Parameters
    ----------
    X : list of np.ndarray
        List of marker count arrays with shape (bins, 2).
    window : int, optional
        Width of the smoothing window (default is 40).

    Returns
    -------
    tuple of float
        Estimated (fg_lambda, bg_lambda, empty_fraction).
    """
    X_ordered = align_foreground_column(X, window)
    X_flattened = np.concatenate(X_ordered)
    # estimate initial lambdas from means
    init_fg_lambda, init_bg_lambda = np.mean(X_flattened, axis=0)
    # update lambdas using zero inflated GMM
    gmm = GeneralMixtureModel([
        pmd.DiracDelta([1.0, 1.0]),
        pmd.Poisson([init_fg_lambda, init_bg_lambda])
    ]).fit(X_flattened)
    fg_lambda, bg_lambda = gmm.distributions[1].lambdas.numpy()
    empty_fraction = gmm.priors.numpy()[0]
    return fg_lambda, bg_lambda, empty_fraction


def estimate_diploid_emissions_ordered(X_ordered, window=40):
    """
    Estimate Poisson parameters for diploid data which is ordered so that the dominant
    haplotype is in column 0. This can either be used for backcross data where the
    backcross parent is known, or on F2 data where the dominant column has been estimated

    Parameters
    ----------
    X : list of np.ndarray
        List of marker count arrays with shape (bins, 2), foreground must be column 0.
    window : int, optional
        Width of the smoothing window (default is 40).

    Returns
    -------
    tuple of float
        Estimated (fg_lambda, bg_lambda, empty_fraction).
    """
    mask = detect_homozygous_bins(X_ordered, window)
    X_flattened = np.concatenate(X_ordered)
    mask = np.concatenate(mask)
    init_fg_lambda = X_flattened[mask, 1].mean()
    init_bg_lambda = X_flattened[~mask, 1].mean()

    hom = GeneralMixtureModel([
        pmd.Poisson([init_fg_lambda, init_bg_lambda]),
        pmd.Poisson([init_fg_lambda, init_bg_lambda]),
    ], frozen=True) # freezes priors but not lambdas
    het = GeneralMixtureModel([
        pmd.Poisson([init_fg_lambda, init_bg_lambda]),
        pmd.Poisson([init_bg_lambda, init_fg_lambda]),
    ], frozen=True)
    zi = pmd.DiracDelta([1.0, 1.0])
    gmm = GeneralMixtureModel([zi, hom, het]).fit(X_flattened)
    empty_fraction = gmm.priors.numpy()[0]
    fg_lambda, bg_lambda = hom.distributions[1].lambdas.numpy()
    return fg_lambda, bg_lambda, empty_fraction


def estimate_diploid_emissions_f2(X, window=40):
    """
    Estimate Poisson parameters for diploid F2 data by reordering and calling the ordered estimator.

    Parameters
    ----------
    X : list of np.ndarray
        List of marker count arrays with shape (bins, 2).
    window : int, optional
        Width of the smoothing window (default is 40).

    Returns
    -------
    tuple of float
        Estimated (fg_lambda, bg_lambda, empty_fraction).
    """
    X_ordered = align_foreground_column(X, window)
    return estimate_diploid_emissions_ordered(X_ordered, window)
