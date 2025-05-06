import logging

import numpy as np
from scipy.ndimage import convolve1d

import torch
from pomegranate import distributions as pmd
from pomegranate.gmm import GeneralMixtureModel

from .base import RigidHMM


log = logging.getLogger('snco')
DEFAULT_DEVICE = torch.device('cpu')


def predict_foreground_convolution(m, ws=100):
    """
    Predict foreground signal by convolution across bins.

    Parameters
    ----------
    m : np.ndarray
        Marker count matrix with shape (bins, haplotypes).
    ws : int, default=100
        Width of the convolution window in bins.

    Returns
    -------
    np.ndarray
        Array of foreground haplotype indices per bin.
    """
    rs = convolve1d(m, np.ones(ws), axis=0, mode='constant', cval=0)
    fg_idx = rs.argmax(axis=1)
    return fg_idx


def estimate_haploid_parameters(X, rfactor):
    X_fg = []
    X_bg = []
    for x in X:
        fg_idx = predict_foreground_convolution(x, rfactor)
        idx = np.arange(len(x))
        X_fg.append(x[idx, fg_idx])
        X_bg.append(x[idx, 1 - fg_idx])
    X_fg = np.concatenate(X_fg)
    X_bg = np.concatenate(X_bg)
    X_ordered = torch.from_numpy(np.stack([X_fg, X_bg], axis=1))
    init_fg_lambda = np.mean(X_fg)
    init_bg_lambda = np.mean(X_bg)
    gmm = GeneralMixtureModel([
            pmd.DiracDelta([1.0, 1.0]),
            pmd.Poisson([init_fg_lambda, init_bg_lambda]),
        ],
        inertia=0.9,
    ).fit(X_ordered)
    empty_fraction = gmm.priors.numpy()[0]
    fg_lambda, bg_lambda = gmm.distributions[1].lambdas.numpy()
    return fg_lambda, bg_lambda, empty_fraction


def predict_homozygous_convolution(m, ws=100, bc_haplotype=0):
    rs = convolve1d(m, np.ones(ws) / ws, axis=0, mode='constant', cval=0)
    return rs[:, bc_haplotype] < (2 * rs[:, 1 - bc_haplotype])


def fit_hom_het_gmm(X_ordered, init_fg_lambda, init_bg_lambda):
    hom_zip = GeneralMixtureModel([
        pmd.DiracDelta([1.0, 1.0]),
        pmd.Poisson([init_fg_lambda, init_bg_lambda]),
        pmd.Poisson([init_fg_lambda, init_bg_lambda]),
    ])
    het_zip = GeneralMixtureModel([
        pmd.DiracDelta([1.0, 1.0]),
        pmd.Poisson([init_fg_lambda, init_bg_lambda]),
        pmd.Poisson([init_bg_lambda, init_fg_lambda]),
    ])
    gmm = GeneralMixtureModel([hom_zip, het_zip])
    gmm.fit(X_ordered)
    empty_fraction = hom_zip.priors.numpy()[0]
    fg_lambda, bg_lambda = hom_zip.distributions[1].lambdas.numpy()
    return fg_lambda, bg_lambda, empty_fraction


def estimate_diploid_bc1_parameters(bc_haplotype=0):
    def _estimate_backcross_parameters(X, rfactor=100):
        X = np.concatenate(X, axis=0)
        het_mask = predict_homozygous_convolution(X, rfactor, bc_haplotype)
        init_fg_lambda = X[het_mask, 1 - bc_haplotype].mean()
        init_bg_lambda = X[~het_mask, 1 - bc_haplotype].mean()
        fg_lambda, bg_lambda, empty_fraction = fit_hom_het_gmm(
            np.flip(X, axis=1) if bc_haplotype == 1 else X,
            init_fg_lambda,
            init_bg_lambda
        )
        return fg_lambda, bg_lambda, empty_fraction
    return _estimate_backcross_parameters


def estimate_diploid_f2_parameters(X, rfactor=100):
    pass


def train_rhmm(co_markers, model_type='haploid', cm_per_mb=4.5,
               segment_size=1_000_000, terminal_segment_size=50_000,
               model_lambdas=None, empty_fraction=None,
               bc_haplotype=0, device=DEFAULT_DEVICE):
    """
    Constructs a RigidHMM instance and fits it to the crossover marker data.

    Parameters
    ----------
    co_markers : MarkerRecords
        MarkerRecords dataset with haplotype specific read/variant information.
    model_type : str
        Type of rHMM to create. One of "haploid", "diploid_backcross" or "diploid_f2"
    cm_per_mb : float, optional
        Recombination rate in centimorgans per megabase (default: 4.5).
    segment_size : int, optional
        Size of internal genomic segments, i.e. minimum distance between crossovers 
        (default: 1,000,000 bp).
    terminal_segment_size : int, optional
        Size of terminal genomic segments, i.e. minimum distance between crossovers and 
        ends of chromosomes (default: 50,000 bp).
    model_lambdas : float, optional
        lambdas for poisson used to represent foreground and background. If None, estimate from data.
    empty_fraction : float, optional
        Estimated fraction of empty bins in zero inflated model. If None, estimate from data.
    bc_haplotype : int, optional
        The haplotype which was used as the backcross parent. Either 0 for ref or 1 for alt (default 0)
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
    if model_type == 'haploid':
        states = [(0,), (1,)]
        estimate_parameters = estimate_haploid_parameters
    elif model_type == 'diploid_bc1':
        states = [(0, 0), (0, 1)] if bc_haplotype == 0 else [(0, 1), (1, 1)]
        estimate_parameters = estimate_diploid_bc1_parameters(bc_haplotype)
    elif model_type == 'diploid_f2':
        states = [(0, 0), (0, 1), (1, 1)]
        raise NotImplementedError('coming soon...')
    else:
        raise ValueError('rhmm_type must be one of "haploid", "backcross", or "diploid"')
    if model_lambdas is None or empty_fraction is None:
        fg_lambda, bg_lambda, empty_fraction = estimate_parameters(list(co_markers.deep_values()), rfactor)
        log.debug(
            'Estimated model parameters from data: '
            f'fg_lambda {fg_lambda:.2g}, bg_lambda {bg_lambda:.2g}, empty_fraction {empty_fraction:.2g}'
        )
    else:
        bg_lambda, fg_lambda = sorted(model_lambdas)
    rhmm = RigidHMM(
        states=states,
        rfactor=rfactor,
        term_rfactor=term_rfactor,
        trans_prob=trans_prob,
        fg_lambda=fg_lambda,
        bg_lambda=bg_lambda,
        empty_fraction=empty_fraction,
        device=device
    )
    return rhmm