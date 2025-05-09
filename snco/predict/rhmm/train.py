import logging
import numpy as np
import torch

from .model import RigidHMM
from .estimate import (
    estimate_haploid_emissions,
    estimate_diploid_emissions_ordered,
    estimate_diploid_emissions_f2
)


log = logging.getLogger('snco')
DEFAULT_DEVICE = torch.device('cpu')


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
        Initialized RigidHMM model with emission and transition parameters estimated or supplied.
    """
    bin_size = co_markers.bin_size
    rfactor = segment_size // bin_size
    term_rfactor = terminal_segment_size // bin_size
    trans_prob = cm_per_mb * (bin_size / 1e8)

    X = list(co_markers.deep_values())

    if model_type == 'haploid':
        states = [(0,), (1,)]
        estimate = estimate_haploid_emissions
    elif model_type == 'diploid_bc1':
        if bc_haplotype == 0: 
            states = [(0, 0), (0, 1)]
        else:
            states = [(0, 1), (1, 1)]
            # flip the training data so that dominant haplotype is column 0
            X = [np.flip(x, axis=1) for x in X]
        estimate = estimate_diploid_emissions_ordered
    elif model_type == 'diploid_f2':
        states = [(0, 0), (0, 1), (1, 1)]
        estimate = estimate_diploid_emissions_f2
    else:
        raise ValueError(f"Unsupported data ploidy type: {model_type}")
    if model_lambdas is None or empty_fraction is None:
        fg_lambda, bg_lambda, empty_fraction = estimate(X, rfactor)
        log.debug(
            'Estimated model parameters from data: '
            f'fg_lambda {fg_lambda:.2g}, bg_lambda {bg_lambda:.2g}, empty_fraction {empty_fraction:.2g}'
        )
    else:
        bg_lambda, fg_lambda = sorted(model_lambdas)
    return RigidHMM(
        states=states,
        rfactor=rfactor,
        term_rfactor=term_rfactor,
        trans_prob=trans_prob,
        fg_lambda=fg_lambda,
        bg_lambda=bg_lambda,
        empty_fraction=empty_fraction,
        device=device
    )
