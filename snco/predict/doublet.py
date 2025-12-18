import logging

import numpy as np
import pandas as pd

import torch
from pomegranate.gmm import GeneralMixtureModel
from pomegranate import distributions as pmd

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


def detect_doublets(co_markers, co_preds, rhmm, n_doublets=1000,
                    freeze_doublet_dist=True, inertia=0.9,
                    batch_size=128, processes=1, rng=DEFAULT_RNG):

    if n_doublets > 1:
        n_sim = int(n_doublets)
    else:
        if not n_doublets:
            raise ValueError('n-doublets must be >0 for doublet detection')
        n_sim = int(len(co_markers) * n_doublets)

    log.info(f'Simulating {n_sim} doublets')
    sim_co_markers = simulate_doublets(co_markers, n_sim, rng=rng)
    sim_co_preds = detect_crossovers(
        sim_co_markers, rhmm,
        sample_paths=False,
        batch_size=batch_size,
        processes=processes,
    )

    barcodes = co_preds.barcodes
    log_acc_score_real = np.log(np.array([
        stats.accuracy_score(co_markers[cb], co_preds[cb])
        for cb in barcodes
    ]))

    log_acc_score_sim = np.log(np.array([
        stats.accuracy_score(sim_co_markers[cb], sim_co_preds[cb])
        for cb in sim_co_markers.barcodes
    ]))

    if log_acc_score_real.mean() < log_acc_score_sim.mean():
        log.warning('Real data looks more doublet-y than simulated data! '
                    'Doublet predictions may not be accurate')

    doublet_dist = pmd.Normal(
        [log_acc_score_sim.mean()],
        [log_acc_score_sim.var()],
        covariance_type='diag',
        frozen=freeze_doublet_dist
    )

    nondoublet_dist = pmd.Normal(
        [log_acc_score_real.mean()],
        [log_acc_score_real.var()],
        covariance_type='diag',
    )

    gmm = GeneralMixtureModel([nondoublet_dist, doublet_dist], inertia=inertia)
    gmm = gmm.fit(log_acc_score_real.reshape(-1, 1))

    doublet_probs = gmm.predict_proba(log_acc_score_real.reshape(-1, 1)).numpy()[:, 1]
    co_preds.add_metadata(
        doublet_probability=NestedData(
            levels=('cb',), dtype=float, data=dict(zip(barcodes, doublet_probs.tolist()))
        )
    )
    return co_preds
