from collections import namedtuple
import numpy as np
import pandas as pd

import torch

from ..records import PredictionRecords, NestedData
from snco.main.logger import progress_bar


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
        rhmm_params=NestedData(
            levels=('misc', ),
            dtype=(float, list),
            data=rhmm.params
        )
    )
    return co_preds
