import os
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from snco.utils import load_json
from snco.records import MarkerRecords, PredictionRecords
from snco.clean.background import (
    estimate_overall_background_signal, subtract_background, random_bg_sample
)
from snco.opts import DEFAULT_RANDOM_SEED


log = logging.getLogger('snco')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def co_invs_to_gt(co_invs, bin_size, chrom_nbins):
    """
    Convert intervals from a BED file representing haplotypes into a binary NumPy array.

    Parameters
    ----------
    co_invs : DataFrame
        A DataFrame containing columns ['start', 'end', 'haplo'] representing haplotype intervals.
    bin_size : int
        Size of genomic bins.
    chrom_nbins : int
        Number of bins for the chromosome.

    Returns
    -------
    gt : ndarray
        Binary NumPy array representing haplotype values per bin.

    Raises
    ------
    ValueError
        If intervals do not completely cover all chromosomes.
    """
    gt = np.full(chrom_nbins, np.nan)

    start_bin = np.ceil(co_invs.start.values / bin_size).astype(int)
    end_bin = np.ceil(co_invs.end.values / bin_size).astype(int)
    haplo = co_invs.haplo.values

    for s, e, h in zip(start_bin, end_bin, haplo):
        gt[s: e] = h
    if np.isnan(gt).any():
        raise ValueError('Supplied intervals in haplo-bed-fn do not completely cover chromosomes')
    return gt


def read_ground_truth_haplotypes_bed(co_invs_fn, chrom_sizes, bin_size=25_000):
    """
    Read a BED file containing haplotype intervals and convert to binned binary arrays.

    Parameters
    ----------
    co_invs_fn : str
        Path to BED file.
    chrom_sizes : dict
        Dictionary mapping chromosome names to their lengths.
    bin_size : int, optional
        Size of genomic bins. Default is 25,000.

    Returns
    -------
    gt : PredictionRecords
        PredictionRecords object with ground truth haplotype data.
    """
    co_invs = pd.read_csv(
        co_invs_fn,
        sep='\t',
        names=['chrom', 'start', 'end', 'sample_id', 'haplo', 'strand']
    )

    gt = PredictionRecords(chrom_sizes, bin_size, set(co_invs.sample_id))

    for sample_id, sample_invs in co_invs.groupby('sample_id'):
        for chrom, n in gt.nbins.items():
            chrom_invs = sample_invs.query('chrom == @chrom')
            gt[sample_id, chrom] = co_invs_to_gt(chrom_invs, bin_size, n)
    return gt


def read_ground_truth_haplotypes_json(pred_json_fn):
    """
    Read ground truth haplotypes from a JSON file.

    Parameters
    ----------
    pred_json_fn : str
        Path to the JSON file.

    Returns
    -------
    gt : PredictionRecords
        Ground truth haplotypes with haplotype probabilities rounded to integers.
    """
    gt = PredictionRecords.read_json(pred_json_fn)
    for *_, m in gt.deep_items():
        np.round(m, decimals=0, out=m)
    return gt


def apply_gt_to_markers(gt, m, bg_rate, bg_signal, rng=DEFAULT_RNG):
    """
    Simulate marker counts using a known ground truth haplotype.

    Parameters
    ----------
    gt : ndarray
        Ground truth haplotype (binary array).
    m : ndarray
        Input marker array for a single chromosome.
    bg_rate : float
        Estimated background rate for the cell.
    bg_signal : ndarray
        Per-bin background signal for the chromosome.
    rng : Generator, optional
        NumPy random generator.

    Returns
    -------
    sim : ndarray
        Simulated marker array with shape (n_bins, 2).
    """
    gt = gt.astype(int)
    s = len(m)

    # simulate a realistic background signal using the average background across the dataset
    fg, bg = subtract_background(m, bg_signal, bg_rate, return_bg=True, rng=rng)

    # flatten haplotypes
    fg = fg.sum(axis=1)
    bg = bg.sum(axis=1)

    sim = np.zeros(shape=(s, 2))
    idx = np.arange(s)

    # apply fg at ground truth haplotype, and bg on other haplotype
    sim[idx, gt] = fg
    sim[idx, 1 - gt] = bg

    return sim


def simulate_singlets(co_markers, ground_truth, bg_signal, frac_bg, nsim_per_sample, rng=DEFAULT_RNG):
    """
    Simulate single-cell barcodes based on ground truth haplotypes.

    Parameters
    ----------
    co_markers : MarkerRecords
        Haplotype-specific markers from a real dataset, to use as basis for simulation.
    ground_truth : PredictionRecords
        Ground truth haplotype calls from a different dataset, to be simulated.
    bg_signal : dict
        Per-bin background signal per chromosome.
    frac_bg : dict
        Background fraction per cell barcode.
    nsim_per_sample : int
        Number of simulated cells per ground truth haplotype.
    rng : Generator, optional
        NumPy random generator.

    Returns
    -------
    sim_co_markers : MarkerRecords
        Simulated haplotype-specific marker records.
    """
    sim_co_markers = MarkerRecords.new_like(co_markers)
    sim_co_markers.metadata['ground_truth'] = PredictionRecords.new_like(co_markers)
    for sample_id in ground_truth.barcodes:
        cbs_to_sim = rng.choice(co_markers.barcodes, replace=False, size=nsim_per_sample)
        for cb in cbs_to_sim:
            sim_id = f'{sample_id}:{cb}'
            for chrom in ground_truth.chrom_sizes:
                gt = ground_truth[sample_id, chrom]
                sim_co_markers[sim_id, chrom] = apply_gt_to_markers(
                    gt, co_markers[cb, chrom], frac_bg[cb], bg_signal[chrom], rng=rng
                )
                sim_co_markers.metadata['ground_truth'][sim_id, chrom] = gt
    return sim_co_markers


def simulate_doublets(co_markers, n_doublets, rng=DEFAULT_RNG):
    """
    Simulate doublet barcodes by summing markers from random barcode pairs.

    Parameters
    ----------
    co_markers : MarkerRecords
        Haplotype-specific markers from a real dataset, to use as basis for simulation.
    n_doublets : int
        Number of doublet barcodes to simulate.
    rng : Generator, optional
        NumPy random generator.

    Returns
    -------
    sim_co_markers_doublets : MarkerRecords
        Simulated haplotype-specific marker records for doublets.
    """
    sim_co_markers_doublets = MarkerRecords.new_like(co_markers)
    barcodes = rng.choice(co_markers.barcodes, size=n_doublets * 2, replace=True)
    # sorting by total markers makes m_i and m_j relatively similar in size
    barcodes = sorted(barcodes, key=co_markers.total_marker_count)
    for cb_i, cb_j in zip(barcodes[0::2], barcodes[1::2]):
        sim_id = f'doublet:{cb_i}_{cb_j}'
        for chrom in sim_co_markers_doublets.chrom_sizes:
            m_i = co_markers[cb_i, chrom]
            m_j = co_markers[cb_j, chrom]
            sim_co_markers_doublets[sim_id, chrom] = m_i + m_j
    return sim_co_markers_doublets


def generate_simulated_data(co_markers, ground_truth, conv_window_size=2_500_000,
                            bg_rate=None, nsim_per_sample=100,
                            doublet_rate=0.0, rng=DEFAULT_RNG):
    """
    Generate simulated crossover marker data using real background and ground truth haplotypes.

    Parameters
    ----------
    co_markers : MarkerRecords
        Haplotype-specific markers from a real dataset, to use as basis for simulation.
    ground_truth : PredictionRecords
        Ground truth haplotype calls from a different dataset, to be simulated.
    conv_window_size : int, optional
        Window size for convolution-based background estimation.
    bg_rate : float, optional
        Fixed background rate (override background rate estimation from data).
    nsim_per_sample : int, optional
        Number of simulated cells per ground truth haplotype.
    doublet_rate : float, optional
        Fraction or number of doublets to simulate.
    rng : Generator, optional
        NumPy random generator.

    Returns
    -------
    sim_co_markers : MarkerRecords
        Simulated haplotype-specific marker records, including simulated singlet and doublet barcodes
    """
    co_markers = estimate_overall_background_signal(
        co_markers,
        conv_window_size,
        max_frac_bg=1.0,
        apply_per_geno=False # todo, should this be exposed?
    )
    bg_signal = co_markers.metadata['background_signal']['ungrouped'] # bit of a hack
    frac_bg = co_markers.metadata['estimated_background_fraction']
    if bg_rate is not None:
        frac_bg = defaultdict(lambda: bg_rate)

    sim_co_markers = simulate_singlets(
        co_markers, ground_truth, bg_signal, frac_bg, nsim_per_sample, rng=rng
    )

    if doublet_rate:
        doublet_rate = int(len(sim_co_markers) * doublet_rate) if doublet_rate < 1 else int(doublet_rate)
        log.info(f'Simulating {doublet_rate} doublet barcodes')
        sim_co_markers.merge(
            simulate_doublets(co_markers, doublet_rate, rng=rng),
            inplace=True
        )
    return sim_co_markers


def ground_truth_from_marker_records(co_markers):
    """
    Extract ground truth haplotype calls from marker record metadata.

    Parameters
    ----------
    co_markers : MarkerRecords
        Simulated marker records containing ground truth metadata.

    Returns
    -------
    ground_truth : PredictionRecords
        Extracted ground truth haplotypes.
    """
    ground_truth = PredictionRecords.new_like(co_markers)
    for cb, sd in co_markers.metadata['ground_truth'].items():
        for chrom, arr in sd.items():
            ground_truth[cb, chrom] = np.array(arr)
    return ground_truth


def run_sim(marker_json_fn, output_json_fn, ground_truth_fn, *,
            cb_whitelist_fn=None, bin_size=25_000, bg_marker_rate=None,
            bg_window_size=2_500_000, nsim_per_sample=100, n_doublets=0.0,
            rng=DEFAULT_RNG):
    """
    Run the full simulation pipeline to create synthetic marker data from ground truth.

    Parameters
    ----------
    marker_json_fn : str
        Path to marker JSON file.
    output_json_fn : str
        Output path for simulated JSON file.
    ground_truth_fn : str
        Path to ground truth file (BED or JSON).
    cb_whitelist_fn : str, optional
        Optional path to cell barcode whitelist.
    bin_size : int, optional
        Genomic bin size.
    bg_marker_rate : float, optional
        Fixed background rate.
    bg_window_size : int, optional
        Window size for convolution-based background estimation.
    nsim_per_sample : int, optional
        Number of simulations per ground truth haplotype.
    n_doublets : float, optional
        Number or fraction of doublets to simulate.
    rng : Generator, optional
        NumPy random generator.

    Returns
    -------
    sim_co_markers : MarkerRecords
        Simulated marker data.
    """
    co_markers = load_json(marker_json_fn, cb_whitelist_fn, bin_size)
    if os.path.splitext(ground_truth_fn)[1] == '.bed':
        ground_truth_haplotypes = read_ground_truth_haplotypes_bed(
            ground_truth_fn, co_markers.chrom_sizes, bin_size
        )
    else:
        ground_truth_haplotypes = read_ground_truth_haplotypes_json(ground_truth_fn)
    log.info(f'Read {len(ground_truth_haplotypes)} ground truth samples from {ground_truth_fn}')
    sim_co_markers = generate_simulated_data(
        co_markers,
        ground_truth_haplotypes,
        bg_rate=bg_marker_rate,
        conv_window_size=bg_window_size,
        nsim_per_sample=nsim_per_sample,
        doublet_rate=n_doublets,
        rng=rng
    )
    log.info(f'Simulated {len(sim_co_markers)} barcodes total')
    if output_json_fn is not None:
        log.info(f'Writing markers to {output_json_fn}')
        sim_co_markers.write_json(output_json_fn)
    return sim_co_markers
