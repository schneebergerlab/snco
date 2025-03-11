import re
import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

from .utils import load_json
from .recombination import recombination_landscape
from .opts import DEFAULT_RANDOM_SEED


log = logging.getLogger('snco')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)
XLIM_OFFSET = 1e4


def chrom_subplots(chrom_sizes, figsize=(18, 5), xtick_every=1e7,
                   span_features=None, span_kwargs=None):

    fig, axes = plt.subplots(
        figsize=figsize,
        ncols=len(chrom_sizes),
        width_ratios=list(chrom_sizes.values()),
        sharey='row',
        sharex='col'
    )

    for chrom, ax in zip(chrom_sizes, axes):
        ax.set_xlim(0, chrom_sizes[chrom])
        xticks = np.arange(0, chrom_sizes[chrom], xtick_every)
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(i // 1e6) for i in xticks])
        chrom_label = re.sub('^[Cc]hr', '', chrom)
        ax.set_xlabel(f'Chromosome {chrom_label} (Mb)')

    if span_features is not None:

        default_span_kwargs = {'alpha': 0.3, 'color': '#252525'}
        if span_kwargs is not None:
            span_kwargs.setdefault(default_span_kwargs)
        else:
            span_kwargs = default_span_kwargs

        for chrom, ax in zip(chrom_sizes, axes):
            for s, e in span_features.get(chrom, []):
                ax.axvspan(s, e, **span_kwargs)

    plt.tight_layout()
    return fig, axes


def chrom_markerplot(co_markers, chrom_size, bin_size, ax=None, max_yheight=20,
                     ref_colour='#0072b2', alt_colour='#d55e00', ori_colour='#252525'):

    ref_markers = co_markers[:, 0].copy()
    ref_markers[ref_markers > max_yheight] = max_yheight
    alt_markers = co_markers[:, 1].copy()
    alt_markers[alt_markers > max_yheight] = max_yheight
    alt_markers = np.negative(alt_markers)

    nbins = len(co_markers)
    pos = np.arange(nbins) * bin_size
    base = np.repeat(0, nbins)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.vlines(pos, base, ref_markers, color=ref_colour, zorder=0)
    ax.vlines(pos, base, alt_markers, color=alt_colour, zorder=0)
    ax.plot([0, chrom_size], [0, 0], ls='-', color=ori_colour)
    return ax


def _add_co_prob_colormesh(ax, hp, chrom_size, bin_size, ylims, cmap):
    nbins = len(hp)
    xpos = np.insert(np.arange(nbins) * bin_size, nbins, chrom_size)
    ax.pcolormesh(
        xpos,
        ylims,
        hp.reshape(1, -1),
        cmap=cmap,
        norm=Normalize(0, 1),
        alpha=0.5,
        zorder=-2,
        rasterized=True
    )
    return ax


def _add_gt_vlines(ax, gt, bin_size, ylims, colour='#eeeeee'):
    gt_pos = (np.where(np.diff(gt))[0] + 1) * bin_size
    if len(gt_pos):
        ax.vlines(
            gt_pos,
            np.repeat(ylims[0], len(gt_pos)),
            np.repeat(ylims[1], len(gt_pos)),
            ls='--',
            colors=colour,
            zorder=-1,
        )


def single_cell_markerplot(cb, co_markers, *, co_preds=None, figsize=(18, 4), chroms=None,
                           show_mesh_prob=True, annotate_co_number=True,
                           nco_min_prob_change=5e-3, show_gt=True,
                           max_yheight='auto', ref_colour='#0072b2', alt_colour='#d55e00'):

    if cb not in co_markers.barcodes:
        raise KeyError(f'cb {cb} not in co_marker object')

    chrom_sizes = co_markers.chrom_sizes
    if chroms is not None:
        chrom_sizes = {c: chrom_sizes[c] for c in chroms}
    fig, axes = chrom_subplots(chrom_sizes, figsize=figsize)
    try:
        hap1, hap2 = co_markers.metadata['genotypes'][cb]['genotype']
    except KeyError:
        hap1, hap2 = 'hap1', 'hap2'
    axes[0].set_ylabel(f'Marker coverage ({hap1} vs {hap2})')
    cmap = LinearSegmentedColormap.from_list('hap_cmap', [ref_colour, alt_colour])

    if show_gt:
        try:
            gt = co_markers.metadata['ground_truth'][cb]
        except KeyError:
            show_gt = False
            gt = None

    if max_yheight == 'auto':
        m = np.concatenate([co_markers[cb, chrom].ravel() for chrom in co_markers.chrom_sizes])
        max_yheight = np.percentile(m, 99.5)
    ylim_offset = max_yheight * 0.05

    for chrom, ax in zip(chrom_sizes, axes):

        ax.set_xlim(-XLIM_OFFSET, chrom_sizes[chrom] + XLIM_OFFSET)
        ylims = np.array([-max_yheight - ylim_offset, max_yheight + ylim_offset])
        ax.set_ylim(*ylims)

        chrom_markerplot(
            co_markers[cb, chrom],
            chrom_sizes[chrom],
            co_markers.bin_size,
            ax=ax,
            max_yheight=max_yheight,
            ref_colour=ref_colour,
            alt_colour=alt_colour,
        )
        if co_preds is not None:
            hp = co_preds[cb, chrom]
            if show_mesh_prob:
                _add_co_prob_colormesh(
                    ax, hp, co_markers.chrom_sizes[chrom], co_markers.bin_size, ylims, cmap
                )
            if annotate_co_number:
                p_co = np.abs(np.diff(hp))
                p_co = np.where(p_co > nco_min_prob_change, p_co, 0)
                n_co = p_co.sum()
                ax.annotate(text=f'{n_co.sum():.2f} COs', xy=(0.05, 0.05), xycoords='axes fraction')
        if show_gt:
            _add_gt_vlines(
                ax, gt[chrom], co_markers.bin_size, ylims
            )
    plt.tight_layout()
    return fig, axes


def plot_recombination_landscape(co_preds, co_markers=None,
                                 cb_whitelist=None,
                                 rolling_mean_window_size=1_000_000,
                                 nboots=100, ci=95,
                                 min_prob=0.01,
                                 axes=None,
                                 figsize=(12, 4),
                                 colour="#0072b2",
                                 label=None,
                                 rng=DEFAULT_RNG):

    if axes is None:
        fig, axes = chrom_subplots(co_preds.chrom_sizes, figsize=figsize)
    else:
        assert len(axes) == len(co_preds.chrom_sizes)

    if cb_whitelist is not None:
        co_preds = co_preds.copy()
        co_preds.filter(cb_whitelist)
        if co_markers is not None:
            co_markers = co_markers.copy()
            co_markers.filter(cb_whitelist)

    cm_per_mb = recombination_landscape(
        co_preds,
        co_markers=co_markers,
        rolling_mean_window_size=rolling_mean_window_size,
        nboots=nboots,
        min_prob=min_prob,
        rng=rng
    )

    lower = (100 - ci) / 2
    upper = 100 - lower

    for chrom, ax in zip(cm_per_mb, axes):
        x = np.arange(0, co_preds.nbins[chrom]) * co_preds.bin_size
        c = cm_per_mb[chrom]
        ax.step(x, np.nanmean(c, axis=0), color=colour)
        ax.fill_between(
            x=x,
            y1=np.nanpercentile(c, lower, axis=0),
            y2=np.nanpercentile(c, upper, axis=0),
            alpha=0.25,
            color=colour
        )
    axes[0].set_ylabel('cM / Mb')
    axes[-1].plot([], [], color=colour, label=label)
    if label is not None:
        axes[-1].legend()
    plt.tight_layout()
    return axes


def run_plot(cell_barcode, marker_json_fn, pred_json_fn, output_fig_fn=None,
             cb_whitelist_fn=None, plot_type='markerplot', figsize=(18, 4), display_plot=False,
             show_pred=True, show_co_num=True, show_gt=True, max_yheight=20,
             window_size=1_000_000, nboots=100, confidence_intervals=95,
             ref_colour='#0072b2', alt_colour='#d55e00', rng=DEFAULT_RNG):
    co_markers = load_json(marker_json_fn, cb_whitelist_fn=cb_whitelist_fn, bin_size=None)
    if pred_json_fn is not None:
        co_preds = load_json(
            pred_json_fn, cb_whitelist_fn=None, bin_size=None, data_type='predictions'
        )
    else:
        co_preds = None

    if plot_type == 'markerplot':
        if cell_barcode is None:
            raise ValueError('Must specify a cell barcode for plot-type "markerplot"')
        single_cell_markerplot(
            cell_barcode,
            co_markers,
            co_preds=co_preds,
            figsize=figsize,
            show_mesh_prob=show_pred,
            annotate_co_number=show_co_num,
            nco_min_prob_change=nco_min_prob_change,
            show_gt=show_gt,
            max_yheight=max_yheight,
            ref_colour=ref_colour,
            alt_colour=alt_colour
        )
    elif plot_type == 'recombination':
        plot_recombination_landscape(
            co_preds, co_markers,
            rolling_mean_window_size=window_size,
            nboots=nboots, ci=confidence_intervals,
            min_prob=nco_min_prob_change,
            figsize=figsize,
            colour=ref_colour,
            rng=rng
        )
    if output_fig_fn is not None:
        plt.savefig(output_fig_fn)
    if display_plot:
        #plt.switch_backend('TkAgg')
        plt.show()