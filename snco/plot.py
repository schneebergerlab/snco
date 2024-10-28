import re

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

from .utils import load_json

YLIM_OFFSET = 1
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


def single_cell_markerplot(cb, co_markers, *, co_preds=None, figsize=(18, 4),
                           show_mesh_prob=True, annotate_co_number=True,
                           max_yheight=20, ref_colour='#0072b2', alt_colour='#d55e00'):

    fig, axes = chrom_subplots(co_markers.chrom_sizes, figsize=figsize)
    axes[0].set_ylabel('Marker coverage')
    cmap = LinearSegmentedColormap.from_list('hap_cmap', [ref_colour, alt_colour])

    for chrom, ax in zip(co_markers.chrom_sizes, axes):

        ax.set_xlim(-XLIM_OFFSET, co_markers.chrom_sizes[chrom] + XLIM_OFFSET)
        ylims = np.array([-max_yheight - YLIM_OFFSET, max_yheight + YLIM_OFFSET])
        ax.set_ylim(*ylims)

        chrom_markerplot(
            co_markers[cb, chrom],
            co_markers.chrom_sizes[chrom],
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
                n_co = np.abs(np.diff(hp)).sum()
                ax.annotate(text=f'{n_co.sum():.2f} COs', xy=(0.05, 0.05), xycoords='axes fraction')
    plt.tight_layout()
    return fig, axes


def run_plot(cell_barcode, marker_json_fn, plot_json_fn,
             output_fig_fn, figsize=(18, 4),
             show_pred=True, show_co_num=True,
             max_yheight=20,
             ref_colour='#0072b2', alt_colour='#d55e00'):
    co_markers = load_json(marker_json_fn, cb_whitelist_fn=None, bin_size=None)
    if plot_json_fn is not None:
        co_preds = load_json(
            plot_json_fn, cb_whitelist_fn=None, bin_size=None, data_type='predictions'
        )
    else:
        co_preds = None

    single_cell_markerplot(
        cell_barcode,
        co_markers,
        co_preds=co_preds,
        figsize=figsize,
        show_mesh_prob=show_pred,
        annotate_co_number=show_co_num,
        max_yheight=max_yheight,
        ref_colour=ref_colour,
        alt_colour=alt_colour
    )
    plt.savefig(output_fig_fn)
    