import re
import logging
import itertools as it

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

from .utils import load_json
from .recombination import recombination_landscape
from .distortion import segregation_distortion
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


def chrom2d_subplots(chrom_sizes, figsize=(10, 10), xtick_every=1e7):

    fig, axes = plt.subplots(
        figsize=figsize,
        ncols=len(chrom_sizes),
        nrows=len(chrom_sizes),
        width_ratios=list(chrom_sizes.values()),
        height_ratios=list(chrom_sizes.values())[::-1],
        sharey='row',
        sharex='col'
    )
    axes = axes[::-1]

    for chrom, ax in zip(chrom_sizes, axes[0]):
        ax.set_xlim(0, chrom_sizes[chrom])
        xticks = np.arange(0, chrom_sizes[chrom], xtick_every)
        ax.set_xticks(xticks)
        ax.set_xticklabels([int(i // 1e6) for i in xticks])
        chrom_label = re.sub('^[Cc]hr', '', chrom)
        ax.set_xlabel(f'Chr{chrom_label} (Mb)')

    for chrom, ax in zip(chrom_sizes, axes[:, 0]):
        ax.set_ylim(0, chrom_sizes[chrom])
        yticks = np.arange(0, chrom_sizes[chrom], xtick_every)
        ax.set_yticks(yticks)
        ax.set_yticklabels([int(i // 1e6) for i in yticks])
        chrom_label = re.sub('^[Cc]hr', '', chrom)
        ax.set_ylabel(f'Chr{chrom_label} (Mb)')

    plt.tight_layout()
    return fig, axes


def chrom2dtriangle_subplots(chrom_sizes, figsize=(10, 10), xtick_every=1e7):

    fig, axes = plt.subplots(
        figsize=figsize,
        ncols=len(chrom_sizes) - 1,
        nrows=len(chrom_sizes) - 1,
        width_ratios=list(chrom_sizes.values())[:-1],
        height_ratios=list(chrom_sizes.values())[1:][::-1],
        sharey='row',
        sharex='col'
    )
    chroms = list(chrom_sizes)
    axes = axes[::-1]
    for i, j in it.product(range(len(chroms) - 1), repeat=2):
        ax = axes[i, j]
        chrom_i, chrom_j = chroms[i], chroms[j + 1]
        if i < j:
            ax.set_axis_off()
            continue
        if i == j:
            ax.tick_params(
                which='both', bottom=True, top=True, left=True, right=True,
                labeltop=False, labelbottom=True, labelright=True, labelleft=False,
            )
            ax.set_xlim(0, chrom_sizes[chrom_i])
            xticks = np.arange(0, chrom_sizes[chrom_i], xtick_every)
            ax.set_xticks(xticks)
            ax.set_xticklabels([int(i // 1e6) for i in xticks])
            chrom_i_label = re.sub('^[Cc]hr', '', chrom_i)
            ax.set_xlabel(f'Chr{chrom_i_label} (Mb)', labelpad=0.1)

            ax.set_ylim(0, chrom_sizes[chrom_j])
            yticks = np.arange(0, chrom_sizes[chrom_j], xtick_every)
            ax.set_yticks(yticks)
            ax.set_yticklabels([int(i // 1e6) for i in yticks])
            chrom_j_label = re.sub('^[Cc]hr', '', chrom_j)
            ax.set_ylabel(f'Chr{chrom_j_label} (Mb)', labelpad=0.1)
            ax.yaxis.set_label_position('right')
        else:
            ax.tick_params(
                which='both', bottom=True, top=True, left=True, right=True,
                labeltop=False, labelbottom=False, labelright=False, labelleft=False,
            )

    plt.tight_layout(pad=0)
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
                                 colour=None,
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
        line, = ax.step(x, np.nanmean(c, axis=0), color=colour)
        # when colour is None this guarantees shading and label colours are correct
        colour = line.get_color()
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


def plot_allele_ratio(co_preds, cb_whitelist=None,
                      nboots=100, ci=95,
                      axes=None,
                      figsize=(12, 4),
                      colour=None,
                      label=None,
                      rng=DEFAULT_RNG):
    if axes is None:
        fig, axes = chrom_subplots(co_preds.chrom_sizes, figsize=figsize)
    else:
        assert len(axes) == len(co_preds.chrom_sizes)

    if cb_whitelist is not None:
        co_preds = co_preds.copy()
        co_preds.filter(cb_whitelist)

    lower = (100 - ci) / 2
    upper = 100 - lower
    N = len(co_preds)

    for chrom, ax in zip(co_preds.chrom_sizes, axes):
        x = np.arange(0, co_preds.nbins[chrom]) * co_preds.bin_size
        haps = co_preds[..., chrom]
        c = []
        for _ in range(nboots):
            idx = rng.integers(0, N, size=N)
            c.append(haps[idx].mean(axis=0))
        line, = ax.step(x, np.nanmean(c, axis=0), color=colour)
        colour = line.get_color()
        ax.fill_between(
            x=x,
            y1=np.nanpercentile(c, lower, axis=0),
            y2=np.nanpercentile(c, upper, axis=0),
            alpha=0.25,
            color=colour
        )
    axes[0].set_ylabel('Allele ratio')
    axes[-1].plot([], [], color=colour, label=label)
    axes[0].set_ylim(0, 1)
    if label is not None:
        axes[-1].legend()
    plt.tight_layout()
    return axes


def _plot_segdist_1d(seg_dist, chrom_sizes, fig, axes, colour=None, label=None):
    for chrom, ax in zip(chrom_sizes, axes):
        chrom_sd = seg_dist.query('chrom_1 == @chrom')
        ax.step(chrom_sd.pos_1.values, chrom_sd.lod_score.values, color=colour)
        colour = line.get_color()
    axes[0].set_ylabel('LOD score')
    axes[-1].plot([], [], color=colour, label=label)
    if label is not None:
        axes[-1].legend()
    plt.tight_layout()
    return axes


def _plot_segdist_2d(seg_dist, chrom_sizes, fig, axes, cmap='Blues',
                     vmin=None, vmax=None, label=None):
    chroms = list(chrom_sizes)
    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = seg_dist.lod_score.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    for i, j in it.product(range(len(chroms) - 1), repeat=2):
        ax = axes[i, j]
        if i < j:
            continue
        chrom_1, chrom_2 = chroms[j], chroms[i + 1]
        chrom_sd = seg_dist.query('chrom_1 == @chrom_1 & chrom_2 == @chrom_2')
        chrom_sd = chrom_sd.pivot(columns='pos_1', index='pos_2', values='lod_score')
        im = ax.imshow(
            chrom_sd.values,
            norm=norm,
            extent=(0, chrom_sizes[chrom_1], chrom_sizes[chrom_2], 0),
            cmap=cmap
        )
    plt.colorbar(im, ax=axes[0, 1:], label='LOD score', orientation='horizontal')
    if label is not None:
        fig.suptitle(label)
    plt.tight_layout()
    return axes


def plot_segregation_distortion(co_preds, cb_whitelist=None,
                                order=1, resolution=250_000, processes=1,
                                axes=None, figwidth=12, figheight=4,
                                colour=None, cmap='Blues',
                                vmin=None, vmax=None, label=None):

    if order not in (1, 2):
        raise ValueError('Can only generate plots for distortions of order 1 or 2')

    seg_dist = segregation_distortion(
        co_preds, order=order, resolution=resolution, processes=processes
    )

    if axes is None:
        if order == 1:
            fig, axes = chrom_subplots(co_preds.chrom_sizes, figsize=(figwidth, figheight))
        elif order == 2:
            fig, axes = chrom2dtriangle_subplots(co_preds.chrom_sizes, figsize=(figwidth, figwidth))
    else:
        assert len(axes) == len(co_preds.chrom_sizes)

    if order == 1:
        _plot_segdist_1d(seg_dist, co_preds.chrom_sizes, fig, axes,
                         colour=colour, label=label)
    elif order == 2:
        _plot_segdist_2d(seg_dist, co_preds.chrom_sizes, fig, axes,
                         cmap=cmap, vmin=vmin, vmax=vmax, label=label)

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