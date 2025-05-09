import re
import itertools as it

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import seaborn as sns

from snco.plot import chrom_subplots, chrom2d_subplots

DEFAULT_PALETTE = sns.color_palette(
    ['#0072b2', '#d55e00', '#009e73', '#f0e442', '#cc79a7', '#56b4e9', '#e69f00']
)


def eqtl_scatterplot(eqtl_peaks, chrom_sizes,
                     hue_variable='eqtl_type', hue_order=None,
                     fig_size=(10, 10),
                     point_sizes=(5, 50),
                     point_size_lod_norm=(5, 50),
                     hue_palette=DEFAULT_PALETTE,
                     errorbar_colour='#777777', cis_shade_colour='#f0e442',
                     cis_range=5e5):
    chroms = list(chrom_sizes)
    fig, axes = chrom2d_subplots(chrom_sizes, fig_size)
    if hue_order is None:
        hue_order = sorted(eqtl_peaks[hue_variable].unique())
    for i, j in it.product(range(len(chrom_sizes)), repeat=2):
        c_i, c_j = chroms[i], chroms[j]
        ax_peaks = eqtl_peaks.query('chrom_gene == @c_i & chrom_eqtl == @c_j')
        sns.scatterplot(
            x='pos_eqtl',
            y='pos_gene',
            hue=hue_variable,
            size='lod_score',
            data=ax_peaks,
            sizes=point_sizes,
            size_norm=point_size_lod_norm,
            hue_order=hue_order,
            palette=hue_palette[:len(hue_order)],
            ax=axes[i][j]
        )
        axes[i][j].errorbar(
            x=ax_peaks.pos_eqtl,
            y=ax_peaks.pos_gene,
            xerr=[ax_peaks.pos_eqtl - ax_peaks.left_ci_eqtl,
                  ax_peaks.right_ci_eqtl - ax_peaks.pos_eqtl],
            color=errorbar_colour,
            zorder=0,
            linestyle='None',
            marker='None',
            alpha=0.75
        )
        try:
            axes[i][j].legend_.remove()
        except:
            pass

        if i == j:
            cs_i = chrom_sizes[c_i]
            cs_j = chrom_sizes[c_j]
            axes[i][j].fill_between(
                x=[0, cis_range, cs_j - cis_range, cs_j],
                y1=[0, 0, cs_i - cis_range * 2, cs_i - cis_range],
                y2=[cis_range, cis_range * 2, cs_i, cs_i],
                color=cis_shade_colour,
                alpha=0.5,
                zorder=-1
            )

    fig.supxlabel('eQTL position')
    fig.supylabel('Gene position')
    plt.tight_layout()
    return fig, axes


def label_transform(label, coltype):
    if label == coltype:
        return 'All'
    else:
        return label[:-(len(coltype) + 1)].capitalize()


class eQTLPlotter:

    def __init__(self, eqtl_results, chrom_sizes, eqtl_peaks=None, gene_locs=None,
                 coltype='lod_score', usecols=None, labels=None,
                 palette=DEFAULT_PALETTE, figsize=(12.5, 3)):
        self.eqtl_results = eqtl_results
        self.chrom_sizes = chrom_sizes
        self.eqtl_peaks = eqtl_peaks
        self.gene_locs = gene_locs
        if coltype == 'lod_score':
            self._col_transform = lambda lod: lod
            self.ylabel = 'LOD score'
        elif coltype == 'pval':
            self._col_transform = lambda pval: -np.log10(pval)
            self.ylabel = 'Negative log10 FDR'
        else:
            raise ValueError('coltype must be either lod_score or pval')
        if usecols is None:
            self.plot_columns = [col for col in eqtl_results.columns if col.endswith(coltype)]
        else:
            self.plot_columns = usecols
        if labels is None:
            self.labels = [label_transform(col, coltype) for col in self.plot_columns]
        elif isinstance(labels, dict):
            self.labels = [labels[col] for col in self.plot_columns]
        else:
            self.labels = labels
        self._label_mapper = {col: lab for col, lab in zip(self.plot_columns, self.labels)}
        self.palette = palette[:len(self.plot_columns)]
        self.figsize = figsize        

    def __call__(self, gene_id):
        gene_eqtl_results = self.eqtl_results.query('gene_id == @gene_id')
        gene_eqtl_results = gene_eqtl_results.melt(
            id_vars=['chrom', 'pos'],
            value_vars=self.plot_columns,
            var_name='hue',
            value_name='y'
        )
        if not len(gene_eqtl_results):
            raise ValueError('Gene was not tested')
        gene_eqtl_results = gene_eqtl_results.assign(
            hue=gene_eqtl_results.hue.map(self._label_mapper),
            y=gene_eqtl_results.y.apply(self._col_transform)
        )
        if self.gene_locs is not None:
            try:
                gene_chrom, gene_pos = self.gene_locs.loc[gene_id]
            except KeyError:
                gene_chrom, gene_pos = None
        else:
            gene_chrom, gene_pos = None, None
        fig, axes = chrom_subplots(self.chrom_sizes, figsize=self.figsize)
        for (chrom, chrom_eqtl_results), ax in zip(gene_eqtl_results.groupby('chrom'), axes):
            sns.lineplot(
                x='pos',
                y='y',
                hue='hue',
                data=chrom_eqtl_results,
                drawstyle='steps-post',
                ax=ax,
                hue_order=self.labels,
                palette=self.palette,
                estimator=None,
            )
            if chrom == gene_chrom:
                ax.axvline(gene_pos, ls='-', color='#252525', zorder=-1)
            if self.eqtl_peaks is not None:
                for _, se in self.eqtl_peaks.query('gene_id == @gene_id & chrom_eqtl == @chrom').iterrows():
                    ax.axvline(se.pos_eqtl, ls='--', color='#777777', zorder=-1)
                    ax.axvspan(se.left_ci_eqtl, se.right_ci_eqtl, color='#777777', zorder=-1, alpha=0.2)

        axes[0].set_ylabel(self.ylabel)
        for ax in axes[:-1]:
            ax.legend_.remove()
        axes[-1].legend_.set_title('')

        plt.tight_layout(pad=0.2)
        return axes


class HaplotypeExpressionPlotter:

    def __init__(self, haplotypes, exprs_mat, eqtls=None, gene_locs=None,
                 parental_genotypes=None, celltypes=None, cb_whitelist=None,
                 palette=DEFAULT_PALETTE, figsize=(12.5, 3)):
        if cb_whitelist is None:
            self.cb_whitelist = list(
                set(haplotypes.barcodes).intersection(exprs_mat.columns)
            )
            if not len(self.cb_whitelist):
                raise ValueError('haplotypes and exprs_mat barcodes do not overlap')
        self.haplotypes = haplotypes.filter(self.cb_whitelist, inplace=False)
        self.exprs_mat = exprs_mat.loc[:, self.cb_whitelist]
        self.eqtls = eqtls
        self.gene_locs = gene_locs
        if parental_genotypes is not None:
            parental_genotypes = parental_genotypes.loc[self.cb_whitelist].rename('Genotype')
        self.parental_genotypes = parental_genotypes
        if celltypes is not None:
            celltypes = celltypes.loc[self.cb_whitelist].rename('Cell/Nucleus type')
        self.celltypes = celltypes
        self.palette = palette
        self.figsize = figsize

    def get_haplotype(self, gene_id, haplotype_type, haplotype_chrom=None, haplotype_pos=None):
        if haplotype_type == 'cis':
            gene_loc = self.gene_locs.loc[gene_id]
            haplotype = self.haplotypes.get_haplotype(gene_loc.chrom, gene_loc.pos, self.cb_whitelist)
            label = 'cis-haplotype'
        elif haplotype_type == 'best_hit':
            if self.eqtls is None:
                raise ValueError('eqtls were not provided')
            gene_eqtls = self.eqtls.query('gene_id == @gene_id')
            if not len(gene_eqtls):
                raise ValueError('cannot find best_hit, gene was not tested')
            best_hit = gene_eqtls.loc[gene_eqtls.lod_score.idxmax()]
            haplotype = self.haplotypes.get_haplotype(best_hit.chrom, best_hit.pos, self.cb_whitelist)
            label= f'Haplotype at {best_hit.chrom}:{best_hit.pos}'
        elif haplotype_type == 'specified':
            if haplotype_chrom is None:
                raise ValueError('If haplotype_type == "specified", at least haplotype_chrom must be supplied')
            if haplotype_pos is not None:
                haplotype_pos = int(haplotype_pos)
                haplotype = self.haplotypes.get_haplotype(haplotype_chrom, haplotype_pos, self.cb_whitelist)
            else:
                if self.eqtls is None:
                    raise ValueError('eqtls were not provided')
                gene_eqtls = self.eqtls.query('gene_id == @gene_id & chrom == @haplotype_chrom')
                if not len(gene_eqtls):
                    raise ValueError('cannot find best_hit, gene was not tested')
                best_hit = gene_eqtls.loc[gene_eqtls.lod_score.idxmax()]
                haplotype_pos = best_hit.pos
                haplotype = self.haplotypes.get_haplotype(haplotype_chrom, haplotype_pos, self.cb_whitelist)
            label= f'Haplotype at {haplotype_chrom}:{haplotype_pos}'
        haplotype = haplotype.round().map({0: 'Parent 1', 1: 'Parent 2'})
        return haplotype.rename(label)

    def _assign_variable(self, vartype, gene_id, haplotype_subtype, haplotype_chrom, haplotype_pos):
        if vartype == 'haplotype':
            variable = self.get_haplotype(gene_id, haplotype_subtype, haplotype_chrom, haplotype_pos)
        elif vartype == 'genotype':
            if self.parental_genotypes is None:
                raise ValueError('parental_genotypes were not provided')
            variable = self.parental_genotypes
        elif vartype == 'celltype':
            if self.celltypes is None:
                raise ValueError('celltypes were not provided')
            variable = self.celltypes
        elif vartype is None:
            variable = None
        else:
            raise NotImplementedError(f'Variables x and hue should be one of "haplotype", "genotype" or "celltype"')
        return variable

    def __call__(self, gene_id, x='haplotype', hue=None,
                 x_haplotype_subtype='cis', x_haplotype_chrom=None, x_haplotype_pos=None,
                 hue_haplotype_subtype=None, hue_haplotype_chrom=None, hue_haplotype_pos=None,
                 celltype_filter_func=None, genotype_filter_func=None, cb_whitelist=None,
                 ax=None, figsize=(6, 3), title=None, **violin_kwargs):
        x = self._assign_variable(
            x, gene_id, x_haplotype_subtype, x_haplotype_chrom, x_haplotype_pos
        )
        if hue is None:
            hue = x
        else:
            hue = self._assign_variable(
                hue, gene_id, hue_haplotype_subtype, hue_haplotype_chrom, hue_haplotype_pos
            )
        y = self.exprs_mat.loc[gene_id]

        if cb_whitelist is None:
            cb_whitelist = self.cb_whitelist

        if celltype_filter_func is not None:
            celltype_cb_whitelist = self.celltypes[celltype_filter_func].index.tolist()
            cb_whitelist = list(set(cb_whitelist).intersection(celltype_cb_whitelist))

        if genotype_filter_func is not None:
            genotype_cb_whitelist = self.parental_genotypes[genotype_filter_func].index.tolist()
            cb_whitelist = list(set(cb_whitelist).intersection(genotype_cb_whitelist))

        x = x[cb_whitelist]
        y = y[cb_whitelist]
        hue = hue[cb_whitelist]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        sns.violinplot(x=x, y=y, hue=hue, ax=ax, **violin_kwargs)
        ax.set_ylabel(f'{gene_id} log2 gene expression')
        if title is not None:
            ax.set_title(title)
        return ax
