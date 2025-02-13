import logging
import itertools as it
from functools import cached_property

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import joblib

from .. import PredictionRecords
from ..opts import DEFAULT_RANDOM_SEED
from ..logger import progress_bar
from . import covars, utils

log = logging.getLogger('snco')

LOD = 2 * np.log(10)
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def null_model_covariates(exprs_mat):
    return pd.DataFrame(
        np.ones(shape=(exprs_mat.shape[1], 1)),
        columns=['const'],
        index=exprs_mat.columns
    )


def likelihood_ratio(lln, llf):
    return -2 * (lln - llf)


def compare_lr_test(m_full, m_nest):
    
    Ln, Lf = m_nest.llf, m_full.llf
    dfn, dff = m_nest.df_model, m_full.df_model
    
    chi2_stat = likelihood_ratio(Ln, Lf)
    p = stats.chi2.sf(chi2_stat, dff - dfn)

    return chi2_stat / LOD, p


def calculate_wald_test_lod_scores(m, variables):
    lods = []
    pvals = []
    coefs = []
    for var in variables:
        w = m.wald_test(var, scalar=True, use_f=False)
        lods.append(w.statistic / LOD)
        pvals.append(w.pvalue)
        coefs.append(m.params.loc[var])
    return lods, pvals, coefs


def fit_model(gene_exprs, variables, model_type='ols'):
    if model_type == 'ols':
        m = sm.OLS(gene_exprs, variables, hasconst=True)
    elif model_type == 'logit':
        m = sm.Logit(gene_exprs, variables)
    else:
        raise ValueError('unknown model type')
    return m.fit(disp=0)


def estimate_effective_haplotypes(hap_probs):
    '''
    Estimate the effective number of haplotypes in order to perform FWER correction
    of p values from eQTL tests (which are not independent due to LD)
    Uses the Li and Hi method (Heredity volume 95, pages221–227 (2005))
    '''
    H_eff = 0

    for chrom, h in hap_probs.T.groupby(level='chrom'):
        corr = h.T.corr(method='pearson')
        eig = np.abs(np.linalg.eigvalsh(corr))
        chrom_H_eff = np.sum((eig >= 1).astype(int) + (eig - np.floor(eig)))
        H_eff += round(chrom_H_eff)
    return H_eff


class SNeQTLAnalysis:

    def __init__(self, exprs_mat, haplotypes, cb_whitelist=None, covariates=None,
                 interacting_variables=None, parental_genotypes=None, control_haplotypes=None,
                 interacting_haplotypes=None, control_cis_haplotype=False, gene_locs=None):
        
        (exprs_mat, haplotypes, covariates,
         interacting_variables, parental_genotypes) = utils.align_input(
            exprs_mat, haplotypes, cb_whitelist, covariates,
            interacting_variables, parental_genotypes
        )

        self.exprs_mat = exprs_mat
        self.haplotypes = haplotypes
        covariates = covariates
        self.model_type = 'ols'

        self.interacting_variables = utils.get_dummies(interacting_variables)
        self.parental_genotypes = utils.get_dummies(parental_genotypes)

        if control_haplotypes is not None:
            control_haplotype_variables = []
            for chrom, pos in control_haplotypes:
                control_haplotype_variables.append(self.get_closest_haplotype(chrom, pos))
            self.control_haplotype_variables = pd.concat(control_haplotype_variables, axis=1)
        else:
            self.control_haplotype_variables = pd.DataFrame()

        if interacting_haplotypes is not None:
            interacting_haplotype_variables = []
            for chrom, pos in interacting_haplotypes:
                haplo = self.get_closest_haplotype(chrom, pos)
                interacting_haplotype_variables.append(haplo)
                interacting_haplotype_variables.append((1 - haplo).rename('inv-' + haplo.name))
            self.interacting_haplotype_variables = pd.concat(interacting_haplotype_variables, axis=1)
        else:
            self.interacting_haplotype_variables = pd.DataFrame()

        self.covariates = utils.concat_handle_none([
            utils.drop_first_column(self.interacting_variables),
            self.interacting_haplotype_variables.iloc[:, ::2],
            utils.drop_first_column(self.parental_genotypes),
            self.control_haplotype_variables,
            utils.get_dummies(covariates, drop_first=True)
        ], axis=1)

        if self.covariates.empty:
            # just add intercept as null model
            self.covariates = null_model_covariates(exprs_mat)
        else:
            self.covariates = sm.add_constant(self.covariates)

        if control_cis_haplotype and gene_locs is None:
            raise ValueError('when control_cis_haplotype is True, gene_locs must be provided')
        self.control_cis_haplotype = control_cis_haplotype
        self.gene_locs = gene_locs

        self._gene_specific_covariates = {}
        self._haplotype_specific_variables = {}
        self._nested_models = {}
        self._param_names = None
        self._result_colnames = None
        self._H_eff = None

    def haplotype_variables(self, chrom, pos):
        try:
            haplo_variables = self._haplotype_specific_variables[(chrom, pos)]
        except KeyError:
            haplo = self.haplotypes.loc[:, (chrom, pos)]
            haplo_variables = haplo.rename('haplo').to_frame()
            if not self.parental_genotypes.empty:
                haplo_variables = utils.expanding_mul(
                    haplo_variables, self.parental_genotypes
                )
            if not self.interacting_variables.empty:
                haplo_variables = utils.expanding_mul(
                    haplo_variables, self.interacting_variables
                )
            if not self.interacting_haplotype_variables.empty:
                haplo_variables = utils.expanding_mul(
                    haplo_variables, self.interacting_haplotype_variables
                )
            self._haplotype_specific_variables[(chrom, pos)] = haplo_variables
        return haplo_variables

    def get_closest_haplotype(self, chrom, pos):
        chrom_bins = self.haplotypes[chrom].columns.values
        pos = chrom_bins[np.searchsorted(chrom_bins, pos, side='right') - 1]
        return self.haplotypes[(chrom, pos)].rename(f'{chrom}:{pos/1e6:.1f}Mb')

    def get_cis_haplotype(self, gene_id):
        chrom, pos = self.gene_locs.loc[gene_id].values
        return self.get_closest_haplotype(chrom, pos).rename(f'cishaplo-{gene_id}')

    def get_covariates(self, gene_id):
        try:
            covars = self._gene_specific_covariates[gene_id]
        except KeyError:
            covars = self.covariates
            if self.control_cis_haplotype:
                cis_haplo = self.get_cis_haplotype(gene_id)
                covars = pd.concat([covars, cis_haplo], axis=1)
            self._gene_specific_covariates[gene_id] = covars
        return covars

    def nested_model(self, gene_id):
        try:
            m_nest = self._nested_models[gene_id]
        except KeyError:
            gene_exprs = self.exprs_mat.loc[gene_id]
            m_nest = fit_model(
                gene_exprs,
                self.get_covariates(gene_id),
                model_type = self.model_type
            )
            self._nested_models[gene_id] = m_nest
        return m_nest

    def full_model(self, gene_id, chrom, pos):
        gene_exprs = self.exprs_mat.loc[gene_id]
        haplo_variables = self.haplotype_variables(chrom, pos)
        m_full = fit_model(
            gene_exprs,
            pd.concat([haplo_variables, self.get_covariates(gene_id)], axis=1),
            model_type=self.model_type
        )
        return m_full

    @property
    def param_names(self):
        if self._param_names is None:
            variables = [('haplo',)]
            if not self.parental_genotypes.empty:
                variables.append(self.parental_genotypes.columns.tolist())
            if not self.interacting_variables.empty:
                variables.append(self.interacting_variables.columns.tolist())
            if not self.interacting_haplotype_variables.empty:
                variables.append(self.interacting_haplotype_variables.columns.tolist())
            self._param_names = ['_'.join(v) for v in it.product(*variables)]
        return self._param_names

    @property
    def result_colnames(self):
        if self._result_colnames is None:
            rn = ['gene_id', 'chrom', 'pos', 'lod_score', 'pval']
            for var in self.param_names:
                rn.append(f'{var}_lod')
                rn.append(f'{var}_pval')
                rn.append(f'{var}_coef')
            self._result_colnames = rn
        return self._result_colnames

    @property
    def gene_ids(self):
        return self.exprs_mat.index.tolist()

    @property
    def barcodes(self):
        return self.exprs_mat.columns.tolist()

    @property
    def haplotype_positions(self):
        return self.haplotypes.columns.tolist()

    @property
    def effective_haplotypes(self):
        if self._H_eff is None:
            self._H_eff = estimate_effective_haplotypes(self.haplotypes)
        return self._H_eff

    def apply_fwer_correction(self, results, inplace=True):
        if not inplace:
            results = results.copy()
        H_eff = self.effective_haplotypes
        pval_cols = [f'{p}_pval' for p in self.param_names]
        results[pval_cols] = np.clip(results[pval_cols] * H_eff, a_min=1.0, a_max=np.finfo(np.float64).tiny)
        return results

    def apply_fdr_correction(self, results, inplace=True):
        if not inplace:
            results = results.copy()
        pval_cols = [f'{p}_pval' for p in self.param_names]
        results[pval_cols] = results.groupby(
            ['chrom', 'pos'],
            as_index=False,
            group_keys=False,
            sort=False
        )[pval_cols].transform(stats.false_discovery_control)
        return results

    def run_single_test(self, gene_id, chrom, pos):
        m_nest = self.nested_model(gene_id)
        m_full = self.full_model(gene_id, chrom, pos)
        lrt_res = compare_lr_test(m_full, m_nest)
        lods, pvals, coefs = calculate_wald_test_lod_scores(m_full, self.param_names)
        return pd.Series(
            [gene_id, chrom, pos, *lrt_res, *it.chain(*zip(lods, pvals, coefs))],
            index=self.result_colnames
        )

    def run_gene(self, gene_id, fwer_correction=False):
        results = []
        for chrom, pos in self.haplotype_positions:
            results.append(self.run_single_test(gene_id, chrom, pos))
        results = pd.DataFrame(results)
        if fwer_correction:
            results = self.apply_fwer_correction(results)
        return results

    def run_all_genes(self, genewise_fdr_correction=True, haplowise_fwer_correction=True, processes=1):
        with joblib.Parallel(n_jobs=processes, backend='loky') as pool:
            gene_progress = progress_bar(
                self.gene_ids,
                label='Running eQTL',
                item_show_func=str
            )
            with gene_progress:
                results = pool(
                    joblib.delayed(self.run_gene)(gene_id)
                    for gene_id in gene_progress
                )
        results = pd.concat(results, axis=0)
        if genewise_fdr_correction:
            results = self.apply_fdr_correction(results)
        if haplowise_fwer_correction:
            results = self.apply_fwer_correction(results)
        return results


def run_eqtl(exprs_mat_dir, pred_json_fn, cb_stats_fn, output_prefix, gtf_fn=None,
             min_cells_exprs=0.02,
             control_principal_components=True, min_pc_var_explained=0.01,
             max_pc_haplotype_var_explained=0.05, whitelist_covar_names=None,
             model_parental_genotype=False, parental_genotype_colname='geno',
             celltype_haplotype_interaction=False, celltype_n_clusters='auto',
             control_haplotypes=None, control_cis_haplotype=False,
             control_haplotypes_r2=0.95, cb_filter_exprs=None,
             processes=1, rng=DEFAULT_RNG):

    if cb_stats_fn is not None:
        cb_stats = utils.read_cb_stats(cb_stats_fn, cb_filter_exprs)
        cb_whitelist = cb_stats.index.tolist()
        log.info(f'Read {len(cb_whitelist)} barcodes')
    else:
        cb_stats = None
        cb_whitelist = None
    exprs_mat = utils.read_expression_matrix(exprs_mat_dir, cb_whitelist, min_cells_exprs)
    log.info(f'Identified {len(exprs_mat)} genes to be tested')

    haplotypes = PredictionRecords.read_json(pred_json_fn)
    log.info(f'Read haplotype information in {haplotypes.bin_size/1e3:.0f} kb bins')
    log.info(f'There are {len(haplotypes.nbins)} chromosomes and '
             f'{sum(haplotypes.nbins.values())} bins to be tested')
    haplotypes = haplotypes.to_frame().loc[cb_whitelist]

    principal_components = covars.get_principal_components(
        exprs_mat, haplotypes,
        min_var_explained=min_pc_var_explained,
        max_pc_haplotype_var_explained=max_pc_haplotype_var_explained
    )

    if whitelist_covar_names is not None and cb_stats is not None:
        covariates = pd.concat([
            principal_components,
            cb_stats[whitelist_covar_names]
        ], axis=1)
    else:
        covariates = principal_components

    if celltype_haplotype_interaction:
        celltype_clusters = covars.create_celltype_clusters(
            principal_components,
            n_clusters=celltype_n_clusters,
            rng=rng
        )
    else:
        celltype_clusters = None

    if model_parental_genotype and cb_stats is not None:
        parental_genotypes = cb_stats[parental_genotype_colname]
        n_geno = len(parental_genotypes.unique())
        if n_geno == 1:
            log.warn('All parental (diploid) genotypes are the same. Genotype will not be modelled')
            parental_genotypes = None
        else:
            log.info(f'There are {n_geno} parental (diploid) genotypes to be tested')
    else:
        log.info('No parental (diploid) genotypes provided. Genotypes are assumed to be the same for all nuclei')
        parental_genotypes = None

    if control_cis_haplotype and gtf_fn is not None:
        gene_locs = utils.read_gtf_gene_locs(gtf_fn)
    else:
        gene_locs = None

    e = SNeQTLAnalysis(
        exprs_mat.head(10), haplotypes, cb_whitelist,
        covariates=covariates,
        interacting_variables=celltype_clusters,
        parental_genotypes=parental_genotypes,
        control_haplotypes=control_haplotypes,
        control_cis_haplotype=control_cis_haplotype,
        gene_locs=gene_locs
    )
    example_gene_id = e.gene_ids[0]
    example_covars = e.get_covariates(example_gene_id).columns.tolist()
    example_covars = [c.replace(example_gene_id, 'gene_id') for c in example_covars]
    haplo_test_vars = e.param_names
    log.debug(f'Controlling for {len(example_covars)} covariates: {example_covars}')
    log.debug(f'{len(haplo_test_vars)} haplotype-associated variables will be tested: {haplo_test_vars}')

    results = e.run_all_genes(processes=processes)
    
    log.info(f'eQTL analysis complete - writing full results to {output_prefix}.eqtls.tsv')
    results.to_csv(
        f'{output_prefix}.eqtls.tsv',
        sep='\t',
        index=False,
        header=True,
        float_format='%.4g'
    )