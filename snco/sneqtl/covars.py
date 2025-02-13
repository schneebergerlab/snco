import logging

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from ..opts import DEFAULT_RANDOM_SEED

log = logging.getLogger('snco')


def filter_haplotype_correlated_principal_components(principal_components, haplotypes,
                                                     max_var_explained=0.05):
    n_pcs = principal_components.shape[1]
    keep_pcs = []
    for pc_idx, pc in principal_components.items():
        pc_max_r2 = 0.0
        best_corr_chrom = None
        best_corr_pos = None
        for (chrom, pos), haplo in haplotypes.items():
            _, _, r, *_ = stats.linregress(pc, haplo)
            r2 = r ** 2
            if r2 > pc_max_r2:
                pc_max_r2 = r2
                best_corr_chrom = chrom
                best_corr_pos = pos
        if pc_max_r2 <= max_var_explained:
            keep_pcs.append(pc_idx)
        else:
            log.info(
                f'Removing {pc_idx} as it correlates with '
                f'{best_corr_chrom}:{best_corr_pos/1e6:.1f} '
                f'with R squared of {pc_max_r2:.2f}'
            )
    return principal_components[keep_pcs]


def get_principal_components(exprs_mat, haplotypes,
                             min_var_explained=0.01, max_pcs=50,
                             max_pc_haplotype_var_explained=0.05):
    pca = PCA(n_components=max_pcs, svd_solver='arpack')
    principal_components = pca.fit_transform(exprs_mat.T)
    stop_idx = max_pcs - np.searchsorted(
        pca.explained_variance_ratio_[::-1],
        min_var_explained
    )
    principal_components = pd.DataFrame(
        principal_components[:, :stop_idx],
        columns=[f'PC{i}' for i in range(1, stop_idx + 1)],
        index=exprs_mat.columns
    )
    tot_ve = pca.explained_variance_ratio_[:stop_idx].sum() * 100
    log.info(f'Identified {stop_idx} principal components explaining {tot_ve:.2f}% of variance')
    return filter_haplotype_correlated_principal_components(
        principal_components, haplotypes, max_pc_haplotype_var_explained
    )


def create_celltype_clusters(principal_components, n_clusters=None,
                             auto_max_clusters=10,
                             covariance_type='spherical',
                             rng=None):
    if rng is None:
        random_state = np.random.RandomState(DEFAULT_RANDOM_SEED)
    else:
        random_state = np.random.RandomState(rng.integers(1000))
    if principal_components.empty:
        return None
    max_s = 0
    if n_clusters is None:
        for n in range(2, auto_max_clusters + 1):
            gmm = GaussianMixture(
                n_components=n,
                covariance_type=covariance_type,
                random_state=random_state
            ).fit(principal_components)
            s = silhouette_score(principal_components, gmm.predict(principal_components))
            if s > max_s:
                max_s = s
                best_gmm = gmm
    else:
        best_gmm = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            covariance_type=covariance_type,
        ).fit(principal_components)
        max_s = silhouette_score(principal_components, best_gmm.predict(principal_components))
    log.info(f'Identified {best_gmm.n_components} putative '
             f'cell-type clusters with silhouette score {max_s:.2f}')
    cluster_preds = best_gmm.predict_proba(principal_components)
    _, cluster_counts = np.unique(cluster_preds.argmax(axis=1), return_counts=True)
    cluster_preds = cluster_preds[:, np.argsort(cluster_counts)[::-1]]
    log.debug(f'Cell-type cluster sizes: {sorted(cluster_counts)}')
    celltype_clusters = pd.DataFrame(
        cluster_preds,
        index=principal_components.index,
        columns=[f'celltype{i}' for i in range(1, best_gmm.n_components + 1)]
    )
    return celltype_clusters