import os
from scipy.io import mmread
from .vcf import read_vcf
from ..barcodes import read_cb_whitelist


def parse_cellsnp_lite(csl_dir, validate_barcodes=True):
    """
    Parse the output of cellsnp-lite into sparse matrices.

    Parameters
    ----------
    csl_dir : str
        Path to the directory containing the cellsnp-lite output files.
    validate_barcodes : bool, optional
        Whether to validate the barcodes (default is True).

    Returns
    -------
    dep_mm : scipy.sparse matrix
        The depth matrix.
    alt_mm : scipy.sparse matrix
        The alternate allele matrix.
    barcodes : list of str
        The list of cell barcodes.
    variants : VariantRecords
        The parsed variant records.
    """
    dep_fn = os.path.join(csl_dir, 'cellSNP.tag.DP.mtx')
    alt_fn = os.path.join(csl_dir, 'cellSNP.tag.AD.mtx')
    vcf_fn = os.path.join(csl_dir, 'cellSNP.base.vcf')
    if not os.path.exists(vcf_fn):
        vcf_fn = f'{vcf_fn}.gz'
    barcode_fn = os.path.join(csl_dir, 'cellSNP.samples.tsv')
    dep_mm = mmread(dep_fn)
    alt_mm = mmread(alt_fn).tocsr()
    barcodes = read_cb_whitelist(barcode_fn, validate_barcodes=validate_barcodes)
    variants = read_vcf(vcf_fn)
    return dep_mm, alt_mm, barcodes, variants
