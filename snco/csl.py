import os
import numpy as np
from scipy.io import mmread
import pysam

from .bam import read_cb_whitelist
from .records import MarkerRecords


def read_chrom_sizes(chrom_sizes_fn):
    chrom_sizes = {}
    with open(chrom_sizes_fn) as f:
        for record in f:
            chrom, cs = record.strip().split('\t')[:2]
            chrom_sizes[chrom] = int(cs)
    return chrom_sizes


def read_vcf(vcf_fn, bin_size):
    with pysam.VariantFile(vcf_fn) as vcf:
        variants = []
        for snp in vcf.fetch():
            variants.append((snp.contig, snp.pos // bin_size))
        return variants


def parse_cellsnp_lite(csl_dir, chrom_sizes_fn, bin_size, cell_barcode_whitelist=None):
    dep_fn = os.path.join(csl_dir, 'cellSNP.tag.DP.mtx')
    alt_fn = os.path.join(csl_dir, 'cellSNP.tag.AD.mtx')
    vcf_fn = os.path.join(csl_dir, 'cellSNP.base.vcf')
    barcode_fn = os.path.join(csl_dir, 'cellSNP.samples.tsv')

    chrom_sizes = read_chrom_sizes(chrom_sizes_fn)
    dep_mm = mmread(dep_fn)
    alt_mm = mmread(alt_fn).tocsr()
    barcodes = read_cb_whitelist(barcode_fn)
    variants = read_vcf(vcf_fn, bin_size)

    co_markers = MarkerRecords(chrom_sizes, bin_size, cell_barcode_whitelist)

    for cb_idx, var_idx, tot in zip(dep_mm.col, dep_mm.row, dep_mm.data):
        alt = alt_mm[var_idx, cb_idx]
        ref = tot - alt
        cb = barcodes[cb_idx]
        chrom, bin_idx = variants[var_idx]
        co_markers[cb, chrom, bin_idx, 0] += ref
        co_markers[cb, chrom, bin_idx, 1] += alt
    return co_markers