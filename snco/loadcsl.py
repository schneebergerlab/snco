'''functions for converting cellsnp-lite output into snco.MarkerRecords format'''
import os
import logging
import itertools as it

import numpy as np
from scipy.io import mmread

import pysam

from .utils import read_cb_whitelist, genotyping_results_formatter
from .records import MarkerRecords
from .bam import IntervalMarkerCounts
from .genotype import genotype_from_inv_counts
from .clean import filter_low_coverage_barcodes, filter_genotyping_score
from .opts import DEFAULT_RANDOM_SEED


log = logging.getLogger('snco')
DEFAULT_RNG = np.random.default_rng(DEFAULT_RANDOM_SEED)


def read_chrom_sizes(chrom_sizes_fn):
    '''
    load dict of chromosome lengths from a 2 column text file or faidx file
    '''
    chrom_sizes = {}
    with open(chrom_sizes_fn) as f:
        for record in f:
            chrom, cs = record.strip().split('\t')[:2]
            chrom_sizes[chrom] = int(cs)
    return chrom_sizes


def get_vcf_samples(vcf_fn, ref_name):
    with pysam.VariantFile(vcf_fn) as vcf:
        samples = set(vcf.header.samples)
    samples.add(ref_name)
    return frozenset(samples)


def parse_sample_alleles(variant, ref_name):
    if len(variant.alleles) != 2:
        raise ValueError('Only biallelic variants are allowed')
    ref_samples = set()
    ref_samples.add(ref_name)
    alt_samples = set()
    for sample_id, sample in variant.samples.items():
        if len(sample.alleles) > 1:
            raise ValueError('Only haploid haplotype calls are allowed for each sample')
        if not sample.allele_indices[0]:
            ref_samples.add(sample_id)
        else:
            alt_samples.add(sample_id)
    return frozenset(ref_samples), frozenset(alt_samples)


class VariantRecords:

    def __init__(self):
        self._records = []
        self._samples = {}

    def add(self, contig, pos, sample_alleles=None):
        self._records.append((contig, pos))
        self._samples[(contig, pos)] = sample_alleles

    def __getitem__(self, index):
        return self._records[index]

    def get_samples(self, contig, pos):
        return self._samples[(contig, pos)]


def read_vcf(vcf_fn, drop_samples=True, reference_name='col0'):
    with pysam.VariantFile(vcf_fn, drop_samples=drop_samples) as vcf:
        variants = VariantRecords()
        for record in vcf.fetch():
            if drop_samples:
                sample_alleles = None
            else:
                try:
                    sample_alleles = parse_sample_alleles(record, reference_name)
                except ValueError:
                    continue
            variants.add(record.contig, record.pos, sample_alleles)
    return variants


def parse_cellsnp_lite(csl_dir, validate_barcodes=True):
    '''
    read data from cellsnp-lite output into a MarkerRecords object
    '''
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


def parse_cellsnp_lite_interval_counts(csl_dir, bin_size, cb_whitelist,
                                       snp_counts_only=False,
                                       keep_genotype=False,
                                       genotype_vcf_fn=None,
                                       validate_barcodes=True,
                                       reference_name='col0'):
    dep_mm, alt_mm, barcodes, variants = parse_cellsnp_lite(
        csl_dir, validate_barcodes=validate_barcodes
    )
    if keep_genotype:
        if genotype_vcf_fn is None:
            raise ValueError('must supply genotype_vcf_fn when using run_genotype')
        genotype_alleles = read_vcf(
            genotype_vcf_fn,
            drop_samples=False,
            reference_name=reference_name
        )
    else:
        genotype_alleles = None

    inv_counts = {}

    for cb_idx, var_idx, tot in zip(dep_mm.col, dep_mm.row, dep_mm.data):
        cb = barcodes[cb_idx]
        if cb in cb_whitelist:
            alt = alt_mm[var_idx, cb_idx]
            ref = tot - alt
            if snp_counts_only:
                if alt > ref:
                    alt = 1.0
                    ref = 0.0
                elif ref > alt:
                    alt = 0.0
                    ref = 1.0
                else:
                    continue
            chrom, pos = variants[var_idx]
            bin_idx = pos // bin_size
            inv_counts_idx = (chrom, bin_idx)
            if inv_counts_idx not in inv_counts:
                inv_counts[inv_counts_idx] = IntervalMarkerCounts(chrom, bin_idx)
            if genotype_alleles is None:
                inv_counts[inv_counts_idx][cb][0] += ref
                inv_counts[inv_counts_idx][cb][1] += alt
            else:
                ref_genos, alt_genos = genotype_alleles.get_samples(chrom, pos)
                inv_counts[inv_counts_idx][cb][ref_genos] += ref
                inv_counts[inv_counts_idx][cb][alt_genos] += alt
    return list(inv_counts.values())


def cellsnp_lite_to_co_markers(csl_dir, chrom_sizes_fn, bin_size, cb_whitelist,
                               validate_barcodes=True, snp_counts_only=False,
                               run_genotype=False, genotype_vcf_fn=None,
                               reference_name='col0', genotype_kwargs=None):

    inv_counts = parse_cellsnp_lite_interval_counts(
        csl_dir, bin_size, cb_whitelist,
        snp_counts_only=snp_counts_only,
        keep_genotype=run_genotype,
        genotype_vcf_fn=genotype_vcf_fn,
        validate_barcodes=validate_barcodes,
        reference_name=reference_name,
    )

    chrom_sizes = read_chrom_sizes(chrom_sizes_fn)
    co_markers = MarkerRecords(
        chrom_sizes,
        bin_size,
        seq_type='csl_snps',
    )

    if run_genotype:
        if genotype_kwargs is None:
            genotype_kwargs = {}
        if genotype_kwargs.get('crossing_combinations', None) is None:
            haplotypes = get_vcf_samples(genotype_vcf_fn, reference_name)
            genotype_kwargs['crossing_combinations'] = [frozenset(g) for g in it.combinations(haplotypes, r=2)]
        log.info(f'Genotyping barcodes with {len(genotype_kwargs["crossing_combinations"])} possible genotypes')
        genotypes, inv_counts = genotype_from_inv_counts(inv_counts, **genotype_kwargs)
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug(genotyping_results_formatter(genotypes))
        co_markers.metadata['genotypes'] = genotypes

    for ic in inv_counts:
        co_markers.update(ic)

    return co_markers


def run_loadcsl(cellsnp_lite_dir, chrom_sizes_fn, output_json_fn, *,
                cb_whitelist_fn=None, bin_size=25_000, snp_counts_only=False,
                run_genotype=False, genotype_vcf_fn=None,
                genotype_crossing_combinations=None, reference_genotype_name='col0',
                genotype_em_max_iter=1000, genotype_em_min_delta=1e-3, genotype_em_bootstraps=25,
                min_markers_per_cb=100, min_markers_per_chrom=20, min_geno_prob=0.9,
                validate_barcodes=True, processes=1, rng=DEFAULT_RNG):
    '''
    Read matrix files generated by cell snp lite to generate a json file of binned
    haplotype marker distributions for each cell barcode. These can be used to
    call recombinations using the downstream `predict` command.
    '''

    cb_whitelist = read_cb_whitelist(cb_whitelist_fn, validate_barcodes=validate_barcodes)
    co_markers = cellsnp_lite_to_co_markers(
        cellsnp_lite_dir,
        chrom_sizes_fn,
        bin_size=bin_size,
        cb_whitelist=cb_whitelist,
        validate_barcodes=validate_barcodes,
        run_genotype=run_genotype,
        genotype_vcf_fn=genotype_vcf_fn,
        reference_name=reference_genotype_name,
        genotype_kwargs={
            'crossing_combinations': genotype_crossing_combinations,
            'max_iter': genotype_em_max_iter,
            'min_delta': genotype_em_min_delta,
            'n_bootstraps': genotype_em_bootstraps,
            'min_markers_per_cb': min_markers_per_cb,
            'processes': processes,
            'rng': rng
        },
    )
    n = len(co_markers)
    log.info(f'Identified {n} cell barcodes from cellsnp-lite files')
    if min_markers_per_cb or min_markers_per_chrom:
        co_markers = filter_low_coverage_barcodes(
            co_markers, min_markers_per_cb, min_markers_per_chrom
        )
        log.info(
            f'Removed {n - len(co_markers)} barcodes with fewer than {min_markers_per_cb} markers '
            f'or fewer than {min_markers_per_chrom} markers per chromosome'
        )
    n = len(co_markers)
    if min_geno_prob:
        co_markers = filter_genotyping_score(co_markers, min_geno_prob)
        log.info(
            f'Removed {n - len(co_markers)} barcodes with genotyping probability < {min_geno_prob}'
        )
    if output_json_fn is not None:
        log.info(f'Writing markers to {output_json_fn}')
        co_markers.write_json(output_json_fn)
    return co_markers
