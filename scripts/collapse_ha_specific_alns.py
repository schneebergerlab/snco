#!/usr/bin/env python3
import sys
import os
from statistics import median
from contextlib import contextmanager

import pysam
import click


def test_close_alignment_positions(chroms, positions, pos_tol=10):
    assert len(chroms) == len(positions)
    if not len(chroms):
        return True
    same_chrom = all([c == chroms[0] for c in chroms])
    if not same_chrom:
        return False
    med_pos = median(positions)
    same_pos = all([abs(pos - med_pos) <= pos_tol for pos in positions])
    return same_pos


def group_by_queryname(bam_iter):
    prev_qname = None
    group = []
    for aln in bam_iter:
        qname = aln.query_name
        if qname != prev_qname and group:
            yield group
            group = []
        group.append(aln)
        prev_qname = qname
    if group:
        yield group


def strip_mapping_info(aln):
    aln.is_unmapped = True
    aln.reference_id = -1
    aln.reference_start = -1
    aln.mapping_quality = 0
    aln.cigar = None
    aln.next_reference_id = -1
    aln.next_reference_start = -1
    return aln


def aln_collapser(bam_bundle_iter, decoy_haplotypes, pos_tol=10):
    for bundle in bam_bundle_iter:
        high_score = -float('inf')
        representative_aln = None
        representative_mate = None
        repr_is_decoy = None
        is_paired = False
        ha_tag = None
        dh_tag = None
        hs_chroms = []
        hs_positions = []
        for aln in bundle:
            if aln.is_unmapped:
                continue
            if aln.is_paired and not aln.is_proper_pair:
                continue
            if aln.get_tag('NH') > 1:
                # skip bundles with multimappers
                break
            score = aln.get_tag('AS')
            rg_tag = aln.get_tag('RG')
            is_decoy = rg_tag in decoy_haplotypes
            if score > high_score:
                if not aln.is_paired or aln.is_read1:
                    representative_aln = aln
                    representative_mate = None
                    hs_chroms = [aln.reference_name, ] if not is_decoy else []
                    hs_positions = [aln.reference_start, ] if not is_decoy else []
                else:
                    representative_mate = aln
                    representative_aln = None
                    is_paired = True
                    hs_chroms = []
                    hs_positions = []
                if not is_decoy:
                    ha_tag = {rg_tag,}
                    dh_tag = set()
                else:
                    ha_tag = set()
                    dh_tag = {rg_tag,}
                repr_is_decoy = is_decoy
                high_score = score
            elif score == high_score:
                if not aln.is_paired or aln.is_read1:
                    if representative_aln is None or repr_is_decoy:
                        representative_aln = aln
                        repr_is_decoy = is_decoy
                    hs_chroms.append(aln.reference_name)
                    hs_positions.append(aln.reference_start)
                else:
                    if representative_mate is None or repr_is_decoy:
                        representative_mate = aln
                        repr_is_decoy = is_decoy
                    is_paired = True
                if not is_decoy:
                    ha_tag.add(rg_tag)
                else:
                    dh_tag.add(rg_tag)
            else:
                continue
        else:
            if representative_aln is not None:
                ha_tag = sorted(ha_tag)
                ha_tag = ','.join(ha_tag)
                dh_tag = sorted(dh_tag)
                dh_tag = ','.join(dh_tag)
                if repr_is_decoy:
                    representative_aln = strip_mapping_info(representative_aln)
                    if is_paired:
                        representative_mate = strip_mapping_info(representative_mate)
                if test_close_alignment_positions(hs_chroms, hs_positions, pos_tol):
                    representative_aln.set_tag('ha', ha_tag)
                    representative_aln.set_tag('dh', dh_tag)
                    representative_aln.set_tag('RG', None)
                    yield representative_aln
                    if is_paired:
                        representative_mate.set_tag('ha', ha_tag)
                        representative_mate.set_tag('dh', dh_tag)
                        representative_mate.set_tag('RG', None)
                        yield representative_mate


@contextmanager
def collapse_bam_alignments(bam_fn, decoy_haplotypes, position_tolerance=10, threads=1):
    bam = pysam.AlignmentFile(bam_fn, mode='rb', threads=threads)

    header = bam.header.to_dict()

    accessions = sorted([rg['ID'] for rg in header.pop('RG')])
    header['HD']['ha'] = ','.join(accessions)

    assert header['HD']['SO'] == "queryname"

    header['PG'].append(
        {
            'ID': os.path.split(sys.argv[0])[1],
            'PN': os.path.split(sys.argv[0])[1],
            'CL': ' '.join(sys.argv)
        }
    )

    bam_iter = aln_collapser(
        group_by_queryname(bam.fetch(until_eof=True)),
        decoy_haplotypes,
        position_tolerance
    )

    try:
        yield header, bam_iter
    finally:
        bam.close()


@click.command()
@click.argument('input-bam', nargs=1, required=True)
@click.option('-o', '--output-bam', nargs=1, required=True)
@click.option('--decoy-haplotypes', required=False, help='Comma-separated list of decoy RG IDs', default=None)
@click.option('--position-tolerance', required=False, default=10)
@click.option('-t', '--threads', required=False, default=1)
def main(input_bam, output_bam, decoy_haplotypes, position_tolerance, threads):
    decoy_haplotypes = set(decoy_haplotypes.split(',')) if decoy_haplotypes else set()
    with collapse_bam_alignments(input_bam, decoy_haplotypes, position_tolerance, threads // 2) as (header, bam_iter):
        with pysam.AlignmentFile(output_bam, 'wb', header=header, threads=threads // 2) as output_bam:
            for aln in bam_iter:
                output_bam.write(aln)


if __name__ == "__main__":
    main()