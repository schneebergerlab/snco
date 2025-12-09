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
    same_chrom = all(c == chroms[0] for c in chroms)
    if not same_chrom:
        return False
    med_pos = median(positions)
    same_pos = all(abs(pos - med_pos) <= pos_tol for pos in positions)
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


def aln_collapser(bam_bundle_iter, pos_tol=10):
    for bundle in bam_bundle_iter:
        high_score = -float('inf')
        representative_aln = None
        representative_mate = None
        ha_tag = None
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
            if score > high_score:
                if not aln.is_paired or aln.is_read1:
                    representative_aln = aln
                    representative_mate = None
                    hs_chroms = [aln.reference_name, ]
                    hs_positions = [aln.reference_start, ]
                else:
                    representative_mate = aln
                    representative_aln = None
                    hs_chroms = []
                    hs_positions = []
                ha_tag = {rg_tag,}
                high_score = score
            elif score == high_score:
                if not aln.is_paired or aln.is_read1:
                    if representative_aln is None:
                        representative_aln = aln
                    hs_chroms.append(aln.reference_name)
                    hs_positions.append(aln.reference_start)
                else:
                    if representative_mate is None:
                        representative_mate = aln
                ha_tag.add(rg_tag)
            else:
                continue
        else:
            if representative_aln is not None:
                ha_tag = sorted(ha_tag)
                ha_tag = ','.join(ha_tag)
                if test_close_alignment_positions(hs_chroms, hs_positions, pos_tol):
                    representative_aln.set_tag('ha', ha_tag)
                    representative_aln.set_tag('RG', None)
                    yield representative_aln
                    if representative_mate is not None:
                        representative_mate.set_tag('ha', ha_tag)
                        representative_mate.set_tag('RG', None)
                        yield representative_mate


@contextmanager
def collapse_bam_alignments(bam_fn, position_tolerance=10, threads=1):
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
        position_tolerance
    )

    try:
        yield header, bam_iter
    finally:
        bam.close()


@click.command()
@click.argument('input-bam', nargs=1, required=True)
@click.option('-o', '--output-bam', nargs=1, required=True)
@click.option('--position-tolerance', required=False, default=10)
@click.option('-t', '--threads', required=False, default=1)
def main(input_bam, output_bam, position_tolerance, threads):
    with collapse_bam_alignments(input_bam, position_tolerance, max(1, threads // 2)) as (header, bam_iter):
        with pysam.AlignmentFile(output_bam, 'wb', header=header, threads=max(1, threads // 2)) as output_bam:
            for aln in bam_iter:
                output_bam.write(aln)


if __name__ == "__main__":
    main()