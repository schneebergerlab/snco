import sys
import os
from operator import attrgetter
import subprocess as sp
import tempfile
from contextlib import contextmanager
import itertools as it

import numpy as np

import pysam
import click


def test_close_alignment_positions(chroms, positions, pos_tol=10):
    same_chrom = all([c == chroms[0] for c in chroms])
    if not same_chrom:
        return False
    med_pos = np.median(positions)
    same_pos = all([abs(pos - med_pos) <= pos_tol for pos in positions])
    return same_pos


def aln_collapser(bam_bundle_iter, pos_tol=10):
    for _, bundle in bam_bundle_iter:
        high_score = 0
        representative_aln = None
        ha_flag = []
        hs_chroms = []
        hs_positions = []
        for aln in bundle:
            if aln.is_unmapped:
                continue
            if aln.get_tag('NH') > 1:
                # skip bundles with multimappers
                break
            score = aln.get_tag('AS')
            if score > high_score:
                representative_aln = aln
                ha_flag = [aln.get_tag('RG'),]
                high_score = score
                hs_chroms = [aln.reference_name, ]
                hs_positions = [aln.reference_start, ]
            elif score == high_score:
                ha_flag.append(aln.get_tag('RG'))
                hs_chroms.append(aln.reference_name)
                hs_positions.append(aln.reference_start)
            else:
                continue
        else:
            if representative_aln is not None:
                if test_close_alignment_positions(hs_chroms, hs_positions, pos_tol):
                    ha_flag.sort()
                    representative_aln.set_tag('ha', ','.join(ha_flag))
                    representative_aln.set_tag('RG', None)
                    yield representative_aln


@contextmanager
def collapse_bam_alignments(bam_fn, position_tolerance=10):
    bam = pysam.AlignmentFile(bam_fn, mode='rb')

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

    bam_iter = aln_collapser(it.groupby(
        bam.fetch(until_eof=True),
        key=attrgetter('query_name')
    ), position_tolerance)

    try:
        yield header, bam_iter
    finally:
        bam.close()


@click.command()
@click.argument('input-bam', nargs=1, required=True)
@click.option('-o', '--output-bam', nargs=1, required=True)
@click.option('--position-tolerance', required=False, default=10)
def main(input_bam, output_bam, position_tolerance):
    with collapse_bam_alignments(input_bam, position_tolerance) as (header, bam_iter):
        with pysam.AlignmentFile(output_bam, 'wb', header=header) as output_bam:
            for aln in bam_iter:
                output_bam.write(aln)


if __name__ == "__main__":
    main()