import sys
import os
from operator import attrgetter
import subprocess as sp
import tempfile
from contextlib import contextmanager
import itertools as it

import pysam
import click


def aln_collapser(accessions):
    def _collapse(bam_bundle_iter):
        for _, bundle in bam_bundle_iter:
            high_score = 0
            representative_aln = None
            ha_flag = []
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
                elif score == high_score:
                    ha_flag.append(aln.get_tag('RG'))
                else:
                    continue
            else:
                if representative_aln is not None:
                    ha_flag.sort()
                    representative_aln.set_tag('ha', ','.join(ha_flag))
                    representative_aln.set_tag('RG', None)
                    yield representative_aln
    return _collapse


def get_bam_rgs(bam_fns):
    rgs = []
    for bam_fn in bam_fns:
        with pysam.AlignmentFile(bam_fn) as bam:
            try:
                header_rg = bam.header['RG']
            except KeyError:
                raise IOError('BAM files must have RG tags if --infer-rg is not specified')
            assert len(header_rg) == 1
            rgs.append(header_rg[0]['ID'])
    rgs.sort()
    return rgs


@contextmanager
def collapse_bam_alignments(bam_fns, processes=2, infer_rg=False):
    tmp_bamlist_fh, tmp_bamlist_fn = tempfile.mkstemp()
    with open(tmp_bamlist_fn, 'w') as f:
        f.write('\n'.join(bam_fns))
    if infer_rg:
        collate_threads, r = divmod(processes, 2)
        merge_cmd = f'merge -@{collate_threads + r} -ucr -b {tmp_bamlist_fn} -'
        accessions = sorted([os.path.splitext(os.path.split(fn)[1])[0] for fn in bam_fns])
    else:
        merge_cmd = f'cat -b {tmp_bamlist_fn}'
        collate_threads = processes - 1
        accessions = get_bam_rgs(bam_fns)
    samtools_cmd = f'samtools {merge_cmd} | samtools collate -Ou -@ {collate_threads} -'
    proc = sp.Popen(samtools_cmd, shell=True, stdout=sp.PIPE)
    bam = pysam.AlignmentFile(proc.stdout, mode='rb')

    header = bam.header.to_dict()
    accessions.sort()
    header.pop('RG', None)

    header['CO'].append(
        'ha_flag_accessions: ' + ','.join(accessions)
    )
    header['PG'].append(
        {
            'ID': os.path.split(sys.argv[0])[1],
            'PN': os.path.split(sys.argv[0])[1],
            'CL': ' '.join(sys.argv)
        }
    )

    collapse = aln_collapser(accessions)
    bam_iter = collapse(it.groupby(
        bam.fetch(until_eof=True),
        key=attrgetter('query_name')
    ))

    try:
        yield header, bam_iter
    finally:
        bam.close()
        os.close(tmp_bamlist_fh)
        os.remove(tmp_bamlist_fn)
        proc.terminate()


@click.command()
@click.argument('input-bams', nargs=-1, required=True)
@click.option('-o', '--output-bam', nargs=1, required=True)
@click.option('--infer-rg/--use-existing-rg', required=False, default=False)
@click.option('-p', '--processes', required=False, type=click.IntRange(min=2), default=2)
def main(input_bams, output_bam, infer_rg, processes):
    with collapse_bam_alignments(input_bams, processes, infer_rg) as (header, bam_iter):
        with pysam.AlignmentFile(output_bam, 'wb', header=header) as output_bam:
            for aln in bam_iter:
                output_bam.write(aln)


if __name__ == "__main__":
    main()