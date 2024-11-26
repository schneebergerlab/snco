#!/usr/bin/env python3
import re
import sys
import pysam
import click

try:
    import parasail
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'this script requires parasail for sequence alignment, please install with conda or pip'
    ) from exc


def hdr_to_shv(hdr_record, open_pen=26, extend_pen=1, match_score=1, mismatch_score=-4, min_score=0.25):
    '''
    Resolve a region called as highly diverged by syri into SNPs and indels
    '''
    ref_hdr, alt_hdr = hdr_record.alleles
    assert len(ref_hdr) > 1 and len(alt_hdr) > 1
    ref_start = hdr_record.pos
    hdr_id = hdr_record.id
    
    aln = parasail.sg_trace_striped_32(
        alt_hdr[1:], ref_hdr[1:],
        open_pen, 
        extend_pen,
        parasail.matrix_create("ACGT", match_score, mismatch_score)
    )
    score = aln.score / (min(len(ref_hdr), len(alt_hdr)) - 1)
    if score >= min_score:

        snp_counter = 1
        ins_counter = 1
        del_counter = 1

        ref_pos = ref_start + 1
        ref_idx = 1
        alt_idx = 1
        for iop in aln.cigar.seq:
            ln, op = aln.cigar.decode_len(iop), aln.cigar.decode_op(iop).decode()
            if op == '=':
                # match, no VCF record emitted
                ref_idx += ln
                alt_idx += ln
                ref_pos += ln
            elif op == 'X':
                # mismatch, emit SNP
                for i in range(ln):
                    yield ref_pos + i, ref_hdr[ref_idx + i], alt_hdr[alt_idx + i], f'{hdr_id}_SNP{snp_counter}'
                    snp_counter += 1
                ref_pos += ln
                ref_idx += ln
                alt_idx += ln
            elif op == 'D':
                yield ref_pos - 1, ref_hdr[ref_idx - 1: ref_idx + ln], alt_hdr[alt_idx - 1], f'{hdr_id}_DEL{del_counter}'
                del_counter += 1
                ref_pos += ln
                ref_idx += ln
            elif op == 'I':
                yield ref_pos - 1, ref_hdr[ref_idx - 1], alt_hdr[alt_idx - 1: alt_idx + ln], f'{hdr_id}_INS{ins_counter}'
                ins_counter += 1
                alt_idx += ln


@click.command()
@click.argument('syri-vcf-fn', required=True, type=click.Path(exists=True, dir_okay=False), nargs=1)
@click.argument('output-vcf-fn', required=True, type=click.Path(exists=False), nargs=1)
@click.option('-M', '--max-hdr-length', required=False, default=10_000, type=int,
              help='max length of HDRs to resolve to SNPs and Indels')
@click.option('-s', '--min-hdr-aln-quality', 'min_score', required=False, default=0.25, type=float,
              help='min alignment quality of HDRs to resolve to SNPs and Indels')
@click.option('-i', '--max-indel-length', required=False, default=250, type=int,
              help='max length of indel to output')
@click.option('-n', '--alt-sample-name', 'alt_name', required=False, default='hap2', type=str,
              help='name of alternative/haplotype 2 in output vcf (default: hap2)')
@click.option('-g', '--genotype', required=False, default='dip_het', type=click.Choice(['hap_alt', 'dip_het']),
              help=('how to encode the genotype in the vcf file ("hap_alt" means haploid alternative i.e. 1, '
                    '"dip_het" means phased diploid heterozygous i.e. 0|1'))
def main(syri_vcf_fn, output_vcf_fn, max_hdr_length, min_score, max_indel_length, alt_name, genotype):
    '''
    convert a vcf file from syri into a suitable format for use with STAR diploid and snco analysis
    variants overlapping non-syntenic regions will be filtered out

    NB: syri vcf MUST be sorted by position for filtering to work correctly - this is assumed and not checked
    '''
    if genotype == 'hap_alt':
        gt = '1'
    elif genotype == 'dip_het':
        gt = '0|1'
    else:
        raise ValueError('Unknown genotype')
    with open(output_vcf_fn, 'w') as o:
        o.write(
            '##fileformat=VCFv4.3\n'
            '##source=syri_vcf_to_stardiploid.py\n'
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            '##INFO=<ID=MTD,Number=1,Type=String,Description="Method">\n'
        )
        filt_chrom = None
        filt_end = 0
        with pysam.VariantFile(syri_vcf_fn) as v:
            for chrom, cntg in v.header.contigs.items():
                o.write(f'##contig=<ID={chrom},length={cntg.length}\n')
            o.write(
                f'#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{alt_name}\n'
            )
            for r in v.fetch():
                if r.info['VarType'] == 'SR' and not re.match('(SYN\d+)|(INV\d+)', r.id):
                    # structural variant, we need to filter any ShV that overlap these positions
                    # only works if file is fully sorted
                    if r.chrom == filt_chrom:
                        filt_end = max(filt_end, r.stop)
                    else:
                        filt_chrom = r.chrom
                        filt_end = r.stop
                if r.info['VarType'] ==  'ShV' and r.ref != 'N' and re.match('(SYN\d+)|(INV\d+)', r.info['Parent']) and not (r.pos <= filt_end and r.chrom == filt_chrom):
                    allele_size = max(len(a) for a in r.alleles)
                    if not r.id.startswith('HDR'):
                        if allele_size <= max_indel_length:
                            outrecord = (
                                f'{r.chrom}\t{r.pos}\t{r.id}\t{r.alleles[0]}\t{r.alleles[1]}\t'
                                f'.\tPASS\tMTD=syri\tGT\t{gt}\n'
                            )
                            o.write(outrecord)
                    else:
                        if allele_size <= max_hdr_length:
                            for pos, ref, alt, id_ in hdr_to_shv(r, min_score=min_score):
                                if max(len(ref), len(alt)) <= max_indel_length:
                                    outrecord = (
                                        f'{r.chrom}\t{pos}\t{id_}\t{ref}\t{alt}\t'
                                        f'.\tPASS\tMTD=syri\tGT\t{gt}\n'
                                    )
                                    o.write(outrecord)


if __name__ == '__main__':
    main()