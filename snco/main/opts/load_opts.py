import click
from .snco_opts import snco_opts
from snco.defaults import DEFAULT_EXCLUDE_CONTIGS


snco_opts.option(
    '--cb-correction-method',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=click.Choice(['exact', '1mm', 'auto'], case_sensitive=False),
    default='auto',
    help='method for correcting/matching cell barcodes to whitelist'
)


def _validate_bam_tag(ctx, param, value):
    if not len(value) == 2:
        raise ValueError(f'{param} is not of length 2')
    return value


snco_opts.option(
    '--cb-tag',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=str,
    default='CB',
    callback=_validate_bam_tag,
    help='bam file tag representing cell barcode'
)


snco_opts.option(
    '--umi-collapse-method',
    subcommands=['loadbam', 'bam2pred'],
    is_eager=True,
    required=False,
    type=click.Choice(['exact', 'directional', 'none', 'auto'], case_sensitive=False),
    default='auto',
    help='Method for deduplicating/collapsing UMIs'
)


snco_opts.option(
    '--umi-tag',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=str,
    default='UB',
    help='bam file tag representing UMI'
)


snco_opts.option(
    '--hap-tag',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=str,
    default='ha',
    callback=_validate_bam_tag,
    help='bam file tag representing haplotype.'
)


snco_opts.option(
    '--hap-tag-type',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=click.Choice(['star_diploid', 'multi_haplotype']),
    default='star_diploid',
    help='how the haplotype tag is encoded, see manual for details' # todo!
)


snco_opts.option(
    '--min-alignment-score',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=click.FloatRange(0, 1),
    default=0.95,
    help='only reads with a length normalised alignment score greater than this value are used'
)


snco_opts.option(
    '--genotype/--no-genotype', 'run_genotype',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    required=False,
    default=False,
    help='whether to use EM algorithm to infer genotypes (requires --hap-tag-type="multi_haplotype")',
)


def _parse_crossing_combinations(ctx, param, value):
    if value is not None:
        value = set([frozenset(geno.split(':')) for geno in value.split(',')])
    return value


snco_opts.option(
    '-X', '--crossing-combinations', 'genotype_crossing_combinations',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    required=False,
    default=None,
    callback=_parse_crossing_combinations,
    help=('comma separated list of allowed combinations of parental haplotypes used in crosses, '
          'encoded in format "hap1:hap2,hap1:hap3" etc')
)


snco_opts.option(
    '--genotype-em-max-iter',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(1, 10_000),
    default=1000,
    help='the maximum number of iterations to run the genotyping EM algorithm for'
)

snco_opts.option(
    '--genotype-em-min-delta',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(0, 0.1),
    default=1e-3,
    help='the minimum difference in genotype probability between EM iterations, before stopping'
)


snco_opts.option(
    '--genotype-em-bootstraps',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(1, 1000),
    default=25,
    help='the number of bootstrap resamples to use for estimating genotype probabilities'
)

snco_opts.option(
    '--reference-geno-name', 'reference_genotype_name',
    subcommands=['loadcsl', 'csl2pred'],
    required=False,
    default='col0',
    help='name of the reference genotype (alt genotype names are taken from vcf samples)',
)


snco_opts.option(
    '--validate-barcodes/--no-validate',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred'],
    required=False,
    default=True,
    help='whether to check that cell barcodes are valid sequences',
)


snco_opts.option(
    '--count-snps-only/--count-cov-per-snp', 'snp_counts_only',
    subcommands=['loadcsl', 'csl2pred'],
    required=False,
    default=False,
    help='whether to record snp coverage or just consensus genotype of snp',
)


def _parse_excl_contigs(ctx, param, value):
    if value is None:
        value = DEFAULT_EXCLUDE_CONTIGS
    else:
        value = set(value.split(',')) # todo: check allowed fasta header names
    return value


snco_opts.option(
    '-e', '--exclude-contigs',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=str,
    default=None,
    callback=_parse_excl_contigs,
    help=('comma separated list of contigs to exclude. '
          'Default is a set of common organellar chrom names')
)
