import click
from .snco_opts import snco_opts


snco_opts.option(
    '--run-clean/--no-run-clean',
    subcommands=['bam2pred', 'csl2pred'],
    required=False,
    default=True,
    help='whether to run clean step in pipeline',
)


snco_opts.option(
    '--clean-bg/--no-clean-bg',
    subcommands=['clean', 'bam2pred', 'csl2pred'],
    required=False,
    default=True,
    help='whether to estimate and remove background markers'
)


snco_opts.option(
    '--bg-marker-rate',
    subcommands=['sim'],
    required=False,
    type=click.FloatRange(0.0, 0.49),
    default=None,
    help=('set uniform background marker rate for simulations. '
          'Default is to estimate per cell barcode from markers')
)


snco_opts.option(
    '--bg-window-size',
    subcommands=['sim', 'clean', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(100_000, 10_000_000),
    default=2_500_000,
    help='the size (in basepairs) of the convolution window used to estimate background marker rate'
)


snco_opts.option(
    '--max-frac-bg',
    subcommands=['clean', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(1e-3, 0.5),
    default=0.5,
    help='the estimated background marker rate to allow before filtering'
)


snco_opts.option(
    '--min-genotyping-prob', 'min_geno_prob',
    subcommands=['loadbam', 'loadcsl', 'clean', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(0.0, 1.0),
    default=0.9,
    help='for samples with genotyping, the minimum probability of the assigned genotype'
)



snco_opts.option(
    '--nsim-per-sample',
    subcommands=['sim'],
    required=False,
    type=click.IntRange(1, 1000),
    default=10,
    help='the number of randomly selected cell barcodes to simulate per ground truth sample'
)


snco_opts.option(
    '--min-markers-per-cb',
    subcommands=['loadbam', 'loadcsl', 'clean', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(0, 100000),
    default=100,
    help='minimum total number of markers per cb (cb with lower are filtered)'
)


snco_opts.option(
    '--min-markers-per-chrom',
    subcommands=['loadbam', 'loadcsl', 'clean', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(1, 100000),
    default=20,
    help='minimum total number of markers per chrom, per cb (cb with lower are filtered)'
)


snco_opts.option(
    '--max-bin-count',
    subcommands=['clean', 'bam2pred', 'csl2pred'],
    type=click.IntRange(5, 1000),
    default=None,
    help='maximum number of markers per cell per bin (higher values are thresholded)'
)


snco_opts.option(
    '--mask-imbalanced/--no-mask-imbalanced',
    subcommands=['clean', 'bam2pred', 'csl2pred'],
    required=False,
    default=True,
    help='whether to mask bins with extreme allelic imbalance'
)


snco_opts.option(
    '--max-marker-imbalance',
    subcommands=['clean', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(0.6, 1.0),
    default=0.75,
    help=('maximum allowed global marker imbalance between haplotypes of a bin '
          '(higher values are masked)')
)


snco_opts.option(
    '--clean-by-genotype/--clean-all', 'apply_per_geno',
    subcommands=['clean', 'bam2pred', 'csl2pred'],
    required=False,
    default=False,
    help='Whether to apply cleaning functions to each genotype separately'
)
