import click


bam = click.argument('bam-fn', required=True, nargs=1)
csl_dir = click.argument('cellsnp-lite-dir', required=True, nargs=1)
marker_json = click.argument('marker-json-fn', required=True, nargs=1)
concat_marker_json = click.argument('marker-json-fn', required=True, nargs=-1)
pred_json = click.argument('predict-json-fn', required=True, nargs=1)
output_json = click.option(
    '-o', '--output-json-fn',
    required=True,
    type=str,
    help='Output JSON file name.'
)
output_tsv = click.option(
    '-o', '--output-tsv-fn',
    required=True,
    type=str,
    help='Output TSV file name.'
)
cb_whitelist = click.option(
    '-c', '--cb-whitelist-fn',
    required=False,
    type=str,
    help='Text file containing whitelisted cell barcodes, one per line'
)
chrom_sizes = click.option(
    '-s', '--chrom-sizes-fn',
    required=True,
    type=str,
    help='chrom sizes or faidx file'
)
haplo_bed = click.option(
    '-b', '--haplo-bed-fn',
    required=True,
    type=str,
    help='bed file (6 column) containing ground truth haplotype intervals to simulate'
)
bin_size = click.option(
    '-N', '--bin-size',
    required=False,
    default=25_000,
    help='Bin size for marker distribution'
)
processes = click.option('-p', '--processes', required=False, default=1)
precision = click.option(
    '--output-precision',
    required=False,
    type=click.IntRange(1, 10),
    default=2,
    help='floating point precision in output files'
)
cb_corr_method = click.option(
    '--cb-correction-method',
    required=False,
    type=click.Choice(['exact', '1mm'], case_sensitive=False),
    default='exact',
    help='method for correcting/matching cell barcodes to whitelist'
)
cb_tag = click.option(
    '--cb-tag',
    required=False,
    type=str,
    default='CB',
    help='bam file tag representing cell barcode'
)
umi_collapse_method = click.option(
    '--umi-collapse-method',
    required=False,
    type=click.Choice(['exact', 'directional', 'none'], case_sensitive=False),
    default='directional',
    callback=lambda c, p, v: None if v == 'none' else v, # todo
    help='Method for deduplicating/collapsing UMIs'
)
umi_tag = click.option(
    '--umi-tag',
    required=False,
    type=str,
    default='UB',
    callback=lambda c, p, v: None if c.params['umi_collapse_method'] is None else v, # todo
    help='bam file tag representing UMI'
)
hap_tag = click.option(
    '--hap-tag',
    required=False,
    type=str,
    default='ha',
    help='bam file tag representing haplotype.'
)
excl_contigs = click.option(
    '-e', '--exclude-contigs',
    required=False,
    default=None,
    type=str,
    callback=lambda c, p, v: set(v.split(',')) if v is not None else None, # todo
    help=('comma separated list of contigs to exclude. '
          'Default is a set of common organellar chrom names')
)
bg_marker_rate = click.option(
    '--bg-marker-rate',
    required=False,
    type=float,
    default='auto',
    help='set uniform background marker rate. Default is to estimate per cell barcode from markers'
)
bg_window_size = click.option(
    '--bg-window-size',
    required=False,
    type=int,
    default=2_500_000,
    help='the size (in basepairs) of the convolution window used to estimate background marker rate'
)
nsim_per_samp = click.option(
    '--nsim-per-sample',
    required=False,
    type=int,
    default=100,
    help='the number of randomly selected cell barcodes to simulate per ground truth sample'
)
max_bin_count = click.option(
    '--max-bin-count',
    required=False,
    type=click.IntRange(5, 1000),
    default=20,
    help='maximum number of markers per cell per bin (higher values are thresholded)'
)
max_imbalance = click.option(
    '--max-marker-imbalance',
    required=False,
    type=click.FloatRange(0.6, 1.0),
    default=0.9,
    help=('maximum allowed global marker imbalance between haplotypes of a bin '
          '(higher values are masked)')
)
seg_size = click.option(
    '-r', '--segment-size',
    required=False,
    type=click.IntRange(100_000, 10_000_000), # todo: callback check against binsize/term rfactor
    default=1_000_000,
    help='rfactor of the rigid HMM. Approximately controls minimum distance between COs'
)
term_seg_size = click.option(
    '-t', '--terminal-segment-size',
    required=False,
    type=click.IntRange(10_000, 1_000_000),
    default=50_000,
    help=('terminal rfactor of the rigid HMM. approx. controls min distance of COs '
          'from chromosome ends')
)
cm_per_mb = click.option(
    '-C', '--cm-per-mb',
    required=False,
    type=click.FloatRange(1.0, 10.0),
    default=4.5,
    help=('Estimated average centiMorgans per megabase. '
          'Used to parameterise the rigid HMM transitions')
)
model_lambdas = click.option(
    '--model-lambdas',
    required=False,
    type=(click.FloatRange(1e-5, 5.0), click.FloatRange(1e-5, 5.0)),
    default=None,
    help=('optional lambda parameters for foreground and background Poisson distributions of '
          'model. Default is to fit to the data')
)
