import re
import logging
import click
import click_log

from .bam import DEFAULT_EXCLUDE_CONTIGS


log = logging.getLogger('snco')
click_log.basic_config(log)
verbosity = click_log.simple_verbosity_option(log)

_input_file_type = click.Path(exists=True, file_okay=True, dir_okay=False)
_input_dir_type = click.Path(exists=True, file_okay=False, dir_okay=True)
_output_file_type = click.Path(exists=False)

bam = click.argument(
    'bam-fn',
    required=True,
    nargs=1,
    type=_input_file_type
)

csl_dir = click.argument(
    'cellsnp-lite-dir',
    required=True,
    nargs=1,
    type=_input_dir_type
)

marker_json = click.argument(
    'marker-json-fn',
    required=True,
    nargs=1,
    type=_input_file_type
)

concat_json = click.argument(
    'json-fn',
    required=True,
    nargs=-1,
    type=_input_file_type,
)

pred_json = click.argument(
    'predict-json-fn',
    required=True,
    nargs=1,
    type=_input_file_type,
)

plot_pred_json = click.argument(
    'plot-json-fn',
    required=False,
    nargs=1,
    type=_input_file_type,
)

cell_barcode = click.argument(
    'cell-barcode',
    required=True,
    type=str,
    nargs=1,
)

output_json = click.option(
    '-o', '--output-json-fn',
    required=True,
    type=_output_file_type,
    help='Output JSON file name.'
)

output_tsv = click.option(
    '-o', '--output-tsv-fn',
    required=True,
    type=_output_file_type,
    help='Output TSV file name.'
)

output_fig = click.option(
    '-o', '--output-fig-fn',
    required=True,
    type=_output_file_type,
    help='Output figure file name (filetype automatically determined)'
)

cb_whitelist = click.option(
    '-c', '--cb-whitelist-fn',
    required=False,
    type=_input_file_type,
    help='Text file containing whitelisted cell barcodes, one per line'
)

chrom_sizes = click.option(
    '-s', '--chrom-sizes-fn',
    required=True,
    type=_input_file_type,
    help='chrom sizes or faidx file'
)

haplo_bed = click.option(
    '-b', '--haplo-bed-fn',
    required=True,
    type=_input_file_type,
    help='bed file (6 column) containing ground truth haplotype intervals to simulate'
)

bin_size = click.option(
    '-N', '--bin-size',
    required=False,
    type=click.IntRange(1000, 100_000),
    default=25_000,
    help='Bin size for marker distribution'
)

processes = click.option(
    '-p', '--processes',
    required=False,
    type=click.IntRange(min=1),
    default=1,
    help='number of cpu processes to use'
)

precision = click.option(
    '--output-precision',
    required=False,
    type=click.IntRange(1, 10),
    default=3,
    help='floating point precision in output files'
)


def _check_device(ctx, param, value):
    import torch
    if value == 'cpu':
        return torch.device(value)
    try:
        device_type, device_number = value.split(':')
        device_number = int(device_number)
    except ValueError as exc:
        raise click.BadParameter(
            'device format should be "cpu" or device:number e.g. "cuda:0"'
        ) from exc
    if device_type not in {'cuda', 'mps'}:
        raise click.BadParameter(f'unrecognised device type {device_type}')

    n_devices = getattr(torch, device_type).device_count()
    if not n_devices:
        raise click.BadParameter(f'no devices available of type {device_type}')
    if (device_number + 1) > n_devices:
        raise click.BadParameter(
            f'device number is too high, only {n_devices} device of type {device_type} avaiable'
        )
    return torch.device(value)


device = click.option(
    '-d', '--device',
    required=False,
    type=str,
    default='cpu',
    callback=_check_device,
    help='device to compute predictions on (default cpu)'
)

batch_size = click.option(
    '--batch-size',
    required=False,
    type=click.IntRange(1, 10_000),
    default=1_000,
    help='batch size for prediction. larger may be faster but use more memory'
)


def _replace_other_with_nonetype(ctx, param, value):
    if value == 'other':
        return None
    return value


seq_type = click.option(
    '-x', '--seq-type',
    required=False,
    type=click.Choice(['10x_rna', '10x_atac', 'takara_dna', 'other'], case_sensitive=False),
    default='other',
    callback=_replace_other_with_nonetype,
    help='presets for different sequencing data, see manual' # todo !!
)

cb_corr_method = click.option(
    '--cb-correction-method',
    required=False,
    type=click.Choice(['exact', '1mm'], case_sensitive=False),
    default='exact',
    help='method for correcting/matching cell barcodes to whitelist'
)


def _validate_bam_tag(ctx, param, value):
    if not len(value) == 2:
        raise ValueError(f'{param} is not of length 2')
    return value


cb_tag = click.option(
    '--cb-tag',
    required=False,
    type=str,
    default='CB',
    callback=_validate_bam_tag,
    help='bam file tag representing cell barcode'
)


def _set_default_umi_collapse_method(ctx, param, value):
    if value == 'none':
        if ctx.params['seq_type'] == '10x_rna':
            log.debug('setting default UMI dedup method to directional for 10x RNA data')
            return 'directional'
        return None
    return value


umi_collapse_method = click.option(
    '--umi-collapse-method',
    required=False,
    type=click.Choice(['exact', 'directional', 'none'], case_sensitive=False),
    default='none',
    callback=_set_default_umi_collapse_method,
    help='Method for deduplicating/collapsing UMIs'
)


def _override_umi_tag(ctx, param, value):
    if ctx.params['umi_collapse_method'] is None or ctx.params['umi_collapse_method'] == 'none':
        return None
    return _validate_bam_tag(ctx, param, value)


umi_tag = click.option(
    '--umi-tag',
    required=False,
    type=str,
    default='UB',
    callback=_override_umi_tag,
    help='bam file tag representing UMI'
)

hap_tag = click.option(
    '--hap-tag',
    required=False,
    type=str,
    default='ha',
    callback=_validate_bam_tag,
    help='bam file tag representing haplotype.'
)


def _parse_excl_contigs(ctx, param, value):
    if value is None:
        return DEFAULT_EXCLUDE_CONTIGS
    return set(value.split(',')) # todo: check allowed fasta header names


excl_contigs = click.option(
    '-e', '--exclude-contigs',
    required=False,
    type=str,
    default=None,
    callback=_parse_excl_contigs,
    help=('comma separated list of contigs to exclude. '
          'Default is a set of common organellar chrom names')
)

def _parse_merge_suffixes(ctx, param, value):
    if value is None:
        return value
    value = value.strip(',')
    if not re.match('^[a-zA-Z0-9,]+$', value):
        raise click.BadParameter('merge suffixes must be alphanumeric only')
    value = value.split(',')
    return value


merge_suffixes = click.option(
    '-M', '--merge-suffixes',
    required=False,
    type=str,
    default=None,
    callback=_parse_merge_suffixes,
    help=('comma separated list of suffixes to append to cell barcodes from '
          'each file. Must be alphanumeric with same length as no. json-fns')
)

bg_marker_rate = click.option(
    '--bg-marker-rate',
    required=False,
    type=click.FloatRange(0.0, 0.49),
    default=None,
    help='set uniform background marker rate. Default is to estimate per cell barcode from markers'
)

bg_window_size = click.option(
    '--bg-window-size',
    required=False,
    type=click.IntRange(100_000, 10_000_000),
    default=2_500_000,
    help='the size (in basepairs) of the convolution window used to estimate background marker rate'
)

nsim_per_samp = click.option(
    '--nsim-per-sample',
    required=False,
    type=click.IntRange(1, 1000),
    default=100,
    help='the number of randomly selected cell barcodes to simulate per ground truth sample'
)

min_markers = click.option(
    '--min-markers-per-cb',
    required=False,
    type=click.IntRange(0, 1000),
    default=0,
    help='minimum total number of markers per cb (cb with lower are filtered'
)

max_bin_count = click.option(
    '--max-bin-count',
    required=False,
    type=click.IntRange(5, 1000),
    default=20,
    help='maximum number of markers per cell per bin (higher values are thresholded)'
)

clean_bg = click.option(
    '--clean-bg/--no-clean-bg',
    required=False,
    default=True,
    help='whether to estimate and remove background markers'
)

mask_imbalanced = click.option(
    '--mask-imbalanced/--no-mask-imbalanced',
    required=False,
    default=True,
    help='whether to max bins with extreme allelic imbalance'
)

max_imbalance = click.option(
    '--max-marker-imbalance',
    required=False,
    type=click.FloatRange(0.6, 1.0),
    default=0.9,
    help=('maximum allowed global marker imbalance between haplotypes of a bin '
          '(higher values are masked)')
)


def _validate_seg_size(ctx, param, value):
    if value < ctx.params['bin_size']:
        raise click.BadParameter(f'{param} cannot be lower than --bin-size')
    return value


seg_size = click.option(
    '-r', '--segment-size',
    required=False,
    type=click.IntRange(100_000, 10_000_000),
    default=1_000_000,
    callback=_validate_seg_size,
    help='rfactor of the rigid HMM. Approximately controls minimum distance between COs'
)

term_seg_size = click.option(
    '-t', '--terminal-segment-size',
    required=False,
    type=click.IntRange(10_000, 1_000_000),
    default=50_000,
    callback=_validate_seg_size,
    help=('terminal rfactor of the rigid HMM. approx. controls min distance of COs '
          'from chromosome ends')
)

cm_per_mb = click.option(
    '-C', '--cm-per-mb',
    required=False,
    type=click.FloatRange(1.0, 20.0),
    default=4.5,
    help=('Approximate average centiMorgans per megabase. '
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

figsize = click.option(
    '--figsize',
    required=False,
    type=(click.IntRange(8, 30), click.IntRange(2, 10)),
    default=(18, 4),
    help='size of generated figure'
)

show_pred = click.option(
    '--show-pred/--no-pred',
    required=False,
    default=True,
    help='whether to draw predicted haplotype onto plot as shading'
)

show_co_num = click.option(
    '--show-co-num/--no-co-num',
    required=False,
    default=True,
    help='whether to annotate each chromosome with the no. of predicted COs'
)

max_yheight = click.option(
    '--max-yheight',
    required=False,
    type=click.IntRange(5, 1000),
    default=20,
    help='maximum number of markers per bin to plot (higher values are thresholded)'
)

ref_colour = max_yheight = click.option(
    '--ref-colour',
    required=False,
    type=str,
    default='#0072b2',
    help='hex colour to use for reference (hap1) markers'
)

alt_colour = max_yheight = click.option(
    '--alt-colour',
    required=False,
    type=str,
    default='#d55e00',
    help='hex colour to use for alternative (hap2) markers'
)
