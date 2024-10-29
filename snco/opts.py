import re
import logging
import click

from .logger import click_logger
from .bam import DEFAULT_EXCLUDE_CONTIGS

log = logging.getLogger('snco')


class OptionRegistry:

    def __init__(self, subcommands):
        self.subcommands = subcommands
        self.register = {}

    def register_param(self, click_param, args, kwargs):
        subcommands = kwargs.pop('subcommands', None)
        if subcommands is None:
            raise ValueError('need to provide at least one subcommand to attach option to')

        name = args[-1].split('/')[0].strip('-').replace('-', '_')
        opt = click_param(*args, **kwargs)

        for sc in subcommands:
            if sc not in self.subcommands:
                raise ValueError(f'subcommand {sc} is not pre-registered')
            if sc not in self.register:
                self.register[sc] = {}
            self.register[sc][name] = opt

    def argument(self, *args, **kwargs):
        self.register_param(click.argument, args, kwargs)

    def option(self, *args, **kwargs):
        self.register_param(click.option, args, kwargs)

    def __call__(self, subcommand):
        def _apply_options(func):
            for option in reversed(self.register[subcommand].values()):
                func = option(func)
            return func
        return _apply_options

    def get_kwarg_subset(self, subcommand, kwargs):
        sc_options = list(self.register[subcommand].keys())
        return {kw: val for kw, val in kwargs.items() if kw in sc_options}


snco_opts = OptionRegistry(
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred',
                 'sim', 'concat', 'clean', 'predict',
                 'stats', 'plot']
)


def _log_callback(ctx, param, value):
    log.debug(f'set parameter {param.name} to {value}')
    return value


_input_file_type = click.Path(exists=True, file_okay=True, dir_okay=False)
_input_dir_type = click.Path(exists=True, file_okay=False, dir_okay=True)
_output_file_type = click.Path(exists=False)


snco_opts.argument(
    'bam-fn',
    subcommands=['loadbam', 'bam2pred'],
    required=True,
    nargs=1,
    type=_input_file_type,
    callback=_log_callback,
)

snco_opts.argument(
    'cellsnp-lite-dir',
    subcommands=['loadcsl', 'csl2pred'],
    required=True,
    nargs=1,
    type=_input_dir_type,
    callback=_log_callback,
)

snco_opts.argument(
    'marker-json-fn',
    subcommands=['sim', 'clean', 'predict', 'stats', 'plot'],
    required=True,
    nargs=1,
    type=_input_file_type,
    callback=_log_callback,
)

snco_opts.argument(
    'json-fn',
    subcommands=['concat'],
    required=True,
    nargs=-1,
    type=_input_file_type,
    callback=_log_callback,
)

snco_opts.argument(
    'pred-json-fn',
    subcommands=['stats'],
    required=True,
    nargs=1,
    type=_input_file_type,
    callback=_log_callback,
)

snco_opts.argument(
    'pred-json-fn',
    subcommands=['plot'],
    required=False,
    nargs=1,
    type=_input_file_type,
    callback=_log_callback,
)

snco_opts.argument(
    'cell-barcode',
    subcommands=['plot'],
    required=True,
    type=str,
    nargs=1,
    callback=_log_callback,
)

snco_opts.option(
    '-o', '--output-prefix',
    subcommands=['bam2pred', 'csl2pred'],
    required=True,
    type=_output_file_type,
    callback=_log_callback,
    help='Output prefix'
)

snco_opts.option(
    '-o', '--output-json-fn',
    subcommands=['loadbam', 'loadcsl', 'sim', 'concat', 'clean', 'predict'],
    required=True,
    type=_output_file_type,
    callback=_log_callback,
    help='Output JSON file name.'
)

snco_opts.option(
    '-o', '--output-tsv-fn',
    subcommands=['stats'],
    required=True,
    type=_output_file_type,
    callback=_log_callback,
    help='Output TSV file name.'
)

snco_opts.option(
    '-o', '--output-fig-fn',
    subcommands=['plot'],
    required=True,
    type=_output_file_type,
    callback=_log_callback,
    help='Output figure file name (filetype automatically determined)'
)

snco_opts.option(
    '-c', '--cb-whitelist-fn',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred',
                 'sim', 'clean', 'predict', 'stats'],
    required=False,
    type=_input_file_type,
    callback=_log_callback,
    help='Text file containing whitelisted cell barcodes, one per line'
)

snco_opts.option(
    '-s', '--chrom-sizes-fn',
    subcommands=['loadcsl', 'csl2pred'],
    required=True,
    type=_input_file_type,
    callback=_log_callback,
    help='chrom sizes or faidx file'
)

snco_opts.option(
    '-b', '--haplo-bed-fn',
    subcommands=['sim'],
    required=True,
    type=_input_file_type,
    callback=_log_callback,
    help='bed file (6 column) containing ground truth haplotype intervals to simulate'
)

snco_opts.option(
    '-N', '--bin-size',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred',
                 'sim', 'clean', 'predict', 'stats'], # todo: why do these cmds need this again?
    required=False,
    type=click.IntRange(1000, 100_000),
    default=25_000,
    callback=_log_callback,
    help='Bin size for marker distribution'
)


def _replace_other_with_nonetype(ctx, param, value):
    if value == 'other':
        return None
    return _log_callback(ctx, param, value)


snco_opts.option(
    '-x', '--seq-type',
    required=False,
    subcommands=['loadbam', 'bam2pred'],
    type=click.Choice(
        ['10x_rna', '10x_atac', 'takara_dna', 'other'],
        case_sensitive=False
    ),
    default='other',
    callback=_replace_other_with_nonetype,
    help='presets for different sequencing data, see manual' # todo !!
)

snco_opts.option(
    '--cb-correction-method',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=click.Choice(['exact', '1mm'], case_sensitive=False),
    default='exact',
    callback=_log_callback,
    help='method for correcting/matching cell barcodes to whitelist'
)


def _validate_bam_tag(ctx, param, value):
    if not len(value) == 2:
        raise ValueError(f'{param} is not of length 2')
    return _log_callback(ctx, param, value)


snco_opts.option(
    '--cb-tag',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=str,
    default='CB',
    callback=_validate_bam_tag,
    help='bam file tag representing cell barcode'
)


def _set_default_umi_collapse_method(ctx, param, value):
    if value == 'none':
        if ctx.params['seq_type'] == '10x_rna':
            log.info('setting default UMI dedup method to directional for 10x RNA data')
            value = 'directional'
        else:
            value = None
    return _log_callback(ctx, param, value)


snco_opts.option(
    '--umi-collapse-method',
    subcommands=['loadbam', 'bam2pred'],
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


snco_opts.option(
    '--umi-tag',
    subcommands=['loadbam', 'bam2pred'],
    required=False,
    type=str,
    default='UB',
    callback=_override_umi_tag,
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


def _parse_excl_contigs(ctx, param, value):
    if value is None:
        value = DEFAULT_EXCLUDE_CONTIGS
    else:
        value = set(value.split(',')) # todo: check allowed fasta header names
    return _log_callback(ctx, param, value)


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


def _parse_merge_suffixes(ctx, param, value):
    if value is not None:
        value = value.strip(',')
        if not re.match('^[a-zA-Z0-9,]+$', value):
            raise click.BadParameter('merge suffixes must be alphanumeric only')
        value = value.split(',')
    return _log_callback(ctx, param, value)


snco_opts.option(
    '-M', '--merge-suffixes',
    subcommands=['concat'],
    required=False,
    type=str,
    default=None,
    callback=_parse_merge_suffixes,
    help=('comma separated list of suffixes to append to cell barcodes from '
          'each file. Must be alphanumeric with same length as no. json-fns')
)


snco_opts.option(
    '--run-clean/--no-run-clean',
    subcommands=['bam2pred', 'csl2pred'],
    required=False,
    default=True,
    callback=_log_callback,
    help='whether to run clean step in pipeline',
)


snco_opts.option(
    '--bg-marker-rate',
    subcommands=['sim'],
    required=False,
    type=click.FloatRange(0.0, 0.49),
    default=None,
    callback=_log_callback,
    help='set uniform background marker rate. Default is to estimate per cell barcode from markers'
)

snco_opts.option(
    '--bg-window-size',
    subcommands=['sim', 'clean', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(100_000, 10_000_000),
    default=2_500_000,
    callback=_log_callback,
    help='the size (in basepairs) of the convolution window used to estimate background marker rate'
)

snco_opts.option(
    '--nsim-per-sample',
    subcommands=['sim'],
    required=False,
    type=click.IntRange(1, 1000),
    default=100,
    callback=_log_callback,
    help='the number of randomly selected cell barcodes to simulate per ground truth sample'
)

snco_opts.option(
    '--min-markers-per-cb',
    subcommands=['clean', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(0, 1000),
    default=0,
    callback=_log_callback,
    help='minimum total number of markers per cb (cb with lower are filtered'
)

snco_opts.option(
    '--max-bin-count',
    subcommands=['clean', 'bam2pred', 'csl2pred'],
    type=click.IntRange(5, 1000),
    default=20,
    callback=_log_callback,
    help='maximum number of markers per cell per bin (higher values are thresholded)'
)

snco_opts.option(
    '--clean-bg/--no-clean-bg',
    subcommands=['clean', 'bam2pred', 'csl2pred'],
    required=False,
    default=True,
    callback=_log_callback,
    help='whether to estimate and remove background markers'
)

snco_opts.option(
    '--mask-imbalanced/--no-mask-imbalanced',
    subcommands=['clean', 'bam2pred', 'csl2pred'],
    required=False,
    default=True,
    callback=_log_callback,
    help='whether to max bins with extreme allelic imbalance'
)

snco_opts.option(
    '--max-marker-imbalance',
    subcommands=['clean', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(0.6, 1.0),
    default=0.9,
    callback=_log_callback,
    help=('maximum allowed global marker imbalance between haplotypes of a bin '
          '(higher values are masked)')
)


def _validate_seg_size(ctx, param, value):
    if value < ctx.params['bin_size']:
        raise click.BadParameter(f'{param} cannot be lower than --bin-size')
    return _log_callback(ctx, param, value)


snco_opts.option(
    '-r', '--segment-size',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(100_000, 10_000_000),
    default=1_000_000,
    callback=_validate_seg_size,
    help='rfactor of the rigid HMM. Approximately controls minimum distance between COs'
)

snco_opts.option(
    '-t', '--terminal-segment-size',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(10_000, 1_000_000),
    default=50_000,
    callback=_validate_seg_size,
    help=('terminal rfactor of the rigid HMM. approx. controls min distance of COs '
          'from chromosome ends')
)

snco_opts.option(
    '-C', '--cm-per-mb',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(1.0, 20.0),
    default=4.5,
    callback=_log_callback,
    help=('Approximate average centiMorgans per megabase. '
          'Used to parameterise the rigid HMM transitions')
)

snco_opts.option(
    '--model-lambdas',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=(click.FloatRange(1e-5, 5.0), click.FloatRange(1e-5, 5.0)),
    default=None,
    callback=_log_callback,
    help=('optional lambda parameters for foreground and background Poisson distributions of '
          'model. Default is to fit to the data')
)

snco_opts.option(
    '--figsize',
    subcommands=['plot'],
    required=False,
    type=(click.IntRange(8, 30), click.IntRange(2, 10)),
    default=(18, 4),
    callback=_log_callback,
    help='size of generated figure'
)

snco_opts.option(
    '--show-pred/--no-pred',
    subcommands=['plot'],
    required=False,
    default=True,
    callback=_log_callback,
    help='whether to draw predicted haplotype onto plot as shading'
)

snco_opts.option(
    '--show-co-num/--no-co-num',
    subcommands=['plot'],
    required=False,
    default=True,
    callback=_log_callback,
    help='whether to annotate each chromosome with the no. of predicted COs'
)

snco_opts.option(
    '--max-yheight',
    subcommands=['plot'],
    required=False,
    type=click.IntRange(5, 1000),
    default=20,
    callback=_log_callback,
    help='maximum number of markers per bin to plot (higher values are thresholded)'
)

snco_opts.option(
    '--ref-colour',
    subcommands=['plot'],
    required=False,
    type=str,
    default='#0072b2',
    callback=_log_callback,
    help='hex colour to use for reference (hap1) markers'
)

snco_opts.option(
    '--alt-colour',
    subcommands=['plot'],
    required=False,
    type=str,
    default='#d55e00',
    callback=_log_callback,
    help='hex colour to use for alternative (hap2) markers'
)

snco_opts.option(
    '-p', '--processes',
    subcommands=['loadbam', 'bam2pred', 'predict'],
    required=False,
    type=click.IntRange(min=1),
    default=1,
    callback=_log_callback,
    help='number of cpu processes to use'
)

snco_opts.option(
    '--output-precision',
    subcommands=['predict', 'bam2pred', 'csl2pred', 'stats'],
    required=False,
    type=click.IntRange(1, 10),
    default=3,
    callback=_log_callback,
    help='floating point precision in output files'
)


def _check_device(ctx, param, value):
    import torch
    if value == 'cpu':
        return _log_callback(ctx, param, torch.device(value))
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
    return _log_callback(ctx, param, torch.device(value))


snco_opts.option(
    '-d', '--device',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=str,
    default='cpu',
    callback=_check_device,
    help='device to compute predictions on (default cpu)'
)

snco_opts.option(
    '--batch-size',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(1, 10_000),
    default=1_000,
    callback=_log_callback,
    help='batch size for prediction. larger may be faster but use more memory'
)


snco_opts.option(
    '-v', '--verbosity',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred',
                 'sim', 'concat', 'clean', 'predict', 'stats', 'plot'],
    required=False,
    expose_value=False,
    metavar='LVL',
    type=click.Choice(
        ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        case_sensitive=False
    ),
    default='info',
    callback=click_logger,
    help='Logging level, either debug, info, warning, error or critical'
)
