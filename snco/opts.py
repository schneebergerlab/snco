import re
import logging
import click

import numpy as np

from .logger import click_logger
from .bam import DEFAULT_EXCLUDE_CONTIGS, get_ha_samples

log = logging.getLogger('snco')
DEFAULT_RANDOM_SEED = 101


class OptionRegistry:

    def __init__(self, subcommands):
        self.subcommands = subcommands
        self.register = {}
        self.callback_register = {}

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

    def callback(self, subcommands):
        def _register_callback(callback):
            for sc in subcommands:
                if sc not in self.subcommands:
                    raise ValueError(f'subcommand {sc} is not pre-registered')
                if sc not in self.callback_register:
                    self.callback_register[sc] = []
                self.callback_register[sc].append(callback)
        return _register_callback

    def __call__(self, subcommand):
        def _apply_options(func):
            for callback in reversed(self.callback_register.get(subcommand, [])):
                func = callback(func)
            for option in reversed(self.register[subcommand].values()):
                func = option(func)
            return func
        return _apply_options

    def get_kwarg_subset(self, subcommand, kwargs):
        sc_options = list(self.register[subcommand].keys())
        return {kw: val for kw, val in kwargs.items() if kw in sc_options}


snco_opts = OptionRegistry(
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred',
                 'sim', 'concat', 'clean', 'predict', 'bc1predict',
                 'doublet', 'stats', 'plot']
)


@snco_opts.callback(['loadbam', 'bam2pred'])
def validate_loadbam_input(func):
    '''decorator to validate the input of the loadbam command'''
    def _validate(**kwargs):
        seq_type = kwargs.get('seq_type')
        if kwargs.get('cb_correction_method') == 'auto':
            cb_tag = kwargs.get('cb_tag')
            method = 'exact' if cb_tag in ('CB', 'RG') else '1mm'
            log.info(f"setting CB correction method to '{method}' for data with CB tag '{cb_tag}'")
        if kwargs.get('umi_collapse_method') == 'auto':
            if seq_type in ('10x_rna', 'bd_rna', 'bd_atac'):
                umi_tag = kwargs.get('umi_tag')
                method = 'exact' if umi_tag == 'UB' else 'directional'
                log.info(f"setting UMI dedup method to '{method}' for 10x RNA or BD data with UMI tag '{umi_tag}'")
                kwargs['umi_collapse_method'] = method
            elif seq_type in ('10x_atac', 'takara_dna', 'wgs'):
                log.info('turning off UMI processing for 10x/Takara ATAC data, or WGS data')
                kwargs['umi_tag'] = None
                kwargs['umi_collapse_method'] = None
            elif seq_type is None:
                log.error("'-x' / '--seq-type' must be specified when '--umi-collapse-method' is set to 'auto'")
        elif kwargs.get('umi_collapse_method') == 'none':
            log.info('turning off UMI processing')
            kwargs['umi_tag'] = None
            kwargs['umi_collapse_method'] = None
        if kwargs.get('cb_tag') == 'CB' and kwargs.get('cb_correction_method') == '1mm':
            log.warn("'--cb-tag' is set to 'CB', which usually indicates pre-corrected barcodes, but "
                     "'--cb-correction-method' is set to '1mm'. This may lead to overcorrection")
        if kwargs.get('umi_tag') == 'UB' and kwargs.get('umi_collapse_method') == 'directional':
            log.warn("'--umi-tag' is set to 'UB', which usually indicates pre-corrected UMIs, but "
                     "'--cb-correction-method' is set to 'directional'. This may lead to overcorrection")
        if kwargs.get('run_genotype') and kwargs.get('hap_tag_type') == "star_diploid":
            log.error('--hap-tag-type must be "multi_haplotype" when --genotype is switched on')
        if not kwargs.get('run_genotype') and kwargs.get('hap_tag_type') == "multi_haplotype":
            crossing_combinations = kwargs.get('genotype_crossing_combinations')
            if crossing_combinations is not None:
                if len(crossing_combinations) != 1:
                    log.error('when --genotype is switched off only one --crossing-combinations can be provided')
            else:
                genotypes = get_ha_samples(kwargs['bam_fn'])
                if len(genotypes) != 2:
                    log.error(
                        'when --genotype is switched off and no --crossing-combinations are provided, '
                        'the bam file can only contain two haplotypes'
                    )
        return func(**kwargs)
    return _validate


@snco_opts.callback(['loadcsl', 'csl2pred'])
def validate_loadcsl_input(func):
    '''decorator to validate the input of the loadcsl command'''
    def _validate(**kwargs):
        if kwargs.get('run_genotype') and kwargs.get('genotype_vcf_fn') is None:
            log.error('--genotype-vcf-fn must be provided when --haplotype is switched on')
        return func(**kwargs)
    return _validate



@snco_opts.callback(['predict', 'bc1predict', 'bam2pred', 'csl2pred'])
def validate_pred_input(func):
    '''decorator to validate the input of the predict command'''
    def _validate(**kwargs):
        bin_size = kwargs.get('bin_size')
        seg_size = kwargs.get('segment_size')
        tseg_size = kwargs.get('terminal_segment_size')
        if seg_size < bin_size:
            log.error("'-R' / '--segment-size' cannot be less than '-N' / '--bin-size'")
        if tseg_size < bin_size:
            log.error("'-t' / '--terminal-segment-size' cannot be less than '-N' / '--bin-size'")
        return func(**kwargs)
    return _validate


def log_parameters(func):
    '''decorator which logs the final values of all parameters used to execute snco'''
    def _log(**kwargs):
        for param, value in kwargs.items():
            log.debug(f'set parameter {param} to {value}')
        return func(**kwargs)
    return _log

snco_opts.callback(['loadbam', 'loadcsl', 'bam2pred', 'csl2pred',
                     'sim', 'concat', 'clean', 'predict', 'bc1predict',
                     'doublet', 'stats', 'plot'])(log_parameters)


_input_file_type = click.Path(exists=True, file_okay=True, dir_okay=False)
_input_dir_type = click.Path(exists=True, file_okay=False, dir_okay=True)
_output_file_type = click.Path(exists=False)


snco_opts.argument(
    'bam-fn',
    subcommands=['loadbam', 'bam2pred'],
    required=True,
    nargs=1,
    type=_input_file_type,
)


snco_opts.argument(
    'cellsnp-lite-dir',
    subcommands=['loadcsl', 'csl2pred'],
    required=True,
    nargs=1,
    type=_input_dir_type,
)


snco_opts.argument(
    'marker-json-fn',
    subcommands=['sim', 'clean', 'predict', 'bc1predict', 'doublet', 'stats', 'plot'],
    required=True,
    nargs=1,
    type=_input_file_type,
)


snco_opts.argument(
    'json-fn',
    subcommands=['concat'],
    required=True,
    nargs=-1,
    type=_input_file_type,
)


snco_opts.argument(
    'pred-json-fn',
    subcommands=['doublet', 'stats'],
    required=True,
    nargs=1,
    type=_input_file_type,
)


snco_opts.argument(
    'pred-json-fn',
    subcommands=['plot'],
    required=False,
    nargs=1,
    type=_input_file_type,
)


snco_opts.argument(
    'cell-barcode',
    subcommands=['plot'],
    required=False,
    type=str,
    nargs=1,
)


snco_opts.option(
    '-o', '--output-prefix',
    subcommands=['bam2pred', 'csl2pred'],
    required=True,
    type=_output_file_type,
    help='Output prefix'
)


snco_opts.option(
    '-o', '--output-json-fn',
    subcommands=['loadbam', 'loadcsl', 'sim', 'concat', 'clean', 'predict', 'bc1predict', 'doublet'],
    required=True,
    type=_output_file_type,
    help='Output JSON file name.'
)


snco_opts.option(
    '-o', '--output-tsv-fn',
    subcommands=['stats'],
    required=True,
    type=_output_file_type,
    help='Output TSV file name.'
)


snco_opts.option(
    '-o', '--output-fig-fn',
    subcommands=['plot'],
    required=False,
    type=_output_file_type,
    default=None,
    help='Output figure file name (filetype automatically determined)'
)


snco_opts.option(
    '-c', '--cb-whitelist-fn',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred',
                 'sim', 'clean', 'predict', 'bc1predict', 'doublet', 'stats', 'plot'],
    required=False,
    type=_input_file_type,
    help='Text file containing whitelisted cell barcodes, one per line'
)


snco_opts.option(
    '-s', '--chrom-sizes-fn',
    subcommands=['loadcsl', 'csl2pred'],
    required=True,
    type=_input_file_type,
    help='chrom sizes or faidx file'
)


snco_opts.option(
    '-g', '--genotype-vcf-fn',
    subcommands=['loadcsl', 'csl2pred'],
    required=False,
    type=_input_file_type,
    help='VCF file containing all the parental genotype SNPs'
)


snco_opts.option(
    '-m', '--mask-bed-fn',
    subcommands=['clean', 'bam2pred', 'csl2pred', 'bc1predict'],
    required=False,
    type=_input_file_type,
    default=None,
    help='A bed file of regions to mask when cleaning data'
)


snco_opts.option(
    '-g', '--ground-truth-fn',
    subcommands=['sim'],
    required=True,
    type=_input_file_type,
    help='pred json or bed file (6 column) containing ground truth haplotype intervals to simulate'
)

snco_opts.option(
    '-N', '--bin-size',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred',
                 'sim', 'clean', 'predict', 'bc1predict',
                 'doublet', 'stats'],
    required=False,
    type=click.IntRange(1000, 100_000),
    default=25_000,
    help='Bin size for marker distribution'
)


def _replace_other_with_nonetype(ctx, param, value):
    if value == 'other':
        return None
    return value


snco_opts.option(
    '-x', '--seq-type',
    required=False,
    subcommands=['loadbam', 'bam2pred'],
    type=click.Choice(
        ['10x_rna', '10x_atac', 'bd_rna', 'bd_atac', 'takara_dna', 'wgs', 'other'],
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
    subcommands=['loadcsl', 'csl2pred'],
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


def _parse_merge_suffixes(ctx, param, value):
    if value is not None:
        value = value.strip(',')
        if not re.match('^[a-zA-Z0-9,]+$', value):
            log.error('merge suffixes must be alphanumeric only')
        value = value.split(',')
    return value


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
    subcommands=['loadbam', 'loadcsl', 'clean', 'bc1predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(0, 100000),
    default=100,
    help='minimum total number of markers per cb (cb with lower are filtered)'
)


snco_opts.option(
    '--min-markers-per-chrom',
    subcommands=['loadbam', 'loadcsl', 'clean', 'bc1predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(1, 100000),
    default=20,
    help='minimum total number of markers per chrom, per cb (cb with lower are filtered)'
)


snco_opts.option(
    '--max-bin-count',
    subcommands=['clean', 'bam2pred', 'csl2pred'],
    type=click.IntRange(5, 1000),
    default=20,
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


snco_opts.option(
    '-R', '--segment-size',
    subcommands=['predict', 'bc1predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(100_000, 10_000_000),
    default=1_000_000,
    help='rfactor of the rigid HMM. Approximately controls minimum distance between COs'
)


snco_opts.option(
    '-t', '--terminal-segment-size',
    subcommands=['predict', 'bc1predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(10_000, 1_000_000),
    default=50_000,
    help=('terminal rfactor of the rigid HMM. approx. controls min distance of COs '
          'from chromosome ends')
)


snco_opts.option(
    '-C', '--cm-per-mb',
    subcommands=['predict', 'bc1predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(1.0, 20.0),
    default=4.5,
    help=('Approximate average centiMorgans per megabase. '
          'Used to parameterise the rigid HMM transitions')
)


snco_opts.option(
    '--model-lambdas',
    subcommands=['predict', 'bc1predict', 'bam2pred', 'csl2pred'],
    required=False,
    type=(click.FloatRange(1e-5, 10.0), click.FloatRange(1e-5, 10.0)),
    default=None,
    help=('optional lambda parameters for foreground and background Poisson distributions of '
          'model. Default is to fit to the data')
)


snco_opts.option(
    '--empty-fraction',
    subcommands=['bc1predict'],
    required=False,
    type=click.FloatRange(0, 1),
    default=None,
    help=('optional lambda parameters for foreground and background Poisson distributions of '
          'model. Default is to fit to the data')
)

snco_opts.option(
    '--predict-doublets/--no-predict-doublets',
    subcommands=['predict', 'bam2pred', 'csl2pred'],
    required=False,
    default=True,
    help='whether to use synthetic doublet scoring to predict likely doublets in the data'
)


snco_opts.option(
    '--n-doublets',
    subcommands=['sim', 'predict', 'doublet', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(0.0, 10_000),
    default=0.25,
    metavar='INTEGER OR FLOAT',
    help=('number of doublets to simulate. If >1, treated as integer number of doublets. '
          'If <1, treated as a fraction of the total dataset')
)


snco_opts.option(
    '--k-neighbours',
    subcommands=['predict', 'doublet', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(0.01, 10_000),
    default=0.25,
    metavar='INTEGER OR FLOAT',
    help=('K neighbours to use for doublet calling. If >1, treated as integer k neighbours. '
          'If <1, treated as a fraction of --n-doublets')
)


snco_opts.option(
    '--generate-stats/--no-stats',
    subcommands=['predict', 'bc1predict', 'doublet', 'bam2pred', 'csl2pred'],
    required=False,
    default=True,
    help='whether to use synthetic doublet scoring to predict likely doublets in the data'
)


snco_opts.option(
    '-M', '--nco-min-prob-change',
    subcommands=['stats', 'predict', 'plot', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.FloatRange(0.0, 1.0),
    default=2.5e-3,
    help=('Minimum probability change to take into account when estimating stats e.g. number of '
          'crossovers from the haplotype predictions.')
)


snco_opts.option(
    '--plot-type',
    subcommands=['plot'],
    required=False,
    type=click.Choice(['markerplot', 'recombination'], case_sensitive=False),
    default='markerplot',
    help='type of plot to create'
)


snco_opts.option(
    '--figsize',
    subcommands=['plot'],
    required=False,
    type=(click.IntRange(8, 30), click.IntRange(2, 10)),
    default=(18, 4),
    help='size of generated figure'
)


snco_opts.option(
    '--display/--no-display', 'display_plot',
    subcommands=['plot'],
    required=False,
    default=True,
    help='whether to display the plot to screen (requires interactive mode)'
)


snco_opts.option(
    '--show-pred/--no-pred',
    subcommands=['plot'],
    required=False,
    default=True,
    help='whether to draw predicted haplotype onto plot as shading (markerplot)'
)


snco_opts.option(
    '--show-co-num/--no-co-num',
    subcommands=['plot'],
    required=False,
    default=True,
    help='whether to annotate each chromosome with the no. of predicted COs (markerplot)'
)


snco_opts.option(
    '--show-gt/--no-gt',
    subcommands=['plot'],
    required=False,
    default=True,
    help='when ground truth is present (i.e. sim data), show expected CO positions (markerplot)'
)


snco_opts.option(
    '--max-yheight',
    subcommands=['plot'],
    required=False,
    type=click.IntRange(5, 1000),
    default=20,
    help='maximum number of markers per bin to plot, higher values are thresholded (markerplot)'
)


snco_opts.option(
    '--window-size',
    subcommands=['plot'],
    required=False,
    type=click.IntRange(25_000, 10_000_000),
    default=1_000_000,
    help='Rolling window size for calculating recombination landscape (recombination)'
)


snco_opts.option(
    '--bootstraps', 'nboots',
    subcommands=['plot'],
    required=False,
    type=click.IntRange(1, 10_000),
    default=100,
    help='Number of random subsamples for calculating recombination landscape (recombination)'
)


snco_opts.option(
    '--confidence-intervals',
    subcommands=['plot'],
    required=False,
    type=click.FloatRange(50, 100),
    default=95,
    help='Percentile to use for drawing confidence intervals (recombination)'
)


snco_opts.option(
    '--ref-colour',
    subcommands=['plot'],
    required=False,
    type=str,
    default='#0072b2',
    help='hex colour to use for reference (hap1) markers (also for recombination landscape)'
)


snco_opts.option(
    '--alt-colour',
    subcommands=['plot'],
    required=False,
    type=str,
    default='#d55e00',
    help='hex colour to use for alternative (hap2) markers (markerplot)'
)


snco_opts.option(
    '-p', '--processes',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred', 'predict', 'bc1predict', 'doublet'],
    required=False,
    type=click.IntRange(min=1),
    default=1,
    help='number of cpu processes to use'
)


snco_opts.option(
    '--output-precision',
    subcommands=['predict', 'bc1predict', 'doublet', 'bam2pred', 'csl2pred', 'stats'],
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
        log.error(
            'device format should be "cpu" or device:number e.g. "cuda:0"'
        )
    if device_type not in {'cuda', 'mps'}:
        log.error(f'unrecognised device type {device_type}')

    n_devices = getattr(torch, device_type).device_count()
    if not n_devices:
        log.error(f'no devices available of type {device_type}')
    if (device_number + 1) > n_devices:
        log.error(
            f'device number is too high, only {n_devices} device of type {device_type} avaiable'
        )
    return torch.device(value)


snco_opts.option(
    '-d', '--device',
    subcommands=['predict', 'bc1predict', 'doublet', 'bam2pred', 'csl2pred'],
    required=False,
    type=str,
    default='cpu',
    callback=_check_device,
    help='device to compute predictions on (default cpu)'
)

snco_opts.option(
    '--batch-size',
    subcommands=['predict', 'bc1predict', 'doublet', 'bam2pred', 'csl2pred'],
    required=False,
    type=click.IntRange(1, 10_000),
    default=1_000,
    help='batch size for prediction. larger may be faster but use more memory'
)


def _get_rng(ctx, param, value):
    return np.random.default_rng(value)


snco_opts.option(
    '-r', '--random-seed', 'rng',
    subcommands=['loadbam', 'loadcsl', 'clean', 'sim', 'predict', 'bc1predict',
                 'doublet', 'bam2pred', 'csl2pred', 'plot'],
    required=False,
    type=int,
    default=DEFAULT_RANDOM_SEED,
    callback=_get_rng,
    help='seed for random number generator'
)


snco_opts.option(
    '-v', '--verbosity',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred',
                 'sim', 'concat', 'clean', 'predict', 'bc1predict',
                 'doublet', 'stats', 'plot'],
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


sneqtl_opts = OptionRegistry(subcommands=['eqtl', 'peakcall'])
sneqtl_opts.callback(['eqtl', 'peakcall'])(log_parameters)


sneqtl_opts.argument(
    'exprs-mat-dir',
    subcommands=['eqtl'],
    required=True,
    nargs=1,
    type=_input_dir_type,
)


sneqtl_opts.argument(
    'pred-json-fn',
    subcommands=['eqtl'],
    required=True,
    nargs=1,
    type=_input_file_type,
)


sneqtl_opts.argument(
    'eqtl-tsv-fn',
    subcommands=['peakcall'],
    required=True,
    nargs=1,
    type=_input_file_type,
)


sneqtl_opts.option(
    '-c', '--cb-stats-fn',
    subcommands=['eqtl'],
    required=False,
    type=_input_file_type,
    help='cell barcode stats (output of snco predict/stats)'
)


sneqtl_opts.option(
    '-o', '--output-prefix',
    subcommands=['eqtl'],
    required=True,
    type=_output_file_type,
    help='Output prefix'
)


sneqtl_opts.option(
    '-o', '--output-tsv-fn',
    subcommands=['peakcall'],
    required=True,
    type=_output_file_type,
    help='Output prefix'
)


sneqtl_opts.option(
    '-g', '--gtf-fn',
    subcommands=['eqtl'],
    required=False,
    type=_input_file_type,
    help='GTF file name'
)


sneqtl_opts.option(
    '-g', '--gtf-fn',
    subcommands=['peakcall'],
    required=True,
    type=_input_file_type,
    help='GTF file name'
)


sneqtl_opts.option(
    '--min-cells-exprs',
    subcommands=['eqtl'],
    required=False,
    type=click.FloatRange(0.0, 0.5),
    default=0.02,
    help='the minimum proportion of cell barcodes that must express a gene for it to be tested'
)


sneqtl_opts.option(
    '--control-pcs/--no-pcs', 'control_principal_components',
    subcommands=['eqtl'],
    required=False,
    default=True,
    help='whether to use principal components of the expression matrix as covariates'
)


sneqtl_opts.option(
    '--min-pc-var-explained',
    subcommands=['eqtl'],
    required=False,
    type=click.FloatRange(0, 1.0),
    default=0.01,
    help='the minimum variance explained required to keep a principal component for modelling'
)


sneqtl_opts.option(
    '--max-pc-haplotype-var-explained',
    subcommands=['eqtl'],
    required=False,
    type=click.FloatRange(0.01, 1.0),
    default=0.05,
    help='the maximum variance that a haplotype '
)


def _parse_whitelist_covar_names(ctx, param, value):
    if value is not None:
        value = value.split(',')
    return value


sneqtl_opts.option(
    '--covar-names',
    subcommands=['eqtl'],
    required=False,
    type=str,
    default=None,
    callback=_parse_whitelist_covar_names,
    help=('comma separated list of columns from --cb-stats-fn that should be used as '
          'covariates in the model')
)


sneqtl_opts.option(
    '--genotype/--no-genotype', 'model_parental_genotype',
    subcommands=['eqtl'],
    required=False,
    default=False,
    help='whether to model parental (diploid) genotype'
)


sneqtl_opts.option(
    '--celltype/--no-celltype', 'celltype_haplotype_interaction',
    subcommands=['eqtl'],
    required=False,
    default=False,
    help='whether to cluster barcodes into celltypes and model them as interactors with haplotype'
)


sneqtl_opts.option(
    '--celltype-n-clusters',
    subcommands=['eqtl'],
    required=False,
    type=click.IntRange(2, 10),
    default=None,
    help=('number of clusters to use for celltype clustering. '
          'default is to automatically choose best number using silhouette scoring')
)


sneqtl_opts.option(
    '--control-haplotypes',
    subcommands=['eqtl'],
    required=False,
    default=None,
    help=('comma separated list of positions (in format Chr1:100000), the haplotypes of which '
          'should be controlled for as covariates in the model')
)


sneqtl_opts.option(
    '--control-cis-haplotype/--no-control-cis',
    subcommands=['eqtl'],
    required=False,
    default=False,
    help=('whether to control for the cis-haplotype of genes when modelling them. '
          'Requires GTF file of gene locations to be provided')
)


sneqtl_opts.option(
    '--control-haplotypes-r2',
    subcommands=['eqtl'],
    required=False,
    type=click.FloatRange(0.5, 1),
    default=0.95,
    help=('When controlling for specific haplotypes, do not test linked haplotypes with r2 of more '
          'than this value')
)


sneqtl_opts.option(
    '--cb-filter-exprs',
    subcommands=['eqtl'],
    required=False,
    type=str,
    default=None,
    help=('expression used to filter barcodes by column in --cb-stats-fn '
          'e.g. "doublet_probability < 0.5"')
)

sneqtl_opts.option(
    '--peak-variable',
    subcommands=['peakcall'],
    required=False,
    type=str,
    default='overall',
    help='the variable used for lod peak calling'
)

sneqtl_opts.option(
    '--lod-threshold',
    subcommands=['eqtl', 'peakcall'],
    required=False,
    type=click.FloatRange(2, 20),
    default=5,
    help='Absolute LOD score threshold used for peak calling'
)


sneqtl_opts.option(
    '--rel-lod-threshold',
    subcommands=['eqtl', 'peakcall'],
    required=False,
    type=click.FloatRange(0, 1),
    default=0.1,
    help=('Relative LOD score threshold used for secondary peak calling '
          'compared to highest LOD on chromosome')
)


sneqtl_opts.option(
    '--pval-threshold',
    subcommands=['eqtl', 'peakcall'],
    required=False,
    type=click.FloatRange(0, 1),
    default=1e-3,
    help='P value threshold used for peak calling'
)


sneqtl_opts.option(
    '--rel-prominence',
    subcommands=['eqtl', 'peakcall'],
    required=False,
    type=click.FloatRange(0, 1),
    default=0.25,
    help=('Relative prominence of LOD score used for secondary peak calling '
          'compared to highest LOD on chromosome')
)


sneqtl_opts.option(
    '--ci-lod-drop',
    subcommands=['eqtl', 'peakcall'],
    required=False,
    type=click.FloatRange(1, 10),
    default=1.5,
    help='Drop in LOD score from peak used to calculate eQTL confidence intervals'
)


sneqtl_opts.option(
    '--min-dist-between-eqtls',
    subcommands=['eqtl', 'peakcall'],
    required=False,
    type=click.IntRange(0, 1e10),
    default=3e6,
    help='Minimum allowed distance in bases between two eQTL peaks'
)


sneqtl_opts.option(
    '--cis-eqtl-range',
    subcommands=['eqtl', 'peakcall'],
    required=False,
    type=click.IntRange(0, 5e6),
    default=5e6,
    help='Maximum distance from edge of eQTL CI boundary to gene for it to be considered a cis-eQTL'
)


sneqtl_opts.option(
    '-p', '--processes',
    subcommands=['eqtl',],
    required=False,
    type=click.IntRange(min=1),
    default=1,
    help='number of cpu processes to use'
)


sneqtl_opts.option(
    '-r', '--random-seed', 'rng',
    subcommands=['eqtl'],
    required=False,
    type=int,
    default=DEFAULT_RANDOM_SEED,
    callback=_get_rng,
    help='seed for random number generator'
)


sneqtl_opts.option(
    '-v', '--verbosity',
    subcommands=['eqtl', 'peakcall'],
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
