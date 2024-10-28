import click
from . import opts
from . import logger


def get_kwarg_subset(kw_list, kwargs):
    return {kw: kwargs[kw] for kw in kw_list}


def _clean_predict_pipeline(output_prefix, load_output, kwargs):

    from .clean import run_clean
    from .predict import run_predict

    if kwargs['run_clean']:

        clean_kwargs = [
            'bin_size', 'min_markers_per_cb', 'max_bin_count',
            'clean_bg', 'bg_window_size',
            'mask_imbalanced', 'max_marker_imbalance',
        ]
        clean_kwargs = get_kwarg_subset(clean_kwargs, kwargs)
        clean_kwargs['marker_json_fn'] = load_output
        clean_output = f'{output_prefix}.cmarkers.json'
        clean_kwargs['output_json_fn'] = clean_output

        run_clean(**clean_kwargs)

    else:
        clean_output = load_output

    predict_kwargs = [
        'bin_size', 'segment_size', 'terminal_segment_size',
        'cm_per_mb', 'model_lambdas', 'output_precision',
        'processes', 'batch_size', 'device',
    ]
    predict_kwargs = get_kwarg_subset(predict_kwargs, kwargs)
    predict_kwargs['marker_json_fn'] = clean_output
    predict_output = f'{output_prefix}.pred.json'
    predict_kwargs['output_json_fn'] = predict_output

    run_predict(**predict_kwargs)


pipeline_loadbam_options = [
    opts.output_prefix,
    opts.bam, opts.cb_whitelist, opts.bin_size,
    opts.seq_type, opts.cb_corr_method, opts.cb_tag,
    opts.umi_collapse_method, opts.umi_tag,
    opts.hap_tag, opts.excl_contigs,
]


pipeline_clean_pred_options = [
    opts.min_markers, opts.max_bin_count,
    opts.run_clean, opts.clean_bg, opts.bg_window_size,
    opts.mask_imbalanced, opts.max_imbalance,
    opts.seg_size, opts.term_seg_size, opts.cm_per_mb, opts.model_lambdas,
    opts.precision, opts.processes, opts.batch_size, opts.device,
    opts.processes, logger.verbosity
]


@click.command('bam2pred')
@opts.apply_options(pipeline_loadbam_options)
@opts.apply_options(pipeline_clean_pred_options)
def bam_pipeline_subcommand(**kwargs):
    '''
    Pipeline chaining together the loadbam, clean and predict commands
    '''
    from .loadbam import run_loadbam

    output_prefix = kwargs.pop('output_prefix')
    loadbam_kwargs = [
        'bam_fn', 'cb_whitelist_fn', 'bin_size', 'seq_type',
        'cb_correction_method', 'cb_tag',
        'umi_collapse_method', 'umi_tag', 'hap_tag',
        'exclude_contigs', 'processes'
    ]
    loadbam_kwargs = get_kwarg_subset(loadbam_kwargs, kwargs)
    loadbam_output = f'{output_prefix}.markers.json'
    loadbam_kwargs['output_json_fn'] = loadbam_output

    run_loadbam(**loadbam_kwargs)
    _clean_predict_pipeline(output_prefix, loadbam_output, kwargs)


pipeline_loadcsl_options = [
    opts.output_prefix,
    opts.csl_dir, opts.chrom_sizes,
    opts.cb_whitelist, opts.bin_size,
    logger.verbosity
]


@click.command('csl2pred')
@opts.apply_options(pipeline_loadcsl_options)
@opts.apply_options(pipeline_clean_pred_options)
def csl_pipeline_subcommand(**kwargs):
    '''
    Pipeline chaining together the loadcsl, clean and predict commands
    '''
    from .loadcsl import run_loadcsl

    output_prefix = kwargs.pop('output_prefix')
    loadcsl_kwargs = [
        'cellsnp_lite_dir', 'chrom_sizes_fn',
        'cb_whitelist_fn', 'bin_size',
    ]
    loadcsl_kwargs = get_kwarg_subset(loadcsl_kwargs, kwargs)
    loadcsl_output = f'{output_prefix}.markers.json'
    loadcsl_kwargs['output_json_fn'] = loadcsl_output

    run_loadcsl(**loadcsl_kwargs)
    _clean_predict_pipeline(output_prefix, loadcsl_output, kwargs)
