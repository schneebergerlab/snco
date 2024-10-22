import os
import logging

import click
import click_log


from .bam import read_cb_whitelist
from .markers import get_co_markers, co_markers_to_json, load_co_markers_from_json
from .signal import clean_marker_background, apply_haplotype_imbalance_mask, apply_marker_threshold
from .sim import read_ground_truth_haplotypes, generate_simulated_data
from .predict import create_rhmm, detect_crossovers, co_preds_to_json, load_co_preds_from_json
from .stats import calculate_quality_metrics, calculate_score_metrics, write_metric_tsv

log = logging.getLogger('snco')
click_log.basic_config(log)


@click.group()
def main():
    pass


def _common_options(common_options):
    def _apply_common_options(func):
        for option in reversed(common_options):
            func = option(func)
        return func
    return _apply_common_options


COMMON_OPTIONS = [
    click.option('-c', '--cb-whitelist-fn', required=False, help='Text file containing whitelisted cell barcodes, one per line'),
    click.option('-n', '--bin-size', required=False, default=25_000, help='Bin size for marker distribution'),
]


@main.command()
@click.argument('bam-fns', required=True, nargs=-1)
@click.option('-o', '--output-json-fn', required=True, help='Output JSON file name.')
@_common_options(COMMON_OPTIONS)
@click.option('--cb-tag', required=False, default='CB',
              help=('tag representing cell barcode. '
                    'Should be string type. '
                    'These are currently not corrected but are matched to the whitelist.'))
@click.option('--umi-tag', required=False, default='UB',
              help=('tag representing UMI. '
                    'Should be string type '
                    'These will be corrected using the directional method and deduplicated.'))
@click.option('--hap-tag', required=False, default='ha',
              help=('tag representing haplotype. '
                    'Should be integer type (1: hap1, 2: hap2, 0: both hap1 and hap2) '
                    'as described in STAR diploid documentation.'))
@click.option('--processes', required=False, default=1)
@click_log.simple_verbosity_option(log)
def load(bam_fns, output_json_fn, cb_whitelist_fn, bin_size,
         cb_tag, umi_tag, hap_tag, processes):
    '''
    Read bam files with cell barcode, umi and haplotype tags (aligned with STAR solo+diploid or similar), 
    to generate a json file of binned haplotype marker distributions for each cell barcode. These can be used
    to call recombinations using the downstream `predict` command.
    '''

    if cb_whitelist_fn:
        cell_barcode_whitelist = read_cb_whitelist(cb_whitelist_fn)
    else:
        cell_barcode_whitelist = None
    co_markers, chrom_sizes = get_co_markers(
        bam_fns, processes=processes,
        bin_size=bin_size,
        cb_tag=cb_tag,
        umi_tag=umi_tag,
        hap_tag=hap_tag,
        cell_barcode_whitelist=cell_barcode_whitelist,
    )
    co_markers_to_json(
        output_json_fn,
        co_markers,
        chrom_sizes,
        bin_size,
    )


def _common_json_load(json_fn, cb_whitelist_fn, bin_size, json_load_cmd=load_co_markers_from_json):
    co_markers, chrom_sizes, co_marker_bin_size = json_load_cmd(json_fn)
    if co_marker_bin_size != bin_size:
        raise ValueError('"--bin-size" does not match bin size specified in marker json-fn, '
                         'please modify cli option or rerun snco load')
    if cb_whitelist_fn:
        cell_barcode_whitelist = read_cb_whitelist(cb_whitelist_fn)
        co_markers = {
            cb: m for cb, m in co_markers.items() if cb in cell_barcode_whitelist
        }
        if not len(co_markers):
            raise ValueError('No CBs from --cb-whitelist-fn are present in json-fn')
    return co_markers, chrom_sizes


JSON_COMMON_OPTIONS = [
    click.argument('json-fn', required=True, nargs=1),
    click.option('-o', '--output-json-fn', required=True, help='Output JSON file name.')
]

@main.command()
@_common_options(JSON_COMMON_OPTIONS)
@click.option('-b', '--haplo-bed-fn', required=True,
              help=('bed file (6 column) containing haplotype intervals to simulate. '
                    'Sample ids should be listed in column 4, haplotype (0 or 1) in column 5. '
                    'Haplotype intervals for each sample id should cover the entire genome without gaps.'))
@_common_options(COMMON_OPTIONS)
@click.option('--background-marker-rate', required=False, default='auto')
@click.option('--bg-window-size', required=False, default=2_500_000)
@click.option('--nsim-per-sample', required=False, default=100)
@click_log.simple_verbosity_option(log)
def sim(json_fn, output_json_fn, haplo_bed_fn, cb_whitelist_fn, bin_size,
        background_marker_rate, bg_window_size, nsim_per_sample):
    '''
    Simulate realistic haplotype marker distributions using real data from `load`,
    with known haplotypes/crossovers supplied from a bed file.
    '''
    co_markers, chrom_sizes = _common_json_load(json_fn, cb_whitelist_fn, bin_size)
    ground_truth_haplotypes = read_ground_truth_haplotypes(haplo_bed_fn, chrom_sizes, bin_size)
    sim_data = generate_simulated_data(
        ground_truth_haplotypes,
        co_markers,
        bg_rate=background_marker_rate,
        conv_window_size=bg_window_size,
        nsim_per_sample=nsim_per_sample,
    )
    co_markers_to_json(
        output_json_fn,
        sim_data,
        chrom_sizes,
        bin_size,
    )


@main.command()
@_common_options(JSON_COMMON_OPTIONS)
@_common_options(COMMON_OPTIONS)
@click_log.simple_verbosity_option(log)
def concat(json_fn, output_json_fn, bin_size, cb_whitelist_fn):
    '''
    Concatenate marker jsons
    '''
    raise NotImplementedError()
    

@main.command()
@_common_options(JSON_COMMON_OPTIONS)
@_common_options(COMMON_OPTIONS)
@click.option('--max-bin-count', required=False, default=20)
@click.option('--bg-window-size', required=False, default=2_500_000)
@click.option('--max-marker-imbalance', required=False, default=0.9)
@click_log.simple_verbosity_option(log)
def clean(json_fn, output_json_fn, cb_whitelist_fn, bin_size,
          max_bin_count, bg_window_size, max_marker_imbalance):
    '''
    Removes predicted background markers, that result from ambient nucleic acids, from each cell barcode.
    '''
    co_markers, chrom_sizes = _common_json_load(json_fn, cb_whitelist_fn, bin_size)
    
    # first estimate ambient marker rate for each CB and try to scrub common background markers
    co_markers_c = clean_marker_background(co_markers, bin_size, bg_window_size)
    # next mask any bins that still have extreme imbalance (e.g. due to extreme allele-specific expression differences)
    co_markers_m = apply_haplotype_imbalance_mask(co_markers_c, max_marker_imbalance)
    # finally threshold bins that have a large number of reads
    co_markers_t = apply_marker_threshold(co_markers_m, max_bin_count)

    co_markers_to_json(
        output_json_fn,
        co_markers_t,
        chrom_sizes,
        bin_size,
    )


@main.command()
@_common_options(JSON_COMMON_OPTIONS)
@_common_options(COMMON_OPTIONS)
@click.option('-r', '--segment-size', required=False, default=1_000_000)
@click.option('-t', '--terminal-segment-size', required=False, default=50_000)
@click.option('-c', '--cm-per-mb', required=False, default=4.5)
@click.option('--output-precision', required=False, default=2)
@click.option('--processes', required=False, default=1)
@click_log.simple_verbosity_option(log)
def predict(json_fn, output_json_fn, cb_whitelist_fn, bin_size,
            segment_size, terminal_segment_size, cm_per_mb,
            output_precision, processes):
    '''
    Uses rigid hidden Markov model to predict the haplotypes of each cell barcode at each genomic bin.
    '''
    co_markers, chrom_sizes = _common_json_load(json_fn, cb_whitelist_fn, bin_size)
    rhmm = create_rhmm(
        co_markers,
        bin_size=bin_size,
        cm_per_mb=cm_per_mb,
        segment_size=segment_size,
        terminal_segment_size=terminal_segment_size,
        model_lambdas='auto' # todo, implement cli option with proper type checking
    )
    co_preds = detect_crossovers(co_markers, rhmm, chrom_sizes, processes=processes)
    co_preds_to_json(output_json_fn, co_preds, chrom_sizes, bin_size, precision=output_precision)


@main.command()
@click.argument('marker-json-fn', required=True, nargs=1)
@click.argument('predict-json-fn', required=True, nargs=1)
@click.option('-b', '--haplo-bed-fn', required=False,
              help=('bed file (6 column) containing ground truth haplotypes to score predictions against'))
@click.option('-o', '--output-tsv-fn', required=True, help='Output TSV file name.')
@_common_options(COMMON_OPTIONS)
@click_log.simple_verbosity_option(log)
def stats(marker_json_fn, predict_json_fn, haplo_bed_fn, output_tsv_fn, cb_whitelist_fn, bin_size):
    '''
    Scores the quality of data and predictions for a set of haplotype calls generated with `predict`.
    '''
    co_markers, chrom_sizes = _common_json_load(marker_json_fn, cb_whitelist_fn, bin_size)
    co_preds, _ = _common_json_load(predict_json_fn, cb_whitelist_fn, bin_size, json_load_cmd=load_co_preds_from_json)
    if set(co_preds) != set(co_markers):
        raise ValueError('Cell barcodes from marker-json-fn and predict-json-fn do not match')

    qual_metrics = calculate_quality_metrics(co_markers, co_preds)
    if haplo_bed_fn is not None:
        ground_truth_haplotypes = read_ground_truth_haplotypes(haplo_bed_fn, chrom_sizes, bin_size)
        score_metrics = calculate_score_metrics(co_preds, ground_truth_haplotypes)
    else:
        score_metrics = None

    write_metric_tsv(output_tsv_fn, qual_metrics, score_metrics)