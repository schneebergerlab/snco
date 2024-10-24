import logging

import click
import click_log

from .bam import read_cb_whitelist
from .records import MarkerRecords, PredictionRecords
from .markers import get_co_markers
from .signal import (
    clean_marker_background, apply_haplotype_imbalance_mask, apply_marker_threshold
)
from .sim import (
    read_ground_truth_haplotypes, generate_simulated_data, ground_truth_from_marker_records
)
from .predict import create_rhmm, detect_crossovers
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
    click.option('-c', '--cb-whitelist-fn', required=False,
                 help='Text file containing whitelisted cell barcodes, one per line'),
    click.option('-n', '--bin-size', required=False, default=25_000,
                 help='Bin size for marker distribution'),
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
    Read bam files with cell barcode, umi and haplotype tags (aligned with STAR solo+diploid), 
    to generate a json file of binned haplotype marker distributions for each cell barcode. 
    These can be used to call recombinations using the downstream `predict` command.
    '''

    if cb_whitelist_fn:
        cell_barcode_whitelist = read_cb_whitelist(cb_whitelist_fn)
    else:
        cell_barcode_whitelist = None
    co_markers = get_co_markers(
        bam_fns, processes=processes,
        bin_size=bin_size,
        cb_tag=cb_tag,
        umi_tag=umi_tag,
        hap_tag=hap_tag,
        cell_barcode_whitelist=cell_barcode_whitelist,
    )
    co_markers.write_json(output_json_fn)


def _common_json_load(json_fn, cb_whitelist_fn, bin_size, data_type=MarkerRecords):
    co_markers = data_type.read_json(json_fn)
    if co_markers.bin_size != bin_size:
        raise ValueError('"--bin-size" does not match bin size specified in marker json-fn, '
                         'please modify cli option or rerun snco load')
    if cb_whitelist_fn:
        cell_barcode_whitelist = read_cb_whitelist(cb_whitelist_fn)
        co_markers.set_cb_whitelist(cell_barcode_whitelist)
        if not co_markers:
            raise ValueError('No CBs from --cb-whitelist-fn are present in json-fn')
    return co_markers


JSON_COMMON_OPTIONS = [
    click.argument('json-fn', required=True, nargs=1),
    click.option('-o', '--output-json-fn', required=True, help='Output JSON file name.')
]

@main.command()
@_common_options(JSON_COMMON_OPTIONS)
@click.option('-b', '--haplo-bed-fn', required=True,
              help=('bed file (6 column) containing haplotype intervals to simulate. '
                    'Sample ids should be listed in column 4, haplotype (0 or 1) in column 5. '
                    'Haplotype intervals for each sample id should cover the '
                    'entire genome without gaps.'))
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
    co_markers = _common_json_load(json_fn, cb_whitelist_fn, bin_size)
    ground_truth_haplotypes = read_ground_truth_haplotypes(
        haplo_bed_fn, co_markers.chrom_sizes, bin_size
    )
    sim_co_markers = generate_simulated_data(
        ground_truth_haplotypes,
        co_markers,
        bg_rate=background_marker_rate,
        conv_window_size=bg_window_size,
        nsim_per_sample=nsim_per_sample,
    )
    sim_co_markers.write_json(output_json_fn)


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
    Removes predicted background markers, that result from ambient nucleic acids, 
    from each cell barcode.
    '''
    co_markers = _common_json_load(json_fn, cb_whitelist_fn, bin_size)

    # first estimate ambient marker rate for each CB and try to scrub common background markers
    co_markers_c = clean_marker_background(co_markers, bg_window_size)
    # next mask any bins that still have extreme imbalance
    # (e.g. due to extreme allele-specific expression differences)
    co_markers_m = apply_haplotype_imbalance_mask(co_markers_c, max_marker_imbalance)
    # finally threshold bins that have a large number of reads
    co_markers_t = apply_marker_threshold(co_markers_m, max_bin_count)

    co_markers_t.write_json(output_json_fn)


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
    Uses rigid hidden Markov model to predict the haplotypes of each cell barcode
    at each genomic bin.
    '''
    co_markers = _common_json_load(json_fn, cb_whitelist_fn, bin_size)
    rhmm = create_rhmm(
        co_markers,
        cm_per_mb=cm_per_mb,
        segment_size=segment_size,
        terminal_segment_size=terminal_segment_size,
        model_lambdas='auto' # todo, implement cli option with proper type checking
    )
    co_preds = detect_crossovers(co_markers, rhmm, processes=processes)
    co_preds.write_json(output_json_fn, output_precision)


@main.command()
@click.argument('marker-json-fn', required=True, nargs=1)
@click.argument('predict-json-fn', required=True, nargs=1)
@click.option('-o', '--output-tsv-fn', required=True, help='Output TSV file name.')
@_common_options(COMMON_OPTIONS)
@click_log.simple_verbosity_option(log)
def stats(marker_json_fn, predict_json_fn, output_tsv_fn, cb_whitelist_fn, bin_size):
    '''
    Scores the quality of data and predictions for a set of haplotype calls
    generated with `predict`.
    '''
    co_markers = _common_json_load(marker_json_fn, cb_whitelist_fn, bin_size)
    co_preds = _common_json_load(
        predict_json_fn, cb_whitelist_fn, bin_size, data_type=PredictionRecords
    )

    if set(co_preds.seen_barcodes) != set(co_markers.seen_barcodes):
        raise ValueError('Cell barcodes from marker-json-fn and predict-json-fn do not match')

    qual_metrics = calculate_quality_metrics(co_markers, co_preds)
    if 'ground_truth' in co_markers.metadata:
        ground_truth_haplotypes = ground_truth_from_marker_records(co_markers)
        score_metrics = calculate_score_metrics(co_markers, co_preds, ground_truth_haplotypes)
    else:
        score_metrics = None

    write_metric_tsv(output_tsv_fn, qual_metrics, score_metrics)


@main.command()
@click.argument('marker-json-fn', required=True, nargs=1)
@click.argument('predict-json-fn', required=True, nargs=1)
def plot(marker_json_fn, predict_json_fn):
    raise NotImplementedError()
