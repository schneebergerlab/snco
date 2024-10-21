import os
import logging

import click
import click_log


from .bam import read_cb_whitelist
from .markers import get_co_markers, co_markers_to_json, load_co_markers_from_json
from .signal import clean_marker_background, apply_haplotype_imbalance_mask, apply_marker_threshold
from .sim import read_ground_truth_haplotypes, generate_simulated_data
from .predict import create_rhmm, detect_crossovers, co_preds_to_json

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
    click.option('-c', '--cb-whitelist-fn', required=False),
    click.option('-n', '--bin-size', required=False, default=25_000),
]


@main.command()
@click.argument('bam-fns', required=True, nargs=-1)
@click.option('-o', '--output-json-fn', required=True)
@_common_options(COMMON_OPTIONS)
@click.option('--cb-tag', required=False, default='CB')
@click.option('--umi-tag', required=False, default='UB')
@click.option('--hap-tag', required=False, default='ha')
@click.option('--processes', required=False, default=1)
@click_log.simple_verbosity_option(log)
def load(bam_fns, output_json_fn, cb_whitelist_fn, bin_size,
         cb_tag, umi_tag, hap_tag, processes):

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


def _common_json_load(json_fn, cb_whitelist_fn, bin_size):
    co_markers, chrom_sizes, co_marker_bin_size = load_co_markers_from_json(json_fn)
    if co_marker_bin_size != bin_size:
        raise ValueError('"--bin-size" does not match bin size specified in marker json-fn, please rerun snco load')
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
    click.option('-o', '--output-json-fn', required=True)
]

@main.command()
@_common_options(JSON_COMMON_OPTIONS)
@click.option('-b', '--haplo-bed-fn', required=True)
@_common_options(COMMON_OPTIONS)
@click.option('--background-marker-rate', required=False, default='auto')
@click.option('--bg-window-size', required=False, default=2_500_000)
@click.option('--nsim-per-sample', required=False, default=100)
@click_log.simple_verbosity_option(log)
def sim(json_fn, output_json_fn, haplo_bed_fn, cb_whitelist_fn, bin_size,
        background_marker_rate, bg_window_size, nsim_per_sample):
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
def merge(json_fn, output_json_fn, bin_size, cb_whitelist_fn):
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
@_common_options(JSON_COMMON_OPTIONS)
@_common_options(COMMON_OPTIONS)
@click_log.simple_verbosity_option(log)
def stats(json_fn, output_json_fn, cb_whitelist_fn, bin_size):
    raise NotImplementedError()