import os
import logging

import click
import click_log


from .bam import read_cb_whitelist
from .markers import (
    get_co_markers, estimate_and_subtract_background,
    co_markers_to_json, load_co_markers_from_json
)
from .sim import read_ground_truth_haplotypes, generate_simulated_data


log = logging.getLogger('snco')
click_log.basic_config(log)


@click.group()
def main():
    pass


COMMON_OPTIONS = [
    click.option('-c', '--cb-whitelist-fn', required=False),
    click.option('-n', '--bin-size', required=False, default=25_000),
]

def _common_options(common_options):
    def _apply_common_options(func):
        for option in reversed(common_options):
            func = option(func)
        return func
    return _apply_common_options


@main.command()
@click.argument('bam-fns', required=True, nargs=-1)
@click.option('-o', '--output-json-fn', required=True)
@_common_options(COMMON_OPTIONS)
@click.option('--max-bin-count', required=False, default=20)
@click.option('--bg-window-size', required=False, default=1_000_000)
@click.option('--max-marker-imbalance', required=False, default=0.9)
@click.option('--cb-tag', required=False, default='CB')
@click.option('--umi-tag', required=False, default='UB')
@click.option('--hap-tag', required=False, default='ha')
@click.option('--processes', required=False, default=1)
@click_log.simple_verbosity_option(log)
def load(bam_fns, output_json_fn, cb_whitelist_fn, bin_size,
         max_bin_count, bg_window_size,
         max_marker_imbalance,
         cb_tag, umi_tag, hap_tag,
         processes):

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
    perc_contamination, co_markers = estimate_and_subtract_background(
        co_markers,
        chrom_sizes,
        bin_size=bin_size,
        window_size=bg_window_size,
        max_bin_count=max_bin_count,
        max_imbalance_mask=max_marker_imbalance
    )
    co_markers_to_json(
        output_json_fn,
        co_markers,
        chrom_sizes,
        bin_size,
    )


@main.command()
@click.argument('json-fn', required=True, nargs=1)
@click.option('-o', '--output-json-fn', required=True)
@click.option('-b', '--haplo-bed-fn', required=True)
@_common_options(COMMON_OPTIONS)
@click.option('--background-marker-rate', required=False, default=0.05)
@click.option('--nsim-per-sample', required=False, default=100)
@click_log.simple_verbosity_option(log)
def sim(json_fn, output_json_fn, haplo_bed_fn, cb_whitelist_fn, bin_size,
        background_marker_rate, nsim_per_sample):
    co_markers, chrom_sizes, co_marker_bin_size = load_co_markers_from_json(json_fn)
    if co_marker_bin_size != bin_size:
        raise ValueError('"--bin-size" does not match bin size specified in marker json-fn, please rerun snco load')
    if cb_whitelist_fn:
        cell_barcode_whitelist = read_cb_whitelist(cb_whitelist_fn)
        co_markers = {
            cb: m for cb, m in co_markers.items() if cb.rsplit('_')[0] in cell_barcode_whitelist
        }
    ground_truth_haplotypes = read_ground_truth_haplotypes(haplo_bed_fn, chrom_sizes, bin_size)
    sim_data = generate_simulated_data(
        ground_truth_haplotypes,
        co_markers,
        bg_rate=background_marker_rate,
        nsim_per_sample=nsim_per_sample,
    )
    co_markers_to_json(
        output_json_fn,
        sim_data,
        chrom_sizes,
        bin_size,
    )    


@main.command()
@_common_options(COMMON_OPTIONS)
@click_log.simple_verbosity_option(log)
def train():
    pass


@main.command()
@_common_options(COMMON_OPTIONS)
@click_log.simple_verbosity_option(log)
def predict():
    pass