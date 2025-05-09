import click
from .snco_opts import snco_opts


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
