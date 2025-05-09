import click
from .snco_opts import snco_opts


snco_opts.option(
    '--segdist-order',
    subcommands=['segdist'],
    required=False,
    type=click.IntRange(1, 4),
    default=1,
    help='Number of loci to jointly test for distortion.'
)


snco_opts.option(
    '--downsample-resolution',
    subcommands=['segdist'],
    required=False,
    type=click.IntRange(25_000, 10_000_000),
    default=250_000,
    help='Resolution to downsample the prediction matrix to before testing.'
)
