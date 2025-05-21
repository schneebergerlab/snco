import numpy as np
import click
from .snco_opts import snco_opts
from ..logger import click_logger
from snco.defaults import DEFAULT_RANDOM_SEED

snco_opts.option(
    '-p', '--processes',
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred', 'predict',
                 'doublet', 'segdist'],
    required=False,
    type=click.IntRange(min=1),
    default=1,
    help='number of cpu processes to use'
)


snco_opts.option(
    '--output-precision',
    subcommands=['predict', 'doublet', 'bam2pred', 'csl2pred', 'stats', 'segdist'],
    required=False,
    type=click.IntRange(1, 10),
    default=5,
    help='floating point precision in output files'
)


def _get_rng(ctx, param, value):
    return np.random.default_rng(value)


snco_opts.option(
    '-r', '--random-seed', 'rng',
    subcommands=['loadbam', 'loadcsl', 'clean', 'sim', 'predict',
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
                 'sim', 'concat', 'clean', 'predict',
                 'doublet', 'stats', 'plot', 'segdist'],
    required=False,
    expose_value=False,
    metavar='LVL',
    type=click.Choice(
        ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        case_sensitive=False
    ),
    default='info',
    callback=click_logger('snco'),
    help='Logging level, either debug, info, warning, error or critical'
)
