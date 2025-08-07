from ..registry import OptionRegistry
from ..logger import click_logger
from .callbacks import (
    log_parameters,
    validate_loadbam_input,
    validate_loadcsl_input,
    validate_clean_input,
    validate_pred_input,
)


snco_opts = OptionRegistry(
    subcommands=['loadbam', 'loadcsl', 'bam2pred', 'csl2pred',
                 'sim', 'concat', 'clean', 'predict',
                 'doublet', 'stats', 'plot', 'segdist']
)
snco_opts.register_callback(log_parameters('snco'))
snco_opts.register_callback(
    validate_loadbam_input,
    subcommands=['loadbam', 'bam2pred'],
)
snco_opts.register_callback(
    validate_loadcsl_input,
    subcommands=['loadcsl', 'csl2pred'],
)
snco_opts.register_callback(
    validate_clean_input,
    subcommands=['clean', 'bam2pred', 'csl2pred']
)
snco_opts.register_callback(
    validate_pred_input,
    subcommands=['predict', 'bam2pred', 'csl2pred']
)
