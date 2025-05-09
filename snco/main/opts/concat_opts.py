import re
from .snco_opts import snco_opts


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
