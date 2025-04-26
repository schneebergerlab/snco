import logging
from collections import Counter

import numpy as np

from .barcodes import CellBarcodeWhitelist
from .records import MarkerRecords, PredictionRecords


log = logging.getLogger('snco')


def read_cb_whitelist(barcode_fn, validate_barcodes=True,
                      cb_correction_method='exact',
                      allow_ns=False, allow_homopolymers=False):
    """
    Read a text file of cell barcodes and return them as a list.

    In a multi-column file, the barcode must be in the first column.
    
    Parameters
    ----------
    barcode_fn : str
        Path to the text file containing cell barcodes.
    validate_barcodes : bool, optional
        Whether to validate the barcodes. The default is True.
    cb_correction_method : str, optional
        Method to correct barcodes, can be 'exact' or another method. The default is 'exact'.
    allow_ns : bool, optional
        Whether to allow 'N' bases in barcodes. The default is False.
    allow_homopolymers : bool, optional
        Whether to allow homopolymeric sequences in barcodes. The default is False.

    Returns
    -------
    CellBarcodeWhitelist
        A `CellBarcodeWhitelist` object containing the read barcodes and their filtering options.
    """
    if barcode_fn is not None:
        with open(barcode_fn) as f:
            cb_whitelist = [cb.strip().split('\t')[0] for cb in f.readlines()]
        log.info(f'Read {len(cb_whitelist)} cell barcodes from cb whitelist file {barcode_fn}')
    else:
        cb_whitelist = None
    return CellBarcodeWhitelist(cb_whitelist,
                                validate_barcodes,
                                cb_correction_method,
                                allow_ns=allow_ns,
                                allow_homopolymers=allow_homopolymers)


def load_json(json_fn, cb_whitelist_fn, bin_size, data_type='markers', subset=None):
    """
    Load MarkerRecords or PredictionRecords from a JSON file and filter based on cell barcode whitelist.

    Parameters
    ----------
    json_fn : str
        Path to the JSON file containing the data.
    cb_whitelist_fn : str, optional
        Path to the file containing the cell barcode whitelist.
    bin_size : int, optional
        The bin size used for processing the data.
    data_type : {'markers', 'predictions', 'auto'}, optional
        Specifies the type of data to load. The default is 'markers'.
    subset : iterable, optional
        A subset of the data to load.

    Returns
    -------
    MarkerRecords or PredictionRecords
        The data loaded from the JSON file, filtered based on the barcode whitelist.
    
    Raises
    ------
    ValueError
        If the bin size does not match the bin size in the JSON file or if no barcodes match
        the whitelist.
    NotImplementedError
        If the data type is not recognized.
    IOError
        If the data type cannot be automatically determined from the file.
    """
    if data_type == 'markers':
        data = MarkerRecords.read_json(json_fn, subset=subset)
    elif data_type == 'predictions':
        data = PredictionRecords.read_json(json_fn, subset=subset)
    elif data_type == 'auto':
        try:
            data = MarkerRecords.read_json(json_fn, subset=subset)
        except ValueError as exc:
            try:
                data = PredictionRecords.read_json(json_fn, subset=subset)
            except ValueError:
                raise IOError(
                    f'data type of file {json_fn} could not be determined automatically'
                ) from exc
    else:
        raise NotImplementedError()
    if bin_size is not None and data.bin_size != bin_size:
        raise ValueError('"--bin-size" does not match bin size specified in json-fn, '
                         'please modify cli option or rerun previous snco steps')
    log.info(f'Read {len(data)} cell barcodes from json file {json_fn}')
    if cb_whitelist_fn:
        cb_whitelist = read_cb_whitelist(cb_whitelist_fn,  validate_barcodes=False).toset()
        data.filter(cb_whitelist)
        if not data:
            raise ValueError('No CBs from --cb-whitelist-fn are present in json-fn')
        log.info(f'{len(data)} barcodes remain after cb whitelist filtering')
    return data


def spawn_child_rngs(rng):
    """
    Spawn a series of random number generators (RNGs) based on a given parent RNG.

    This function generates child random number generators using the state of a parent RNG.
    Each child RNG is independent but has a state determined by the parent. Useful for
    retaining deterministic behaviour when using multiprocessing.

    Parameters
    ----------
    rng : np.random.Generator
        The parent random number generator to base child RNGs on.

    Yields
    ------
    np.random.Generator
        A child random number generator, created from the parent RNG.
    """
    while True:
        yield np.random.default_rng(rng.spawn(1)[0])


def genotyping_results_formatter(genotypes):
    """
    Format the genotyping results into a human-readable string.
    """
    geno_counts = Counter(
        [':'.join(sorted(cb_g)) for cb_g in genotypes.values()]
    )
    fmt = 'Genotyping results:\n'
    ljust_size = max([len(g) for g in geno_counts.keys()]) + 5
    ljust_size = max(ljust_size, 12)
    fmt += f'   Genotype'.ljust(ljust_size)
    fmt += 'Num. barcodes\n'
    for geno, count in geno_counts.most_common():
        fmt += f'   {geno}'.ljust(ljust_size)
        fmt += f'{count}\n'
    return fmt