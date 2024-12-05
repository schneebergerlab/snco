import logging
from .barcodes import CellBarcodeWhitelist
from .records import MarkerRecords, PredictionRecords


log = logging.getLogger('snco')


def read_cb_whitelist(barcode_fn, validate_barcodes=True,
                      cb_correction_method='exact',
                      allow_ns=False, allow_homopolymers=False):
    '''
    Read a text file of cell barcodes and return them as a list.
    In a multi-column file, barcode must be the first column
    '''
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
        cb_whitelist = read_cb_whitelist(cb_whitelist_fn).toset()
        data.filter(cb_whitelist)
        if not data:
            raise ValueError('No CBs from --cb-whitelist-fn are present in json-fn')
        log.info(f'{len(data)} barcodes remain after cb whitelist filtering')
    return data
