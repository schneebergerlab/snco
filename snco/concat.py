from .utils import load_json

def concatenate_records(record_objs, merge_suffixes=None):
    if merge_suffixes is None:
        merge_suffixes = range(len(record_objs))
    else:
        if len(record_objs) != len(merge_suffixes):
            raise ValueError('number of merge suffixes is incorrect')
    concat_records = None
    for robj, sfx in zip(record_objs, merge_suffixes):
        robj = robj.add_cb_suffix(sfx)
        if concat_records is None:
            concat_records = robj.copy()
        else:
            concat_records.merge(robj, inplace=True)
    return concat_records


def run_concat(json_fn, output_json_fn, merge_suffixes=None):
    '''
    Concatenates marker jsons, potentially from different datasets, 
    adding suffixes to cell barcodes to avoid name collisions
    '''
    record_objs = []
    for fn in json_fn:
        robj = load_json(fn, cb_whitelist_fn=None, bin_size=None, data_type='auto')
        record_objs.append(robj)
    concat_records = concatenate_records(record_objs, merge_suffixes)
    concat_records.write_json(output_json_fn)
