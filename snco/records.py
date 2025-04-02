import logging
import sys
from copy import copy, deepcopy
from collections import defaultdict
import json

import numpy as np
from scipy import sparse
import pandas as pd

from .bam import IntervalMarkerCounts

log = logging.getLogger('snco')


class BaseRecords:

    '''
    base class for MarkerRecords and PredictionRecords classes
    '''

    def __init__(self,
                 chrom_sizes: dict[str, int],
                 bin_size: int,
                 seq_type: str | None = None,
                 metadata: dict | None = None,
                 frozen=False):
        self.chrom_sizes = chrom_sizes
        self.bin_size = bin_size
        self.seq_type = seq_type
        self.metadata = metadata if metadata is not None else {}
        self._cmd = []
        self._ndim = None
        self._dim2_shape = None
        self._init_val = None
        self.frozen = frozen
        self.nbins = {
            chrom: int(np.ceil(cs / bin_size)) for chrom, cs in chrom_sizes.items()
        }
        self._records = {}

    def _get_arr_shape(self, chrom: str):
        if chrom not in self.chrom_sizes:
            raise KeyError('chrom not in supplied chrom_sizes')
        if self._ndim == 1:
            shape = (self.nbins[chrom], )
        elif self._ndim > 1:
            shape = (self.nbins[chrom], self._dim2_shape)
        else:
            raise NotImplementedError()
        return shape

    def _new_arr(self, chrom: str):
        return np.full(self._get_arr_shape(chrom), self._init_val, dtype=np.float32)

    def _check_arr(self, arr: np.ndarray, chrom: str):
        correct_shape = self._get_arr_shape(chrom)
        if arr.shape != correct_shape:
            raise ValueError('Array has incorrect shape')

    def _check_subdict(self, sd: dict[str, np.ndarray]):
        if not isinstance(sd, dict):
            raise ValueError('Not an dict')
        if set(sd.keys()) != set(self.chrom_sizes.keys()):
            raise ValueError('Chrom names do not match')
        for chrom, arr in sd.items():
            self._check_arr(arr, chrom)

    def get_chrom(self, chrom):
        arrs = []
        for cb in self.keys():
            try:
                m = self._records[cb][chrom]
            except KeyError:
                m = self[cb, chrom]
            arrs.append(m)
        return np.asarray(arrs)

    def iter_chrom(self, chrom):
        for cb in self.keys():
            try:
                yield cb, self._records[cb][chrom]
            except KeyError:
                yield cb, self[cb, chrom]

    def __getitem__(self, index):
        if isinstance(index, str):
            if index not in self._records:
                if self.frozen:
                    raise KeyError(f'Cell barcode {index} not in records')
                self._records[index] = {}
            return self._records[index]
        if len(index) == 2:
            cb, chrom = index
            if cb is Ellipsis:
                return self.get_chrom(chrom)
            try:
                cb_record = self._records[cb]
            except KeyError:
                cb_record = self[cb]
            if chrom not in cb_record:
                cb_record[chrom] = self._new_arr(chrom)
            return cb_record[chrom]
        if len(index) > 2:
            cb, chrom, *arr_idx = index
            try:
                m = self._records[cb][chrom]
            except KeyError:
                m = self[cb, chrom]
            if len(arr_idx) == 1:
                return m[arr_idx[0]]
            if (self._ndim == 2) and (len(arr_idx) == 2):
                return m[arr_idx[0], arr_idx[1]]
            raise KeyError('Too many indices to array')
        raise KeyError(index)

    def _ipython_key_completions_(self):
        return self._records.keys()

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self._check_subdict(value)
            self._records[index] = value
        elif isinstance(index, tuple):
            if len(index) == 2:
                cb, chrom = index
                self._check_arr(value, chrom)
                try:
                    cb_record = self._records[cb]
                except KeyError:
                    cb_record = self[cb]
                cb_record[chrom] = value
            elif len(index) > 2:
                cb, chrom, *arr_idx = index
                try:
                    m = self._records[cb][chrom]
                except KeyError:
                    m = self[cb, chrom]
                if len(arr_idx) == 1:
                    m[arr_idx[0]] = value
                elif (self._ndim == 2) and (len(arr_idx) == 2):
                    m[arr_idx[0], arr_idx[1]] = value
                else:
                    raise KeyError('Too many indices to array')
        else:
            raise KeyError(index)

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)

    @property
    def barcodes(self):
        return list(self._records.keys())

    def items(self):
        '''shallow items i.e. pairs of cb and dict[chrom: marker array]'''
        return self._records.items()

    def deep_items(self):
        '''deep items, i.e. triples of cb, chrom, and marker arrays'''
        for cb, sd in self.items():
            for chrom, arr in sd.items():
                yield cb, chrom, arr

    def keys(self):
        '''iterable of top level keys (i.e. cell barcodes)'''
        return self._records.keys()

    def values(self):
        '''iterable of top level values (i.e. dict[chrom: marker array])'''
        return self._records.values()

    def deep_values(self):
        '''iterable of low level values (i.e. marker arrays)'''
        for sd in self._records.values():
            yield from sd.values()

    def pop(self, cb):
        return self._records.pop(cb)

    @classmethod
    def new_like(cls, other):
        '''create an empty records object with metadata properties from other'''
        # do not use deepcopy directly, as this will unnecessarily copy _records
        new_instance = cls(
            copy(other.chrom_sizes),
            other.bin_size,
            copy(other.seq_type),
            deepcopy(other.metadata)
        )
        new_instance._cmd = other._cmd
        return new_instance

    def copy(self):
        '''create a copy of a records object'''
        new_instance = self.new_like(self)
        new_instance._records = deepcopy(self._records)
        return new_instance

    def add_cb_suffix(self, suffix, inplace=False):
        '''append a suffix to the cell barcodes in a records object'''
        if inplace:
            s = self
        else:
            s = self.copy()
        s._records = {f'{cb}_{suffix}': sd for cb, sd in s.items()}
        if not inplace:
            return s

    def merge(self, other, inplace=False):
        '''
        merge records from other into self.
        Cell barcodes in other that are not present in self will be created.
        where barcodes are present in both, the marker arrays will be added together.
        for PredictionRecord instances, empty positions in self will be filled with non-empty from other
        '''
        if (stype := type(self)) != (otype := type(other)):
            raise ValueError(
                f'cannot merge {stype.__name__} object with object of type {otype.__name__}'
            )
        if self.chrom_sizes != other.chrom_sizes:
            raise ValueError('chrom_sizes do not match')
        if self.bin_size != other.bin_size:
            raise ValueError('bin_sizes do not match')
        if self.seq_type != other.seq_type:
            log.warning(
                'merged datasets do not appear to be the same sequencing data type: '
                f'{self.seq_type} and {other.seq_type}'
            )

        if inplace:
            s = self
        else:
            s = self.copy()

        for cb, cb_m in other.items():
            for chrom, m in cb_m.items():
                s_m = s[cb, chrom]
                if np.isnan(self._init_val):
                    mask = np.isnan(s_m) & np.isfinite(m)
                    s_m[mask] = m[mask]
                else:
                    s_m += m

        if not inplace:
            return s

    def filter(self, cb_whitelist, inplace=True):
        cb_whitelist = set(cb_whitelist)
        if inplace:
            for cb in self.barcodes:
                if cb not in cb_whitelist:
                    self._records.pop(cb)
            return None
        else:
            obj = self.new_like(self)
            for cb in cb_whitelist:
                obj._records[cb] = self._records[cb]
            return obj

    def groupby(self, by):
        if not callable(by):
            if isinstance(by, dict):
                grouper_func = by.get
            else:
                raise ValueError('grouper should be dict or callable')
        else:
            grouper_func = by
        group_mapping = defaultdict(list)
        for cb in self.barcodes:
            g = grouper_func(cb)
            if g is not None:
                group_mapping[g].append(cb)
        for g, g_cb in group_mapping.items():
            g_obj = self.filter(g_cb, inplace=False)
            yield g, g_obj

    def __add__(self, other):
        if isinstance(other, BaseRecords):
            return self.merge(other, inplace=False)
        raise NotImplementedError()

    def __iadd__(self, other):
        if isinstance(other, BaseRecords):
            return self.merge(other, inplace=True)
        raise NotImplementedError()

    @classmethod
    def _records_to_json(cls, records, precision):
        json_serialisable = {}
        for cb, sd in records.items():
            d = {}
            for chrom, arr in sd.items():
                d[chrom] = records._arr_to_json(arr, precision)
            json_serialisable[cb] = d
        return json_serialisable

    @classmethod
    def _metadata_to_json(cls, obj, precision):
        if isinstance(obj, dict):
            json_serialisable = {}
            for key, val in obj.items():
                json_serialisable[key] = cls._metadata_to_json(val, precision)
        elif isinstance(obj, BaseRecords):
            json_serialisable = obj._records_to_json(obj, precision)
        elif isinstance(obj, (list, tuple, set, frozenset, np.ndarray)):
            json_serialisable = []
            for val in obj:
                json_serialisable.append(cls._metadata_to_json(val, precision))
        elif isinstance(obj, (float, np.floating)):
            json_serialisable = round(float(obj), precision)
        elif isinstance(obj, np.integer):
            json_serialisable = int(obj)
        elif isinstance(obj, (str, int, bool)) or obj is None:
            json_serialisable = obj
        elif isinstance(obj, np.bool_):
            json_serialisable = bool(obj)
        else:
            raise NotImplementedError(f'json serialisation not implemented for type: {type(obj)}')
        return json_serialisable

    def _arr_to_json(self, arr, precision=0):
        raise NotImplementedError()

    def _json_to_arr(self, obj, chrom):
        raise NotImplementedError()

    def to_json(self, precision: int = 2):
        '''dump records obejct to a json string'''
        return json.dumps({
            'dtype': self.__class__.__name__,
            'cmd': self._cmd + [' '.join(sys.argv)],
            'bin_size': self.bin_size,
            'sequencing_data_type': self.seq_type,
            'chrom_sizes': self.chrom_sizes,
            'shape': self.nbins,
            'data': self._records_to_json(self, precision),
            'metadata': self._metadata_to_json(self.metadata, precision)
        })

    def write_json(self, fp: str, precision: int = 2):
        '''write json representation of a MarkerRecords obejct to a file'''
        with open(fp, 'w') as f:
            f.write(self.to_json(precision=precision))

    @classmethod
    def read_json(cls, fp: str, subset: list | set | None = None, frozen=False):
        '''read records object from a json file'''
        with open(fp) as f:
            obj = json.load(f)
        if obj['dtype'] != cls.__name__:
            raise ValueError(f'json file does not match signature for {cls.__name__}')
        new_instance = cls(obj['chrom_sizes'],
                           obj['bin_size'],
                           seq_type=obj['sequencing_data_type'],
                           metadata=obj['metadata'],
                           frozen=frozen)
        new_instance._cmd = obj['cmd']
        if subset is None:
            subset = obj['data'].keys()
        for cb in subset:
            try:
                sd = obj['data'][cb]
            except KeyError as exc:
                raise KeyError(f'Cell barcode {cb} not in {fp}') from exc
            # directly access self._records for speed
            new_instance._records[cb] = {}
            for chrom, d in sd.items():
                new_instance._records[cb][chrom] = new_instance._json_to_arr(d, chrom)
        return new_instance


class MarkerRecords(BaseRecords):

    def __init__(self,
                 chrom_sizes: dict[str, int],
                 bin_size: int,
                 seq_type: str | None = None,
                 metadata: dict | None = None,
                 frozen: bool = False):
        super().__init__(chrom_sizes, bin_size, seq_type, metadata, frozen)
        self._ndim = 2
        self._dim2_shape = 2
        self._init_val = 0.0

    def update(self, interval_counts):
        '''
        update MarkerRecords object with values from a IntervalMarkerCounts object.
        '''
        if not isinstance(interval_counts, IntervalMarkerCounts):
            raise ValueError(
                f'can only update {type(self).__name__} with IntervalMarkerCounts object'
            )
        chrom, bin_idx = interval_counts.chrom, interval_counts.bin_idx
        for cb, hap, val in interval_counts.deep_items():
            self[cb, chrom, bin_idx, hap] += val
        return self

    def __iadd__(self, other):
        if isinstance(other, BaseRecords):
            return self.merge(other, inplace=True)
        if isinstance(other, IntervalMarkerCounts):
            return self.update(other)
        raise NotImplementedError()

    def total_marker_count(self, cb):
        tot = 0
        for m in self[cb].values():
            tot += m.sum(axis=None)
        return tot

    def _arr_to_json(self, arr, precision=0):
        idx = np.nonzero(arr.ravel())[0]
        val = arr.ravel()[idx]
        val = [round(float(v), precision) for v in val]
        return (idx.tolist(), val)

    def _json_to_arr(self, obj, chrom):
        idx, val = obj
        arr = np.zeros(shape=self.nbins[chrom] * 2, dtype=np.float32)
        arr[idx] = val
        return arr.reshape(self.nbins[chrom], 2)


class PredictionRecords(BaseRecords):

    def __init__(self,
                 chrom_sizes: dict[str, int],
                 bin_size: int,
                 seq_type: str | None = None,
                 metadata: dict | None = None,
                 frozen: bool = False):
        super().__init__(chrom_sizes, bin_size, seq_type, metadata, frozen)
        self._ndim = 1
        self._dim2_shape = np.nan
        self._init_val = np.nan

    def _arr_to_json(self, arr, precision=2):
        return [round(float(v), precision) for v in arr]

    def _json_to_arr(self, obj, chrom):
        return np.array(obj)

    def to_frame(self):
        '''Convert PredictionRecords object to a pandas.DataFrame'''
        frame = []
        index = []
        columns = pd.MultiIndex.from_tuples(
            [(chrom, i * self.bin_size)
             for chrom in self.chrom_sizes
             for i in range(self.nbins[chrom])],
            names=['chrom', 'pos']
        )
        for cb in self.barcodes:
            index.append(cb)
            frame.append(np.concatenate([self[cb, chrom] for chrom in self.chrom_sizes]))
        return pd.DataFrame(frame, index=index, columns=columns)
