import sys
from copy import copy, deepcopy
import json

import numpy as np
import pandas as pd

from .bam import IntervalCountsDeduped

class BaseRecords:

    def __init__(self,
                 chrom_sizes: dict[str, int],
                 bin_size: int,
                 cb_whitelist: set[str] | None = None,
                 metadata: dict | None = None):
        self.chrom_sizes = chrom_sizes
        self.bin_size = bin_size
        self.cb_whitelist = cb_whitelist
        self.metadata = metadata if metadata is not None else {}
        self._cmd = []
        self._ndim = None
        self._dim2_shape = None
        self._init_val = None
        self.nbins = {
            chrom: int(np.ceil(cs / bin_size)) for chrom, cs in chrom_sizes.items()
        }
        self._records = {}

    def _check_whitelist(self, cb: str, raise_error: bool = False):
        if self.cb_whitelist is None or cb in self.cb_whitelist:
            return True
        if raise_error:
            raise KeyError(f'{cb} not in supplied whitelist')
        return False

    def _get_arr_shape(self, chrom: str):
        if chrom not in self.chrom_sizes:
            raise KeyError('chrom not in supplied chrom_sizes')
        if self._ndim == 1:
            shape = (self.nbins[chrom], )
        elif self._ndim == 2:
            shape = (self.nbins[chrom], self._dim2_shape)
        else:
            raise NotImplementedError()
        return shape

    def _new_arr(self, chrom: str):
        # todo: implement sparse form
        return np.full(self._get_arr_shape(chrom), self._init_val, dtype=np.float64)

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

    def __getitem__(self, index):
        if isinstance(index, str):
            self._check_whitelist(index, raise_error=True)
            if index not in self._records:
                self._records[index] = {}
            return self._records[index]
        if len(index) == 2:
            cb, chrom = index
            cb_record = self[cb]
            if chrom not in cb_record:
                cb_record[chrom] = self._new_arr(chrom)
            return cb_record[chrom]
        if len(index) > 2:
            cb, chrom, *arr_idx = index
            m = self[cb, chrom]
            if len(arr_idx) == 1:
                return m[arr_idx[0]]
            if (self._ndim == 2) and (len(arr_idx) == 2):
                return m[arr_idx[0], arr_idx[1]]
            raise KeyError('Too many indices to array')
        raise KeyError(index)

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self._check_whitelist(index, raise_error=True)
            self._check_subdict(value)
            self._records[index] = value
        elif isinstance(index, tuple):
            if len(index) == 2:
                cb, chrom = index
                self._check_whitelist(cb, raise_error=True)
                self._check_arr(value, chrom)
                cb_m = self[cb]
                cb_m[chrom] = value
            elif len(index) > 2:
                cb, chrom, *arr_idx = index
                arr = self[cb, chrom]
                if len(arr_idx) == 1:
                    arr[arr_idx[0]] = value
                elif (self._ndim == 2) and (len(arr_idx) == 2):
                    arr[arr_idx[0], arr_idx[1]] = value
                else:
                    raise KeyError('Too many indices to array')
        else:
            raise KeyError(index)

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)

    @property
    def seen_barcodes(self):
        return list(self._records.keys())

    def items(self):
        return self._records.items()

    def deep_items(self):
        for cb, sd in self.items():
            for chrom, arr in sd.items():
                yield cb, chrom, arr

    def values(self):
        return self._records.values()

    def deep_values(self):
        for sd in self._records.values():
            yield from sd.values()

    def set_cb_whitelist(self, new_whitelist):
        self.cb_whitelist = new_whitelist
        self._records = {
            cb: val for cb, val in self.items()
            if self._check_whitelist(cb)
        }

    @classmethod
    def new_like(cls, other):
        new_instance = cls(
            copy(other.chrom_sizes),
            other.bin_size,
            deepcopy(other.cb_whitelist),
            deepcopy(other.metadata)
        )
        new_instance._cmd = other._cmd
        return new_instance

    def copy(self):
        new_instance = self.new_like(self)
        new_instance._records = deepcopy(self._records)
        return new_instance

    def merge(self, other, inplace=False):
        if self.chrom_sizes != other.chrom_sizes:
            raise ValueError('chrom_sizes do not match')
        if self.bin_size != other.bin_size:
            raise ValueError('bin_sizes do not match')

        if inplace:
            s = self
        else:
            s = self.copy()

        for cb, cb_m in other.items():
            for chrom, m in cb_m.items():
                s[cb, chrom] += m

        if not inplace:
            return s

    def update(self, interval_counts):
        chrom, bin_idx = interval_counts.chrom, interval_counts.bin_idx
        for cb, hap, val in interval_counts.deep_items():
            self[cb, chrom, bin_idx, hap] += val
        return self

    def __add__(self, other):
        if isinstance(other, BaseRecords):
            return self.merge(other, inplace=False)
        raise NotImplementedError()

    def __iadd__(self, other):
        if isinstance(other, BaseRecords):
            return self.merge(other, inplace=True)
        if isinstance(other, IntervalCountsDeduped):
            return self.update(other)
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
        elif isinstance(obj, (list, tuple, np.ndarray)):
            json_serialisable = []
            for val in obj:
                json_serialisable.append(cls._metadata_to_json(val, precision))
        elif isinstance(obj, float):
            json_serialisable = round(float(obj), precision)
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
        return json.dumps({
            'dtype': self.__class__.__name__,
            'cmd': self._cmd + [' '.join(sys.argv)],
            'bin_size': self.bin_size,
            'chrom_sizes': self.chrom_sizes,
            'cb_whitelist': list(self.cb_whitelist) if self.cb_whitelist is not None else None,
            'shape': self.nbins,
            'data': self._records_to_json(self, precision),
            'metadata': self._metadata_to_json(self.metadata, precision)
        })

    def write_json(self, fp: str, precision: int = 2):
        with open(fp, 'w') as f:
            f.write(self.to_json(precision=precision))

    @classmethod
    def read_json(cls, fp: str):
        with open(fp) as f:
            obj = json.load(f)
        if obj['dtype'] != cls.__name__:
            raise ValueError(f'json file does not match signature for {cls.__name__}')
        new_instance = cls(obj['chrom_sizes'],
                           obj['bin_size'],
                           obj['cb_whitelist'],
                           obj['metadata'])
        new_instance._cmd = obj['cmd']
        for cb, sd in obj['data'].items():
            for chrom, d in sd.items():
                new_instance[cb, chrom] = new_instance._json_to_arr(d, chrom)
        return new_instance


class MarkerRecords(BaseRecords):

    def __init__(self,
                 chrom_sizes: dict[str, int],
                 bin_size: int,
                 cb_whitelist: set[str] | None = None,
                 metadata: dict | None = None):
        super().__init__(chrom_sizes, bin_size, cb_whitelist, metadata)
        self._ndim = 2
        self._dim2_shape = 2
        self._init_val = 0.0

    def _arr_to_json(self, arr, precision=0):
        idx = np.nonzero(arr.ravel())[0]
        val = arr.ravel()[idx]
        val = [round(float(v), precision) for v in val]
        return (idx.tolist(), val)

    def _json_to_arr(self, obj, chrom):
        idx, val = obj
        arr = np.zeros(shape=self.nbins[chrom] * 2, dtype=np.float64)
        arr[idx] = val
        return arr.reshape(self.nbins[chrom], 2)


class PredictionRecords(BaseRecords):

    def __init__(self,
                 chrom_sizes: dict[str, int],
                 bin_size: int,
                 cb_whitelist: set[str] | None = None,
                 metadata: dict | None = None):
        super().__init__(chrom_sizes, bin_size, cb_whitelist, metadata)
        self._ndim = 1
        self._dim2_shape = np.nan
        self._init_val = np.nan

    def _arr_to_json(self, arr, precision=2):
        return [round(float(v), precision) for v in arr]

    def _json_to_arr(self, obj, chrom):
        return np.array(obj)

    def to_frame(self):
        frame = []
        index = []
        columns = pd.MultiIndex.from_tuples(
            [(chrom, i * self.bin_size)
             for chrom in self.chrom_sizes
             for i in range(self.nbins[chrom])],
            names=['chrom', 'pos']
        )
        for cb in self.seen_barcodes:
            index.append(cb)
            frame.append(np.concatenate([self[cb, chrom] for chrom in self.chrom_sizes]))
        return pd.DataFrame(frame, index=index, columns=columns)
