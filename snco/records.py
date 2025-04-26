import logging
import sys
import inspect
from copy import copy, deepcopy
from collections import defaultdict
import json

import numpy as np
from scipy import sparse
import pandas as pd

from .bam import IntervalMarkerCounts
from .metadata import MetadataDict
from .groupby import RecordsGroupyBy

log = logging.getLogger('snco')


class BaseRecords(object):

    """
    Base class for `MarkerRecords` and `PredictionRecords`.

    This class stores cell barcode records across chromosomes with data represented
    in numpy arrays. It supports indexing, merging, filtering, grouping, and serialization.
    """

    def __init__(self,
                 chrom_sizes: dict[str, int],
                 bin_size: int,
                 seq_type: str | None = None,
                 metadata: dict | None = None,
                 frozen=False):
        """
        Parameters
        ----------
        chrom_sizes : dict of str to int
            Dictionary mapping chromosome names to chromosome sizes.
        bin_size : int
            The size of each genomic bin.
        seq_type : str or None, optional
            A string describing the sequencing data type.
        metadata : dict of snco.metadata.MetadataDict or None, optional
            Additional metadata for the record set.
        frozen : bool, default=False
            If True, prevents creation of new keys in the records.
        """
        self.chrom_sizes = chrom_sizes
        self.bin_size = bin_size
        self.seq_type = seq_type
        self.metadata = {}
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise ValueError('metadata must be dict or None')
            self.add_metadata(**metadata)
                    
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

    def get_chrom(self, chrom, cb_whitelist=None):
        """
        Return a 2D array of all records for a given chromosome.

        Parameters
        ----------
        chrom : str
            Chromosome name.

        Returns
        -------
        np.ndarray
            Array of shape (n_cell_barcodes, nbins) or (n_cell_barcodes, nbins, dim2_shape).
        """
        arrs = []
        if cb_whitelist is None:
            cb_whitelist = self.barcodes
        for cb in cb_whitelist:
            try:
                m = self._records[cb][chrom]
            except KeyError:
                m = self[cb, chrom]
            arrs.append(m)
        return np.asarray(arrs)

    def iter_chrom(self, chrom):
        """
        Iterate over all cell barcodes for a given chromosome.

        Parameters
        ----------
        chrom : str
            Chromosome name.

        Yields
        ------
        tuple
            (cell barcode, numpy array) for each record.
        """
        for cb in self.keys():
            try:
                yield cb, self._records[cb][chrom]
            except KeyError:
                yield cb, self[cb, chrom]
    
    def _get_cb_record(self, cb):
        if cb not in self._records:
            if self.frozen:
                raise KeyError(f"Cell barcode '{cb}' not found")
            self._records[cb] = {}
        return self._records[cb]

    def _get_or_create_array(self, cb, chrom):
        cb_record = self._get_cb_record(cb)
        if chrom not in cb_record:
            if chrom not in self.chrom_sizes:
                raise ValueError(f"Chromosome {chrom} not in chrom_sizes")
            cb_record[chrom] = self._new_arr(chrom)
        return cb_record[chrom]

    def _getitem_tuple(self, index):
        if len(index) == 2:
            cb, chrom = index
            if not isinstance(cb, str):
                if cb is Ellipsis:
                    return self.get_chrom(chrom)
                elif isinstance(cb, (list, tuple, set, frozenset, np.ndarray)):
                    return self.get_chrom(chrom, cb_whitelist=list(cb))
                raise KeyError(f'Invalid index type: {cb}')
            return self._get_or_create_array(cb, chrom)
    
        elif len(index) > 2:
            cb, chrom, *arr_idx = index
            if len(arr_idx) > self._ndim:
                raise IndexError(
                    f"Too many indices ({len(arr_idx)}) for array with ndim={self._ndim}"
                )
            arr = self._get_or_create_array(cb, chrom)
            return arr[tuple(arr_idx)]  # handles 1D/2D cleanly

        raise IndexError(f"Too many indices: {index}")

    def __getitem__(self, index):
        if isinstance(index, str):
            return self._get_cb_record(index)
        
        if isinstance(index, tuple):
            return self._getitem_tuple(index)
    
        raise KeyError(f"Invalid index type: {index}")

    def _set_cb_record(self, cb, val):
        self._check_subdict(val)
        self._records[cb] = val
    
    def _setitem_tuple(self, index, value):
        if len(index) == 2:
            cb, chrom = index
            self._check_arr(value, chrom)
            self._get_cb_record(cb)[chrom] = value
    
        elif len(index) > 2:
            cb, chrom, *arr_idx = index
            if len(arr_idx) > self._ndim:
                raise IndexError(f"Too many indices ({len(arr_idx)}) for array with ndim={self._ndim}")
            arr = self._get_or_create_array(cb, chrom)
            arr[tuple(arr_idx)] = value
        else:
            raise KeyError(f"Invalid tuple index: {index}")

    def __setitem__(self, index, value):
        if isinstance(index, str):
            return self._set_cb_record(index, value)

        if isinstance(index, tuple):
            return self._setitem_tuple(index, value)
    
        raise KeyError(f"Invalid index: {index}")

    def add_metadata(self, **metadata):
        for key, value in metadata.items():
            if isinstance(value, MetadataDict):
                self.metadata[key] = value
            elif isinstance(value, dict):
                self.metadata[key] = MetadataDict.from_json(value)
            else:
                raise ValueError('metadata values must be MetadataDict objects or correctly formatted dicts')

    def __contains__(self, cb):
        return cb in self._records
    
    def __delitem__(self, cb):
        del self._records[cb]

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)

    def __repr__(self):
        n_cells = len(self._records)
        chroms = ', '.join(list(self.chrom_sizes))
        is_frozen = self.frozen
        return (f"{self.__class__.__qualname__}(cells: {n_cells}, "
                f"chromosomes: {{{chroms}, }}, frozen: {is_frozen})")

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

    def deep_keys(self):
        '''iterable of deep keys (cell barcode, chrom pairs)'''
        for cb, sd in self._records.items():
            for chrom in sd:
                yield chrom, sd

    def values(self):
        '''iterable of top level values (i.e. dict[chrom: marker array])'''
        return self._records.values()

    def deep_values(self):
        '''iterable of low level values (i.e. marker arrays)'''
        for sd in self._records.values():
            yield from sd.values()

    def pop(self, cb):
        '''remove and return a barcode from the dataset'''
        return self._records.pop(cb)

    @classmethod
    def new_like(cls, other):
        """
        Create a new empty object like another `BaseRecords` instance.

        Parameters
        ----------
        other : BaseRecords
            Template object to mimic.

        Returns
        -------
        BaseRecords
            New instance with copied metadata and structure.
        """
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
        '''
        create a copy of a records object
        
        Returns
        -------
        BaseRecords
            New instance with copied metadata and structure.
        '''
        new_instance = self.new_like(self)
        new_instance._records = deepcopy(self._records)
        return new_instance

    def _add_cb_suffix_metadata(self, suffix):
        for key, val in self.metadata.items():
            if 'cb' in val.levels:
                self.metadata[key] = val.add_level_suffix(suffix, level='cb')

    def add_cb_suffix(self, suffix, inplace=False):
        '''append a suffix to the cell barcodes in a records object'''
        if inplace:
            s = self
        else:
            s = self.copy()
        s._records = {f'{cb}_{suffix}': sd for cb, sd in s.items()}
        s._add_cb_suffix_metadata(suffix)

        if not inplace:
            return s

    def _merge_metadata(self, other):
        for key, val in other.metadata.items():
            if key in self.metadata:
                self.metadata[key].update(val)
            else:
                self.metadata[key] = val

    def merge(self, other, inplace=False):
        '''
        merge records from other into self.
        Cell barcodes in other that are not present in self will be created.
        where barcodes are present in both, the marker arrays will be added together.
        NaN positions in self will be filled with non-NaN from other

        Parameters
        ----------
        other : BaseRecords
            Another records object to merge.
        inplace : bool, default=False
            Whether to modify this object in-place.

        Returns
        -------
        BaseRecords or None
            The merged object or None if in-place.
        '''
        if (stype := type(self)) != (otype := type(other)):
            raise ValueError(
                f'cannot merge {stype.__qualname__} object with object of type {otype.__qualname__}'
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

        for cb, chrom, m in other.deep_items():
            s_m = s[cb, chrom]
            if np.isnan(self._init_val):
                mask = np.isnan(s_m) & np.isfinite(m)
                s_m[mask] = m[mask]
            else:
                s_m += m

        s._merge_metadata(other)

        if not inplace:
            return s

    def _filter_metadata(self, cb_whitelist):
        for mdata in self.metadata.values():
            if 'cb' in mdata.levels:
                mdata.filter(cb_whitelist, level='cb')

    def filter(self, cb_whitelist, inplace=True):
        """
        Filter the records to include only cell barcodes in the provided whitelist.

        Parameters
        ----------
        cb_whitelist : list or set
            A collection of cell barcodes to retain in the records.
        inplace : bool, default=True
            If True, modifies the current object in place.
            If False, returns a new filtered instance.

        Returns
        -------
        BaseRecords or None
            The filtered records object if `inplace=False`, otherwise None.
        """
        cb_whitelist = set(cb_whitelist)
        if inplace:
            for cb in self.barcodes:
                if cb not in cb_whitelist:
                    self._records.pop(cb)
            self._filter_metadata(cb_whitelist)
            return None
        else:
            obj = self.new_like(self)
            for cb in cb_whitelist:
                obj._records[cb] = self._records[cb]
            obj._filter_metadata(cb_whitelist)
            return obj

    def query(self, func):
        """
        Query the records using a function applied to each cell barcode,
        optionally supplying additional context based on the function signature.

        The provided `func` must accept at least a cell barcode (str) as its first argument.
        If `func` declares an additional named parameter called `records`, the full records object
        will be passed in automatically.
        If `func` declares additional named parameters matching keys in `self.metadata`
        these will also be passed in automatically.

        Parameters
        ----------
        func : callable
            A function that accepts a cell barcode (str) as its first argument.
            It may also accept additional keyword arguments matching metadata keys
            or 'records' to access the full records object.

        Returns
        -------
        BaseRecords
            A new records object containing only the barcodes for which `func` returns True.
        """
        func_signature = inspect.signature(func)
        additional_parameters = func_signature.parameters 
        additional_param_names = list(additional_parameters)[1:] # first argument is always cell barcode
        func_kwargs = {}
        for param in additional_param_names:
            if param == 'records':
                func_kwargs[param] = self
            elif param in self.metadata:
                func_kwargs[param] = self.metadata[param]
        return self.filter(
            (cb for cb in self.barcodes if func(cb, **func_kwargs)),
            inplace=False
        )

    def groupby(self, by):
        """
        Group records by a provided mapping from barcodes to group labels.

        Parameters
        ----------
        by : str, dict-like or callable
            Grouping strategy. Can be a string ('none', 'genotype'), a dictionary
            mapping cell barcodes to group labels, or a callable that takes a barcode
            and returns a group label.
            When callable, `by` must accept at least a cell barcode (str) as its first argument.
            If `by` declares an additional named parameter called `records`, the full records object
            will be passed in automatically.
            If `by` declares additional named parameters matching keys in `self.metadata`
            these will also be passed in automatically.

        Returns
        -------
        RecordsGroupyBy
            A RecordsGroupyBy object that allows group-wise operations over the records.

        Raises
        ------
        ValueError
            If the grouper is not recognized or invalid.
        """
        return RecordsGroupyBy(self, by)

    def __add__(self, other):
        if isinstance(other, BaseRecords):
            return self.merge(other, inplace=False)
        raise NotImplementedError()

    def __iadd__(self, other):
        if isinstance(other, BaseRecords):
            return self.merge(other, inplace=True)
        raise NotImplementedError()

    def _records_to_json(self, precision):
        json_serialisable = {}
        for cb, sd in self._records.items():
            d = {}
            for chrom, arr in sd.items():
                d[chrom] = self._arr_to_json(arr, precision)
            json_serialisable[cb] = d
        return json_serialisable

    def _arr_to_json(self, arr, precision=0):
        raise NotImplementedError()

    def _json_to_arr(self, obj, chrom):
        raise NotImplementedError()

    def _metadata_to_json(self, precision):
        return {
            name: metadata.to_json(precision) for name, metadata in self.metadata.items()
        }

    def to_json(self, precision: int = 5):
        """
        Convert the records object to a JSON string.

        Parameters
        ----------
        precision : int, optional
            Decimal precision for floats.

        Returns
        -------
        str
            JSON-formatted string.
        """
        return json.dumps({
            'dtype': self.__class__.__qualname__,
            'cmd': self._cmd + [' '.join(sys.argv)],
            'bin_size': self.bin_size,
            'sequencing_data_type': self.seq_type,
            'chrom_sizes': self.chrom_sizes,
            'shape': self.nbins,
            'data': self._records_to_json(precision),
            'metadata': self._metadata_to_json(precision)
        })

    def write_json(self, fp: str, precision: int = 2):
        """
        Write JSON representation to file.

        Parameters
        ----------
        fp : str
            File path.
        precision : int, optional
            Decimal precision for floats.
        """
        with open(fp, 'w') as f:
            f.write(self.to_json(precision=precision))

    @classmethod
    def read_json(cls, fp: str, subset: list | set | None = None, frozen=False):
        """
        Read a `BaseRecords` object from a JSON file.

        Parameters
        ----------
        fp : str
            File path.
        subset : list or set, optional
            Subset of cell barcodes to load.
        frozen : bool, default=False
            Whether to freeze the object after loading.

        Returns
        -------
        BaseRecords
            The loaded records object.

        Raises
        ------
        ValueError
            If the file contents do not match the expected class.
        """
        with open(fp) as f:
            obj = json.load(f)
        if obj['dtype'] != cls.__qualname__:
            raise ValueError(f'json file does not match signature for {cls.__qualname__}')
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
    """
    Records storage class for interval-based marker counts.

    This class stores cell barcode records across chromosomes with data represented
    in numpy arrays. It supports indexing, merging, filtering, grouping, and serialization.

    Attributes
    ----------

    chrom_sizes : dict of str to int
        Dictionary mapping chromosome names to chromosome sizes.
    bin_size : int
        The size of each genomic bin.
    nbins : dict of str to int
        Dictionary mapping chromosome names to the number of bins per chromosome.
    seq_type : str or None
        A string describing the sequencing data type.
    metadata : dict
        Additional metadata for the record set.
    frozen : bool
        If True, prevents creation of new keys in the records.
    barcodes: list
        List of cell barcodes stored in the Records object

    Methods
    -------
    keys()
        Returns iterable of barcodes for object
    deep_keys()
        Returns iterable of (barcode, chrom) pairs for object
    values()
        Return iterable of dicts containing chrom: array mappings for object
    deep_values()
        Return iterable of array data for object
    items()
        Return iterable of barcode value pairs where values are dicts containing chrom: array mappings
    deep_items()
        Return iterable of (barcode, chrom, array) triplets
    pop(cb)
        Remove and return barcode data from object
    copy()
        Make a copy of self
    filter(barcodes, inplace=False)
        Filters a BaseRecords object to create one containing only the specified barcodes.
    new_like(template)
        Returns a new, empty BaseRecords object with the same structure as `template`.
    merge(other, inplace=False)
        Merges another BaseRecords into the current one.
    query(expr)
        Filters records based on an expression or condition.
    update(interval_counts)
        Update the object with counts from an IntervalCounts object
    groupby(grouper)
        Groups records using a given grouper function or strategy.
    get_chrom(chrom)
        Return a 2D array of all records for a given chromosome.
    iter_chrom(chrom)
        Iterate over all cell barcodes for a given chromosome.
    total_marker_count(cb)
        Return the total number of markers for a barcode across all chromosomes
    add_cb_suffix(suffix, inplace=False)
        Append a suffix to all cell barcodes in object
    to_json(precision=2)
        Convert the records object to a JSON string.
    write_json(fp, precision=2)
        Write the records object to a JSON file path.
    read_json(fp, subset=None, frozen=False)
        Read a BaseRecords object from a file path
    """

    def __init__(self,
                 chrom_sizes: dict[str, int],
                 bin_size: int,
                 seq_type: str | None = None,
                 metadata: dict | None = None,
                 frozen: bool = False):
        """
        Records storage class for interval-based marker counts.

        This class stores cell barcode records across chromosomes with data represented
        in numpy arrays. It supports indexing, merging, filtering, grouping, and serialization.

        Parameters
        ----------
        chrom_sizes : dict of str to int
            Dictionary mapping chromosome names to chromosome sizes.
        bin_size : int
            The size of each genomic bin.
        seq_type : str or None, optional
            A string describing the sequencing data type.
        metadata : dict or None, optional
            Additional metadata for the record set.
        frozen : bool, default=False
            If True, prevents creation of new keys in the records.
        """
        super().__init__(chrom_sizes, bin_size, seq_type, metadata, frozen)
        self._ndim = 2
        self._dim2_shape = 2
        self._init_val = 0.0

    def update(self, interval_counts):
        """
        Update this object with values from an `IntervalMarkerCounts` object.

        Parameters
        ----------
        interval_counts : IntervalMarkerCounts
            Object containing new data to merge in.

        Returns
        -------
        MarkerRecords
            Self, after update.

        Raises
        ------
        ValueError
            If input is not an `IntervalMarkerCounts` instance.
        """
        if not isinstance(interval_counts, IntervalMarkerCounts):
            raise ValueError(
                f'can only update {type(self).__qualname__} with IntervalMarkerCounts object'
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
        """
        Compute the total marker count across all chromosomes for a given cell barcode.

        Parameters
        ----------
        cb : str
            Cell barcode.

        Returns
        -------
        float
            Sum of marker counts for barcode across all chromosomes.
        """
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
    """
    Records storage class for interval-based haplotype predictions.

    This class stores cell barcode records across chromosomes with data represented
    in numpy arrays. It supports indexing, merging, filtering, grouping, and serialization.

    Attributes
    ----------

    chrom_sizes : dict of str to int
        Dictionary mapping chromosome names to chromosome sizes.
    bin_size : int
        The size of each genomic bin.
    nbins : dict of str to int
        Dictionary mapping chromosome names to the number of bins per chromosome.
    seq_type : str or None
        A string describing the sequencing data type.
    metadata : dict
        Additional metadata for the record set.
    frozen : bool
        If True, prevents creation of new keys in the records.
    barcodes: list
        List of cell barcodes stored in the Records object

    Methods
    -------
    keys()
        Returns iterable of barcodes for object
    deep_keys()
        Returns iterable of (barcode, chrom) pairs for object
    values()
        Return iterable of dicts containing chrom: array mappings for object
    deep_values()
        Return iterable of array data for object
    items()
        Return iterable of barcode value pairs where values are dicts containing chrom: array mappings
    deep_items()
        Return iterable of (barcode, chrom, array) triplets
    pop(cb)
        Remove and return barcode data from object
    copy()
        Make a copy of self
    filter(barcodes, inplace=False)
        Filters a BaseRecords object to create one containing only the specified barcodes.
    new_like(template)
        Returns a new, empty BaseRecords object with the same structure as `template`.
    merge(other, inplace=False)
        Merges another BaseRecords into the current one.
    query(expr)
        Filters records based on an expression or condition.
    groupby(grouper)
        Groups records using a given grouper function or strategy.
    get_chrom(chrom)
        Return a 2D array of all records for a given chromosome.
    get_haplotype(chrom, pos, cb_whitelist)
        Retrieve prediction values across cell barcodes for a specific position.
    iter_chrom(chrom)
        Iterate over all cell barcodes for a given chromosome.
    add_cb_suffix(suffix, inplace=False)
        Append a suffix to all cell barcodes in object
    to_json(precision=2)
        Convert the records object to a JSON string.
    to_frame(cb_whitelist=None):
        Convert the records object to a pandas DataFrame
    write_json(fp, precision=2)
        Write the records object to a JSON file path.
    read_json(fp, subset=None, frozen=False)
        Read a BaseRecords object from a file path
    """

    def __init__(self,
                 chrom_sizes: dict[str, int],
                 bin_size: int,
                 seq_type: str | None = None,
                 metadata: dict | None = None,
                 frozen: bool = False):
        """
        Records storage class for interval-based haplotype predictions.

        This class stores cell barcode records across chromosomes with data represented
        in numpy arrays. It supports indexing, merging, filtering, grouping, and serialization.

        Parameters
        ----------
        chrom_sizes : dict of str to int
            Dictionary mapping chromosome names to chromosome sizes.
        bin_size : int
            The size of each genomic bin.
        seq_type : str or None, optional
            A string describing the sequencing data type.
        metadata : dict or None, optional
            Additional metadata for the record set.
        frozen : bool, default=False
            If True, prevents creation of new keys in the records.
        """
        super().__init__(chrom_sizes, bin_size, seq_type, metadata, frozen)
        self._ndim = 1
        self._dim2_shape = np.nan
        self._init_val = np.nan

    def _arr_to_json(self, arr, precision=2):
        return [round(float(v), precision) for v in arr]

    def _json_to_arr(self, obj, chrom):
        return np.array(obj)

    def to_frame(self, cb_whitelist=None):
        """
        Convert the `PredictionRecords` object to a pandas DataFrame.

        Parameters
        ----------
        cb_whitelist : list or None, optional
            List of cell barcodes to filter.

        Returns
        -------
        pd.DataFrame
            DataFrame with shape (n_cells, n_bins) and multi-indexed columns with levels (chrom, pos).
        """
        frame = []
        columns = pd.MultiIndex.from_tuples(
            [(chrom, i * self.bin_size)
             for chrom in self.chrom_sizes
             for i in range(self.nbins[chrom])],
            names=['chrom', 'pos']
        )
        if cb_whitelist is None:
            cb_whitelist = self.barcodes
        for cb in cb_whitelist:
            frame.append(np.concatenate([self[cb, chrom] for chrom in self.chrom_sizes]))
        return pd.DataFrame(frame, index=cb_whitelist, columns=columns)

    def get_haplotype(self, chrom, pos, cb_whitelist=None):
        """
        Retrieve prediction values across cell barcodes for a specific position.

        Parameters
        ----------
        chrom : str
            Chromosome name.
        pos : int
            Genomic position in basepairs.
        cb_whitelist : list or None, optional
            List of cell barcodes to filter.

        Returns
        -------
        pd.Series
            Series of prediction values indexed by cell barcode.
        """
        idx = int(pos // self.bin_size)
        series = []
        if cb_whitelist is None:
            cb_whitelist = self.barcodes
        for cb in cb_whitelist:
            series.append(self._records[cb][chrom][idx])
        return pd.Series(series, index=cb_whitelist, name=f'{chrom}:{pos:d}')
