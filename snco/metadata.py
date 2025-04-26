from copy import copy, deepcopy
import numpy as np


METADATA_DTYPES = {
    'str': str, 'int': int, 'float': float, 'bool': bool,
    'list': list, 'tuple': tuple,
    'set': set, 'frozenset': frozenset,
    'ndarray': np.ndarray,
}


def validate_data(obj, expected_depth, expected_dtype):
    """
    Recursively validates that the object conforms to the expected depth and data type.

    Parameters
    ----------
    obj : dict or any
        The object to validate.
    expected_depth : int
        The expected depth of the object.
    expected_dtype : type
        The expected data type for the values in the object.

    Raises
    ------
    ValueError
        If the object depth or data type does not match the expected values.
    """
    def _validate_recursive(obj, depth):
        if isinstance(obj, dict):
            for key, val in obj.items():
                if not isinstance(key, str):
                    raise ValueError(f'Key {key} at depth {depth} is not str type')
                _validate_recursive(val, depth + 1)
        else:
            if depth != expected_depth:
                raise ValueError(f'Depth of object is not {expected_depth} at all locations')
            if not isinstance(obj, expected_dtype):
                raise ValueError(f'Object values do not match dtype {expected_dtype}')

    _validate_recursive(obj, 0)


def convert_dtype(obj, dtype_converters):
    """
    Recursively converts the data types of a nested dict using the provided converters.

    Parameters
    ----------
    obj : dict or any
        The object whose data types need to be converted.
    dtype_converters : list of callable
        A list of conversion functions to apply to the object.

    Returns
    -------
    dict or any
        The object with converted data types.

    Raises
    ------
    ValueError
        If none of the converters can successfully convert the object.
    """
    if not isinstance(obj, dict):
        for converter in dtype_converters:
            try:
                return converter(obj)
            except ValueError:
                continue
        else:
            raise ValueError(f'Could not convert object {obj} with dtypes {dtype_converters}')
    else:
        for key, val in obj.items():
            obj[key] = convert_dtype(val, dtype_converters)
    return obj


def deep_update(obj, other):
    """
    Recursively updates a dictionary with the values from another dictionary.

    Parameters
    ----------
    obj : dict
        The dictionary to update.
    other : dict
        The dictionary with values to update into `obj`.

    Returns
    -------
    dict
        The updated dictionary.
    """
    for key, val in other.items():
        if key in obj and isinstance(obj[key], dict) and isinstance(val, dict):
            obj[key] = deep_update(obj[key], val)
        else:
            obj[key] = val
    return obj


class MetadataDict:

    """
    A class for storing and manipulating hierarchical metadata with multi-level dictionaries.

    Attributes
    ----------
    levels : tuple of str
        The levels of metadata hierarchy (e.g., 'group', 'cb', 'chrom', 'other').
    dtype : tuple of type
        The expected data types for the metadata values.
    nlevels : int
        The number of levels in the metadata hierarchy.

    Methods
    -------
    keys()
        Returns top level keys for the metadata object
    values()
        Returns top level values for the metadata object
    get_level_keys(level)
        Returns a sorted list of the unique keys for a particular level
    update(other)
        Updates the metadata dictionary with another `MetadataDict` or dictionary.
    new_like()
        Returns an empty metadata object with the same level/dtype structure
    filter(whitelist, level=0, inplace=True)
        Filters metadata by keeping only the keys in the whitelist at the specified level.
    to_json(precision=5)
        Serializes the metadata dictionary to a JSON-compatible format.
    from_json(obj)
        Creates a `MetadataDict` from a JSON-compatible object.
    """

    allowed_levels = {'group', 'cb', 'chrom', 'other'}

    def __init__(self, levels, dtype, data=None):
        """
        Initializes the MetadataDict with the given levels and data types.

        Parameters
        ----------
        levels : tuple of str
            The levels in the metadata hierarchy.
        dtype : tuple of type
            The expected data types for metadata values.
        data : dict, optional
            The initial data for the metadata. If None, an empty dictionary is used.

        Raises
        ------
        ValueError
            If the levels or data types are invalid.
        """
        if not isinstance(levels, tuple):
            try:
                levels = tuple(levels)
            except TypeError:
                levels = (levels, )
        if not levels:
            raise ValueError('Must have at least one level')
        for lvl in levels:
            if lvl not in self.allowed_levels:
                raise ValueError(f'level "{lvl}" not recognised')
        if len(set(levels)) != len(levels):
            raise ValueError('One or more levels are duplicated')
        self.levels = levels
        self.nlevels = len(levels)
        if not isinstance(dtype, tuple):
            try:
                dtype = tuple(dtype)
            except TypeError:
                dtype = (dtype, )
        for d in dtype:
            if not isinstance(d, type):
                raise ValueError('dtypes should be types')
        self.dtype = dtype
        if data is None:
            self._data = {}
        else:
            validate_data(data, self.nlevels, self.dtype)
            self._data = data

    def __getitem__(self, index):
        """
        Advanced indexing into a metadict object

        Parameters
        ----------
        index : str, Ellipsis, slice(None), or tuple of them
            Indexing pattern, allowing partial selection across levels.

        Returns
        -------
        MetadataDict
            A new MetadataDict object with reduced levels if applicable.
        """
        if not isinstance(index, tuple):
            index = (index,)

        # Validate types
        for idx in index:
            if isinstance(idx, slice):
                if idx != slice(None):
                    raise KeyError('Only full slices (:) are allowed')
            elif not isinstance(idx, (str, type(Ellipsis))):
                raise TypeError(
                    f'Invalid index type {type(idx).__name__}: only str, Ellipsis, or full slices (:) are allowed.'
                )

        # Expand ellipsis if present
        if Ellipsis in index:
            idx_ellipsis = index.index(Ellipsis)
            n_missing = self.nlevels - (len(index) - 1)
            index = index[:idx_ellipsis] + (slice(None),) * n_missing + index[idx_ellipsis + 1:]

        # if all indices are str, we simply traverse the nested dicts
        if all(isinstance(idx, str) for idx in index):
            selected_data = self._data
            for idx in index:
                selected_data = selected_data[idx]
            if isinstance(selected_data, self.dtype):
                return selected_data
            else:
                return MetadataDict(levels=self.levels[len(index):], dtype=self.dtype, data=selected_data)

        # otherwise, we select the relevant data from the nested structure and return as a new MetadataDict object
        selected_data = {}

        def _select(obj, depth, key_path):
            if depth == len(index):
                # End of path, set value
                sd = selected_data
                for key in key_path[:-1]:
                    sd = sd.setdefault(key, {})
                sd[key_path[-1]] = obj
                return None

            idx = index[depth]
            if isinstance(idx, slice):
                for key, val in obj.items():
                    _select(val, depth + 1, key_path + (key,))
            else:
                # Allow index by key only
                if idx not in obj:
                    raise KeyError(f'Key {idx} not found at level {self.levels[depth]}')
                _select(obj[idx], depth + 1, key_path)

        _select(self._data, 0, ())

        # right pad index with None to retain levels below those indexed
        index = index + (slice(None),) * (self.nlevels - len(index))
        # New levels: drop levels where index was a specific value (not a slice)
        new_levels = [lvl for lvl, idx in zip(self.levels, index) if isinstance(idx, (slice, None))]

        return MetadataDict(levels=new_levels, dtype=self.dtype, data=selected_data)

    def _setitem_tuple(self, index, value):
        n_idx = len(index)
        if n_idx > self.nlevels:
            raise IndexError('Too many indices provided')
        for lvl in index:
            if not isinstance(lvl, str):
                raise ValueError('metadata keys must be strings')
        validate_data(value, self.nlevels - n_idx, self.dtype)
        subdict = self._data
        for key in index[:-1]:
            if key not in subdict:
                subdict[key] = {}
            subdict = subdict[key]
        subdict[index[-1]] = value

    def __setitem__(self, index, value):
        if isinstance(index, str):
            validate_data(value, self.nlevels - 1, self.dtype)
            self._data[index] = value
        elif index is Ellipsis:
            validate_data(value, self.nlevels, self.dtype)
            self._data = value
        elif isinstance(index, tuple):
            self._setitem_tuple(index, value)

    def __contains__(self, key):
        return key in self._data

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):

        def _gather_leaves(d, path=()):
            if not isinstance(d, dict):
                yield path, d
            else:
                for k, v in d.items():
                    yield from _gather_leaves(v, path + (k,))

        preview = {}
        leaves = _gather_leaves(self._data)
        for path, val in leaves:
            key = ' -> '.join(repr(k) for k in path)
            if isinstance(val, np.ndarray):
                val_repr = f"numpy.ndarray({val.shape}, dtype={val.dtype})"
            else:
                val_repr = repr(val)
            preview[key] = val_repr
            if len(preview) == 3:
                break
        try:
            next(leaves)
            preview['...'] = '...'
        except StopIteration:
            pass

        dtype = tuple(d.__name__ for d in self.dtype)
        preview_str = ', '.join(f"{k}: {v}" for k, v in preview.items())

        # Here's the changed part
        preview_lines = [f"        {k}: {v}" for k, v in preview.items()]
        preview_str = ',\n'.join(preview_lines)

        return (f"{self.__class__.__name__}(\n"
                f"    levels={self.levels},\n"
                f"    dtype={dtype},\n"
                f"    data_preview={{\n{preview_str}\n"
                f"    }}\n"
                f")")


    def update(self, other):
        """
        Updates the metadata dictionary with another `MetadataDict` or dictionary.

        Parameters
        ----------
        other : MetadataDict or dict
            The metadata dictionary or dictionary to update from.

        Raises
        ------
        ValueError
            If the metadata structures are incompatible.
        """
        if isinstance(other, MetadataDict):
            if self.levels != other.levels:
                raise ValueError('cannot merge metadata, levels do not match')
            if self.dtype != other.dtype:
                raise ValueError('cannot merge metadata, dtypes do not match')
            other = other._data
        if isinstance(other, dict):
            validate_data(other, self.nlevels, self.dtype)
            deep_update(self._data, other)
        else:
            raise ValueError('other must be MetadataDict or dict')

    def _get_level_idx(self, level_name):
        if isinstance(level_name, str):
            level_idx = 0
            for lvl in self.levels:
                if lvl == level_name:
                    break
                level_idx += 1
            else:
                raise ValueError(f'level "{level_name}" not recognised')
        elif isinstance(level_name, int):
            if level_name >= self.nlevels:
                raise ValueError('level index is too deep for metadata')
            level_idx = level_name
        else:
            raise ValueError('level must be either a str or int')
        return level_idx

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def asdict(self):
        """Return a copy of the raw underlying dictionary data"""
        return copy(self._data)

    def get(self, *index, default=None):
        try:
            return self[index]
        except KeyError:
            return default
    
    def get_level_keys(self, level):
        """
        Returns the keys in the metadata object at the specified level.

        Parameters
        ----------
        level : int or str, optional
            The level name or index to filter on.

        Returns
        -------
        list
            The sorted list of unique keys in the metadata object at the specified level
        """
        level = self._get_level_idx(level)
        keys = set()

        def _recursive_search_keys(obj, curr_lvl):
            if curr_lvl == level:
                keys.update(set(obj.keys()))
            else:
                for val in obj.values():
                    _recursive_search_keys(val, curr_lvl + 1)

        _recursive_search_keys(self._data, 0)
        return sorted(keys)

    @classmethod
    def new_like(cls, obj):
        """
        Creates a new `MetadataDict` with the same structure as the given `obj` MetadataDict.

        Parameters
        ----------
        obj : MetadataDict
            The `MetadataDict` instance to copy the structure from.

        Returns
        -------
        MetadataDict
            A new `MetadataDict` instance with the same levels and data types as `obj`.
        """
        return cls(obj.levels, obj.dtype)

    def copy(self):
        '''
        create a copy of a metadata object
        
        Returns
        -------
        MetadataDict
            New copied instance
        '''
        new_instance = self.new_like(self)
        new_instance._data = deepcopy(self._data)
        return new_instance

    def add_level_suffix(self, suffix, level=0, inplace=False):
        '''append a suffix to a specified level of a metadata object'''
        level = self._get_level_idx(level)

        if inplace:
            s = self
        else:
            s = self.copy()

        def _recursive_add_suffix(obj, curr_lvl):
            if curr_lvl == level:
                return {f'{key}_{suffix}': val for key, val in obj.items()}
            else:
                for key in list(obj.keys()):
                    obj[key] = _recursive_add_suffix(obj[key], curr_lvl + 1)
            return obj
        s._data = _recursive_add_suffix(s._data, 0)
        if not inplace:
            return s

    def filter(self, whitelist, level=0, inplace=True):
        """
        Filters the metadata dictionary by retaining only keys in the whitelist at the specified level.

        Parameters
        ----------
        whitelist : set of str
            The set of keys to retain at the specified level.
        level : int or str, optional
            The level name or index to filter on (default is 0).
        inplace : bool, optional
            Whether to modify the current object in-place or return a new filtered object (default is True).

        Returns
        -------
        None or MetadataDict
            If `inplace` is True, modifies the current object. Otherwise, returns a new filtered `MetadataDict`.
        """
        level = self._get_level_idx(level)

        def _recursive_filter(obj, curr_lvl):
            if curr_lvl == level:
                return {k: v for k, v in obj.items() if k in whitelist}
            else:
                for key, val in obj.items():
                    obj[key] = _recursive_filter(val, curr_lvl + 1)
            return obj

        if inplace:
            self._data = _recursive_filter(self._data, 0)
            return None
        else:
            obj = self.new_like(self)
            obj._data = _recursive_filter(deepcopy(self._data), 0)
            return obj

    def transpose_levels(self, reordered_levels, inplace=False):
        """
        Transpose the metadata levels to a new order.

        Parameters
        ----------
        reordered_levels : tuple of str or int
            The new order of the levels, specified by names or indices.
        inplace : bool, optional
            Whether to modify the current object in-place or return a new object (default is False).

        Returns
        -------
        MetadataDict
            The transposed MetadataDict (or None if inplace=True).
        """
        reordered_level_indices = [self._get_level_idx(lvl) for lvl in reordered_levels]
        reordered_levels = [self.levels[idx] for idx in reordered_level_indices]

        if set(reordered_levels) != set(self.levels) or len(reordered_levels) != self.nlevels:
            raise ValueError('reordered_levels must be a permutation of current levels')

        def _gather_key_paths(obj, key_path=()):
            if not isinstance(obj, dict):
                yield key_path, obj
            else:
                for key, val in obj.items():
                    yield from _gather_key_paths(val, key_path + (key,))

        reordered_data = {}
        for key_path, val in _gather_key_paths(self._data):
            reordered_path = tuple(key_path[i] for i in reordered_level_indices)
            sd = reordered_data
            for key in reordered_path[:-1]:
                sd = sd.setdefault(key, {})
            sd[reordered_path[-1]] = val

        if inplace:
            self.levels = reordered_levels
            self._data = reordered_data
            return None
        else:
            return MetadataDict(levels=reordered_levels, dtype=self.dtype, data=reordered_data)

    @classmethod
    def _json_serialise(cls, obj, precision):
        if isinstance(obj, dict):
            json_serialisable = {}
            for key, val in obj.items():
                json_serialisable[key] = cls._json_serialise(val, precision)
        elif isinstance(obj, (list, tuple, set, frozenset, np.ndarray)):
            json_serialisable = []
            for val in obj:
                json_serialisable.append(cls._json_serialise(val, precision))
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

    def to_json(self, precision=5):
        """
        Serializes the metadata dictionary to a JSON-compatible format.

        Parameters
        ----------
        precision : int, optional
            The number of decimal places to use when serializing floating-point values (default is 5).

        Returns
        -------
        dict
            A JSON-compatible dictionary representing the metadata.
        """
        return {
            'levels': self.levels,
            'dtype': [dtype.__name__ for dtype in self.dtype],
            'data': self._json_serialise(self._data, precision=precision)
        }

    @classmethod
    def from_json(cls, obj):
        """
        Creates a `MetadataDict` instance from a JSON-compatible dictionary.
        The dictionary must have the keys "levels", "dtype", and "data"

        Parameters
        ----------
        obj : dict
            The JSON-compatible dictionary to create the `MetadataDict` from.

        Returns
        -------
        MetadataDict
            A new `MetadataDict` instance created from the JSON data.

        Raises
        ------
        ValueError
            If the JSON object is incorrectly formatted.
        """
        try:
            levels = obj['levels']
            dtype = tuple(METADATA_DTYPES[d] for d in obj['dtype'])
            data = obj['data']
        except KeyError:
            raise ValueError('Metadata json is incorrectly formatted')

        dtype_converters = tuple(d if d is not np.ndarray else np.array for d in dtype)
        data = convert_dtype(data, dtype_converters)
        return cls(levels=levels, dtype=dtype, data=data)