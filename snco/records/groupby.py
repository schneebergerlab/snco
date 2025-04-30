import inspect
from collections import defaultdict
from functools import partial


def dummy_grouper(records, dummy_name='ungrouped'):
    """
    Returns a grouper function that assigns all records to a single group.

    Parameters
    ----------
    records : BaseRecords
        The records object (unused but included for API consistency).
    dummy_name : str, default='ungrouped'
        The name of the group to assign all records to.

    Returns
    -------
    callable
        A function that takes a cell barcode and returns `dummy_name`.
    """
    def _dummy_grouper(cb):
        return dummy_name
    return _dummy_grouper


def genotype_grouper(records):
    """
    Returns a grouper function that groups records by genotype.

    Parameters
    ----------
    records : BaseRecords
        The records object containing metadata with genotype information.

    Returns
    -------
    callable
        A function that maps each cell barcode to a genotype string.

    Raises
    ------
    ValueError
        If the `genotypes` metadata is not present in the Records object.
    KeyError
        If a cell barcode is not found in the genotypes metadata.
    """
    genotypes = records.metadata.get('genotypes')
    if genotypes is None:
        raise ValueError('Genotype metadata not present, cannot group by genotype')
    def _geno_grouper(cb):
        try:
            geno = ':'.join(sorted(genotypes[cb]))
        except KeyError:
            raise KeyError(f'Cell barcode {cb} not present in genotypes metadata')
        return geno
    return _geno_grouper


class RecordsGroupyBy:
    '''
    class for split-apply-combine operations on BaseRecords, inspired by pandas groupby
    '''

    default_groupers = {
        'none': dummy_grouper,
        'genotype': genotype_grouper,
    }

    def __init__(self, records, grouper):
        self._records = records
        if not callable(grouper):
            if isinstance(grouper, dict):
                grouper = grouper.get
            elif isinstance(grouper, str):
                try:
                    grouper = self.default_groupers[grouper](records)
                except KeyError:
                    raise ValueError(f'grouper "{grouper}" not recognised')
            else:
                raise ValueError('grouper should be dict or callable')
        else:
            func_signature = inspect.signature(grouper)
            additional_parameters = func_signature.parameters 
            additional_param_names = list(additional_parameters)[1:] # first argument is always cell barcode
            func_kwargs = {}
            for param in additional_param_names:
                if param == 'records':
                    func_kwargs[param] = records
                elif param in records.metadata:
                    func_kwargs[param] = records.metadata[param]
            grouper = partial(grouper, **func_kwargs)
        group_mapping = defaultdict(list)
        for cb in records.barcodes:
            g = grouper(cb)
            if g is not None:
                group_mapping[g].append(cb)
        # convert to dict for __getitem__ to work
        self.group_mapping = dict(group_mapping)
        self._items = iter(self.group_mapping.items())

    def __iter__(self):
        self._items = iter(self.group_mapping.items())
        return self

    def __next__(self):
        g, g_cb = next(self._items)
        g_obj = self._records.filter(g_cb, inplace=False)
        return g, g_obj

    def __getitem__(self, key):
        g_cb = self.group_mapping[key]
        return self._records.filter(g_cb, inplace=False)

    def __len__(self):
        return len(self.group_mapping)

    def apply(self, func, **kwargs):
        """
        Applies a function to each group and combines the results.

        Parameters
        ----------
        func : callable
            A function that takes a BaseRecords object (group) and returns a transformed BaseRecords object.
        **kwargs
            Additional keyword arguments to pass to the function.

        Returns
        -------
        BaseRecords
            A new BaseRecords object containing the merged result of applying `func` to each group.
        """
        combined = self._records.new_like(self._records)
        for _, group in self:
            group_transformed = func(group, **kwargs)
            combined.merge(group_transformed, inplace=True)
        return combined
