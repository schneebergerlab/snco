from collections import defaultdict


def dummy_grouper(records, dummy_name='ungrouped'):
    def _dummy_grouper(cb):
        return dummy_name
    return _dummy_grouper


def genotype_grouper(records):
    genotypes = records.metadata.get('genotypes')
    if genotypes is None:
        raise ValueError('Genotype metadata not present, cannot group by genotype')
    def _geno_grouper(cb):
        try:
            geno = ':'.join(sorted(genotypes[cb]['genotype']))
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
        self.group_mapping = defaultdict(list)
        for cb in records.barcodes:
            g = grouper(cb)
            if g is not None:
                self.group_mapping[g].append(cb)
        self._items = iter(self.group_mapping.items())

    def __iter__(self):
        self._items = iter(self.group_mapping.items())
        return self

    def __next__(self):
        g, g_cb = next(self._items)
        g_obj = self._records.filter(g_cb, inplace=False)
        return g, g_obj

    def apply(self, func, **kwargs):
        combined = self._records.new_like(self._records)
        for _, group in self:
            group_transformed = func(group, **kwargs)
            combined.merge(group_transformed, inplace=True)
        return combined
